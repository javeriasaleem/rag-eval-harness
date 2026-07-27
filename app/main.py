"""
main.py

The FastAPI application - this is what turns our collection of scripts
(chunking.py, embedding.py, retrieval.py) into an actual running web
service that anything (a browser, curl, another program, eventually our
own evaluation harness) can send HTTP requests to.

WHY WE NEED THIS FILE, SPECIFICALLY:
Everything so far has been "run this Python file directly from the
terminal." That's fine for building and testing piece by piece, but it's
not how a real system gets used - nobody wants to SSH into a server and
run a script by hand every time someone asks a question. FastAPI exposes
our functions as HTTP endpoints, so a request comes in, gets routed to the
right function, and a response goes out - the standard shape of basically
every web API you've ever used.

WHY Pydantic MODELS FOR REQUEST/RESPONSE (not just raw dicts):
Pydantic (which FastAPI uses internally) validates incoming data
automatically. If someone sends a request missing the "question" field,
or sends a number instead of a string, FastAPI rejects it with a clear
422 error BEFORE our code even runs - we don't have to write manual
"if 'question' not in request" checks ourselves. This is a big part of
why FastAPI pairs so naturally with Pydantic.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval import answer_question
from embedding import build_index
from evaluation import run_evaluation
import os

app = FastAPI(
    title="RAG Evaluation Harness",
    description="A RAG pipeline over FastAPI/Pydantic/Starlette docs, with built-in evaluation.",
)


# --- Request/response schemas ---
# WHY DEFINE THESE EXPLICITLY (instead of accepting/returning raw dicts):
# This is "the contract" of our API - anyone reading this code (or hitting
# the auto-generated /docs page FastAPI builds for us) can see exactly
# what shape of data each endpoint expects and returns, without having to
# read the function body.

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the RAG system")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


@app.get("/health")
def health_check():
    """
    A trivial endpoint that just confirms the server is up and responding.

    Returns server status
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Answers a question using retrieval-augmented generation over the
    indexed corpus. Returns the generated answer along with the source
    documents used to produce it.
    """
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Rebuilds the vector index from the corpus directory: chunks all
    documents, generates embeddings, and stores them in Chroma. Intended
    to be called when the corpus changes, not on every request.
    """
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "corpus")

    try:
        build_index(corpus_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    from embedding import CHROMA_PATH, COLLECTION_NAME
    import chromadb
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    return IngestResponse(status="success", chunks_indexed=collection.count())

@app.post("/evaluate")
def evaluate():
    """
    Runs the evaluation harness against test_set.json: every question is
    run through the live RAG pipeline and scored on three dimensions -
    retrieval accuracy, faithfulness (LLM-as-judge), and abstention
    correctness on unanswerable questions.

    Note: this endpoint makes ~2 API calls per question (one generation
    call, one judge call), so a full run takes several minutes. In a
    production deployment this would run asynchronously as a background
    job rather than a blocking request.
    """
    try:
        report = run_evaluation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    return report