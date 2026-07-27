"""
main.py

The FastAPI application -  turns our collection of scripts
(chunking.py, embedding.py, retrieval.py) into an actual running web
service that anything (a browser, curl, another program, eventually our
own evaluation harness) can send HTTP requests to.


Pydantic MODELS FOR REQUEST/RESPONSE (not just raw dicts):
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

     endpoint makes ~2 API calls per question
    """
    try:
        report = run_evaluation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    return report
