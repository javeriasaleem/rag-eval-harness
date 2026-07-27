"""
retrieval.py

The core RAG query pipeline: given a question, find the most relevant
stored chunks, then ask Gemini to answer using ONLY that retrieved context.

WHY THIS FILE EXISTS:
embedding.py built the knowledge base (chunks -> vectors -> stored in
Chroma). This file is what actually USES that knowledge base to answer a
question - the "retrieval" and "generation" halves of Retrieval-Augmented
Generation.

THE TWO-STEP FLOW, AND WHY BOTH STEPS MATTER:
1. RETRIEVE: embed the incoming question the same way we embedded the
   chunks, then ask Chroma for the top-k stored vectors closest to it.
   This is a similarity search in "meaning space" - not a keyword match.
2. GENERATE: hand those retrieved chunks to Gemini along with the question,
   and explicitly instruct it to answer ONLY from the provided text. This
   is the step that turns "here are some possibly-relevant paragraphs"
   into an actual natural-language answer - but it's also the step where
   hallucination risk lives, which is exactly why Step 5 (evaluation) will
   scrutinize this output separately from retrieval quality.
"""

import os
from google import genai
from dotenv import load_dotenv

from embedding import get_client, embed_texts, CHROMA_PATH, COLLECTION_NAME
from api_utils import call_with_retry
import chromadb

load_dotenv()

GENERATION_MODEL = "gemini-3.5-flash"

# WHY THIS EXACT PROMPT WORDING MATTERS:
# The instruction to explicitly say "not found in the provided context" is
# the single most important line in this whole file for reducing
# hallucination. Without it, the model defaults to its own trained
# knowledge to "be helpful" when the retrieved context doesn't fully
# answer the question - which is exactly the failure mode our
# no_answer_in_corpus test questions are designed to catch. This one
# instruction is a large part of what separates "a chatbot that happens to
# have some context" from an actual grounded RAG system.
SYSTEM_PROMPT = """You are answering questions using ONLY the provided context below.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say exactly: "Not found in the provided context."
- Do not guess or fill gaps with plausible-sounding information.
- Keep answers concise and factual.
"""


def retrieve(client, question: str, top_k: int = 5) -> list[dict]:
    """
    Embeds the question and returns the top_k closest stored chunks from
    Chroma, along with their metadata and similarity distance.

    WHY top_k=5 (not 1, not 10): a single chunk (top_k=1) risks missing the
    answer if it's split awkwardly across two chunks despite our overlap
    handling. Too many chunks (top_k=10) dilutes the context the LLM sees
    with less-relevant material, increasing both cost and hallucination
    risk (more irrelevant text = more chances for it to latch onto the
    wrong thing). 3 is a reasonable middle ground for a corpus this size.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query_vector = embed_texts(client, [question])[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved_chunks


def generate_answer(client, question: str, retrieved_chunks: list[dict]) -> str:
    """
    Sends the question + retrieved chunks to Gemini's generation model,
    instructed to answer strictly from the provided context.
    """
    # We label each chunk with its source in the prompt itself - this lets
    # the model (and later, us) see exactly which file each piece of
    # context came from, supporting traceability/citations.
    context_blocks = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"]["source_file"]
        context_blocks.append(f"[Source: {source}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    def do_call():
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        return response.text

    return call_with_retry(do_call, bucket="generation", min_interval_seconds=13)

def answer_question(question: str, top_k: int = 5) -> dict:
    """
    The full pipeline, callable from our FastAPI endpoint (next step):
    retrieve -> generate -> return everything needed for both the user
    and our evaluation harness (answer, sources, raw retrieved chunks).
    """
    client = get_client()

    retrieved_chunks = retrieve(client, question, top_k=top_k)
    answer = generate_answer(client, question, retrieved_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [c["metadata"]["source_file"] for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    # Manual sanity check - ask it something we know the corpus can answer,
    # and something we know it can't, and eyeball both results.
    print("=== Test 1: should be answerable ===")
    client = get_client()
    chunks = retrieve(client, "What algorithm does FastAPI recommend for signing a JWT?", top_k=5)
    print("--- Retrieved chunks (debug) ---")
    for c in chunks:
        print(f"Source: {c['metadata']['source_file']} | distance: {c['distance']:.4f}")
        print(f"Text: {c['text'][:400]}")
        print("...")
    answer = generate_answer(client, "What algorithm does FastAPI recommend for signing a JWT?", chunks)
    print(f"\nFinal Answer: {answer}")

    print("\n=== Test 2: should say 'not found' ===")
    chunks2 = retrieve(client, "How do you configure FastAPI for Kubernetes autoscaling?", top_k=5)
    print("--- Retrieved chunks (debug) ---")
    for c in chunks2:
        print(f"Source: {c['metadata']['source_file']} | distance: {c['distance']:.4f}")
        print(f"Text: {c['text'][:200]}")
        print("...")
    answer2 = generate_answer(client, "How do you configure FastAPI for Kubernetes autoscaling?", chunks2)
    print(f"\nFinal Answer: {answer2}")