"""
retrieval.py

 RAG query pipeline: given a question, find the most relevant
stored chunks, then ask Gemini to answer using ONLY that retrieved context.

1. RETRIEVE: embed the incoming question the same way we embedded the
   chunks, then ask Chroma for the top-k stored vectors closest to it.
   
2. GENERATE: hand those retrieved chunks to Gemini along with the question,
   and explicitly instruct it to answer ONLY from the provided text.
"""

import os
from google import genai
from dotenv import load_dotenv

from embedding import get_client, embed_texts, CHROMA_PATH, COLLECTION_NAME
from api_utils import call_with_retry
import chromadb

load_dotenv()

GENERATION_MODEL = "gemini-3.5-flash"


SYSTEM_PROMPT = """You are answering questions using ONLY the provided context below.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say exactly: "Not found in the provided context."
- Do not guess or fill gaps with plausible-sounding information.
- Keep answers concise and factual.
"""


def retrieve(client, question: str, top_k: int = 5) -> list[dict]:
    """
   
    WHY top_k=5 (not 1, not 10): a single chunk (top_k=1) risks missing the
    answer if it's split awkwardly across two chunks despite our overlap
    handling. Too many chunks (top_k=10) dilutes the context the LLM sees
    with less-relevant material, increasing both cost and hallucination
    risk..
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
