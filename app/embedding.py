"""
embedding.py

Converts our text chunks into vectors (embeddings) and stores them in
Chroma, our vector database.
"""

import os
import chromadb
from google import genai
from dotenv import load_dotenv

from chunking import chunk_corpus
from api_utils import call_with_retry

load_dotenv()


EMBEDDING_MODEL = "gemini-embedding-2-preview"


CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

COLLECTION_NAME = "docs"


def get_client():
    
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """
    Sends a batch of texts to Gemini's embedding model and returns their
    vectors. Retries automatically on transient errors via call_with_retry
    (see api_utils.py) , this used to have its own inline retry logic,
    now shared with retrieval.py's generation calls too, since both can
    hit the same kinds of transient failures (rate limits, server overload).
    """
    
    def do_call():
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [embedding.values for embedding in result.embeddings]

    
    return call_with_retry(do_call, bucket="embedding", min_interval_seconds=2)


def build_index(corpus_dir: str):
    
    """


    Full pipeline: chunk the corpus -> embed every chunk -> store in Chroma.

    WHY WE EMBED ONE CHUNK AT A TIME (not batched, despite the extra calls):

    originally batched 20 chunks per embed_content call for efficiency.

    In practice, we hit a case where the API returned a mismatched number

    of embeddings for a batch (e.g. 1 embedding back for 20 texts sent) -

    a real, observed inconsistency in the batch response contract. Rather

    than build fragile guessing logic around batch sizes, we embed one

    chunk per call: slower, but each call's result is trivially verifiable

    (exactly one embedding for exactly one input), eliminating this whole

    class of mismatch.


    """

    client = get_client()

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    chunks = chunk_corpus(corpus_dir)
    print(f"Chunked corpus into {len(chunks)} chunks. Embedding now...")

    existing_ids = set(collection.get()["ids"])
    skipped = 0

    for i, c in enumerate(chunks):
        chunk_id = f"{c['source_file']}::{c['chunk_index']}"

        if chunk_id in existing_ids:
            skipped += 1
            continue

        embed_input = f"{c['heading']}\n\n{c['text']}"
        vectors = embed_texts(client, [embed_input])

        if len(vectors) != 1:
            raise RuntimeError(f"Expected 1 embedding for chunk {i}, got {len(vectors)}.")

        metadata = {
            "source_file": c["source_file"],
            "library": c["library"],
            "heading": c["heading"],
        }
        collection.upsert(ids=[chunk_id], embeddings=vectors, documents=[c["text"]], metadatas=[metadata])

        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"  Embedded and stored {i + 1}/{len(chunks)} chunks ({skipped} skipped, already done)")

    print(f"Done. Collection now has {collection.count()} items.")


if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "corpus")
    build_index(corpus_path)
