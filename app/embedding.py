"""
embedding.py

Converts our text chunks into vectors (embeddings) and stores them in
Chroma, our vector database.

WHY THIS FILE EXISTS:
Computers can't compare "meaning" directly - they compare numbers. An
embedding model turns a piece of text into a list of numbers (a vector)
such that texts with similar MEANING end up as vectors that are close
together in that number-space, even if they don't share the same words.
That's what makes "how do I make my API wait for something" retrieve a
chunk about `Depends`/async, even though the chunk never uses that exact
phrasing - semantic search, not keyword search.

Chroma is where we store these vectors so we can later ask "which stored
vectors are closest to this new question's vector?" - that "closest
vectors" operation is what retrieval actually is, under the hood.
"""

import os
import chromadb
from google import genai
from dotenv import load_dotenv

from chunking import chunk_corpus
from api_utils import call_with_retry

load_dotenv()

# Why gemini-embedding-001 specifically: it's Google's current embedding
# model as of 2026, available on the free tier, and designed for exactly
# this "semantic search" use case (as opposed to gemini-2.5-flash, which is
# a generation model, not an embedding model - they do different jobs).
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Chroma stores its data on disk here, in a persistent client. WHY
# persistent (not in-memory): if we used an in-memory client, we'd have to
# re-embed all 436 chunks every single time we restarted the server, which
# wastes API calls/time/quota for no reason. Persisting to disk means we
# embed once, then reuse the stored vectors indefinitely until the corpus
# changes.
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

COLLECTION_NAME = "docs"


def get_client():
    
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """
    Sends a batch of texts to Gemini's embedding model and returns their
    vectors. Retries automatically on transient errors via call_with_retry
    (see api_utils.py) - this used to have its own inline retry logic,
    now shared with retrieval.py's generation calls too, since both can
    hit the same kinds of transient failures (rate limits, server overload).

    WHY BATCH instead of one-by-one: reduces the NUMBER OF CALLS we make
    (fewer round trips, simpler code) - but importantly, Google's free tier
    still counts EACH TEXT in a batch toward the per-minute request quota,
    not each API call. So batching doesn't dodge the rate limit, it just
    keeps our code simpler.
    """
    def do_call():
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [embedding.values for embedding in result.embeddings]

    # WHY bucket="embedding", min_interval_seconds=2:
    # The usage dashboard showed our embedding calls already exceeding
    # the free tier's tokens-per-minute limit (30K) during heavy use.
    # /evaluate calls this function once per question (25 times, one
    # per retrieve() call) with no prior pacing - this was a real gap:
    # we'd only paced GENERATION calls, not embedding calls. A modest
    # 2s spacing keeps embedding calls well under both its RPM and TPM
    # ceilings without noticeably slowing down a single-question embed.
    return call_with_retry(do_call, bucket="embedding", min_interval_seconds=2)


def build_index(corpus_dir: str):
    
    """


    Full pipeline: chunk the corpus -> embed every chunk -> store in Chroma.

    WHY WE EMBED ONE CHUNK AT A TIME (not batched, despite the extra calls):

    We originally batched 20 chunks per embed_content call for efficiency.

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