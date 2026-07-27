"""
debug_rank.py - not part of the final project.

Checks where the chunk containing "HS256" actually ranks among ALL 436
stored chunks for our test question - not just whether it made the top 3.
This tells us whether it's a "just barely missed the cutoff" problem
(a top_k tuning issue) or a "completely unrelated, ranked very low"
problem (a deeper embedding/chunking issue).
"""

from embedding import get_client, embed_texts, CHROMA_PATH, COLLECTION_NAME
import chromadb

client = get_client()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

question = "What algorithm does FastAPI recommend for signing a JWT?"
query_vector = embed_texts(client, [question])[0]

# Ask for top 15 instead of top 3, so we can see further down the ranking
results = collection.query(query_embeddings=[query_vector], n_results=15)

target_id_prefix = "fastapi/security-oauth2-jwt.md::7"  # our known correct chunk

print(f"Question: {question}\n")
for rank, (chunk_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0]), start=1):
    marker = "  <-- THIS IS THE CORRECT CHUNK" if chunk_id == target_id_prefix else ""
    print(f"Rank {rank}: {chunk_id} | distance {distance:.4f}{marker}")
