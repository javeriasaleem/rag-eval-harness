from embedding import get_client, embed_texts, CHROMA_PATH, COLLECTION_NAME
import chromadb

client = get_client()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

# Check dimensionality of a fresh query embedding
query_vector = embed_texts(client, ["What algorithm does FastAPI recommend for signing a JWT?"])[0]
print(f"Query vector length: {len(query_vector)}")
print(f"Query vector first 5 values: {query_vector[:5]}")

# Check dimensionality of a stored embedding (peek directly at the collection)
peek = collection.peek(limit=1)
stored_vector = peek["embeddings"][0]
print(f"\nStored vector length: {len(stored_vector)}")
print(f"Stored vector first 5 values: {stored_vector[:5]}")

print(f"\nDimensions match: {len(query_vector) == len(stored_vector)}")