import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "vector_db", "chroma")

client = chromadb.PersistentClient(path=CHROMA_PATH)

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_docs(query, intent, n_results=5):
   
    query_embedding = model.encode(query).tolist()
    all_docs = []
    all_distances = []

    # Step 1: Search the intent-specific collection
    try:
        collection = client.get_collection(name=intent)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )
        if results["documents"] and results["documents"][0]:
            all_docs.extend(results["documents"][0])
            all_distances.extend(results["distances"][0])
    except Exception as e:
        print(f"  ⚠ Collection '{intent}' not found: {e}")

    # Step 2: If we got fewer than 3 results, search other collections as fallback
    if len(all_docs) < 3:
        try:
            all_collections = [col.name for col in client.list_collections()]
            for col_name in all_collections:
                if col_name == intent:
                    continue  # Already searched
                try:
                    collection = client.get_collection(name=col_name)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        include=["documents", "distances"]
                    )
                    if results["documents"] and results["documents"][0]:
                        all_docs.extend(results["documents"][0])
                        all_distances.extend(results["distances"][0])
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠ Fallback search error: {e}")

    # Step 3: Sort by distance (lower = more relevant) and de-duplicate
    if not all_docs:
        return []

    # Pair docs with distances and sort
    doc_pairs = list(zip(all_distances, all_docs))
    doc_pairs.sort(key=lambda x: x[0])  # Sort by distance ascending

    # De-duplicate (some chunks may appear in multiple collections)
    seen = set()
    unique_docs = []
    for dist, doc in doc_pairs:
        doc_hash = doc[:100]  # Use first 100 chars as a simple fingerprint
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)
        if len(unique_docs) >= n_results:
            break

    return unique_docs
