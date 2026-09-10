import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def ensure_vectorstore():
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.get_collection(name="squad_passages")
        return
    except Exception:
        pass

    dataset = load_dataset("squad")
    all_contexts = list(dataset["train"]["context"]) + list(dataset["validation"]["context"])
    unique_contexts = list(set(all_contexts))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_contexts, show_progress_bar=True)

    collection = client.create_collection(name="squad_passages")
    ids = [str(i) for i in range(len(unique_contexts))]

    batch_size = 5000
    for i in range(0, len(unique_contexts), batch_size):
        collection.add(
            documents=unique_contexts[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            ids=ids[i:i + batch_size]
        )

    return
