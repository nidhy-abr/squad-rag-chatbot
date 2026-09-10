from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

dataset = load_dataset("squad")
all_contexts = list(dataset["train"]["context"]) + list(dataset["validation"]["context"])
unique_contexts = list(set(all_contexts))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(unique_contexts, show_progress_bar=True)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(name="squad_passages")

ids = [str(i) for i in range(len(unique_contexts))]

batch_size = 5000

for i in range(0, len(unique_contexts), batch_size):
    batch_documents = unique_contexts[i:i + batch_size]
    batch_embeddings = embeddings[i:i + batch_size].tolist()
    batch_ids = ids[i:i + batch_size]

    collection.add(
        documents=batch_documents,
        embeddings=batch_embeddings,
        ids=batch_ids
    )

    print(f"Added batch {i} to {i + len(batch_documents)}")

print(f"Stored {collection.count()} passages in Chroma")
