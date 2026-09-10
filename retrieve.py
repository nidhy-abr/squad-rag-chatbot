from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="squad_passages")

def retrieve_passages(question, k=3):
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=k
    )

    return results["documents"][0]

if __name__ == "__main__":
    question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    passages = retrieve_passages(question)

    for i, passage in enumerate(passages):
        print(f"--- Passage {i + 1} ---")
        print(passage)
        print()
