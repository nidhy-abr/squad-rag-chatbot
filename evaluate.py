from datasets import load_dataset
from generate import generate_answer
from retrieve import retrieve_passages
import random

dataset = load_dataset("squad")
validation_data = dataset["validation"]

random.seed(42)
sample_indices = random.sample(range(len(validation_data)), 100)

retrieval_hits = 0
answer_hits = 0

for idx in sample_indices:
    example = validation_data[idx]
    question = example["question"]
    correct_context = example["context"]
    correct_answer = example["answers"]["text"][0]

    retrieved = retrieve_passages(question, k=3)
    if correct_context in retrieved:
        retrieval_hits += 1

    generated = generate_answer(question, k=3)
    if correct_answer.lower() in generated.lower():
        answer_hits += 1

    print(f"Q: {question}")
    print(f"Expected: {correct_answer}")
    print(f"Generated: {generated}")
    print()

retrieval_hit_rate = retrieval_hits / len(sample_indices) * 100
answer_accuracy = answer_hits / len(sample_indices) * 100

print(f"Retrieval hit rate: {retrieval_hit_rate:.1f}%")
print(f"Answer accuracy: {answer_accuracy:.1f}%")
