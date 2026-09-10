from datasets import load_dataset

dataset = load_dataset("squad")

contexts = dataset["train"]["context"]
unique_contexts = list(set(contexts))

print(f"Total rows: {len(contexts)}")
print(f"Unique passages: {len(unique_contexts)}")
