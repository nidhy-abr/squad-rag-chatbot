
from datasets import load_dataset

dataset = load_dataset("squad")
print(dataset)
example = dataset["train"][0]
print(example["context"])
print(example["question"])
print(example["answers"])
