from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="/home/kyan/stage/Datasets/Convert_format_llama/train_llama.json"
)

print(dataset)

print("\nFirst sample:\n")
print(dataset["train"][0])
