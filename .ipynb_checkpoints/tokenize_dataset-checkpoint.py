import os
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from tokenizers import Tokenizer
from tqdm.auto import tqdm

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")


# Split datasets into train and val splits
ds_list = os.listdir("data")

individual_train_splits = []
individual_val_splits = []

for ds_name in ds_list:
    path = os.path.join("data", ds_name)
    ds = load_from_disk(path)
    split = ds.train_test_split(test_size=0.0005, seed=42, shuffle=True)
    individual_train_splits.append(split["train"])
    individual_val_splits.append(split["test"])

train_dataset = concatenate_datasets(individual_train_splits)
val_dataset = concatenate_datasets(individual_val_splits)


# Tokenize dataset
def tokenize(sample):
    ids = tokenizer.encode(sample["content"]).ids
    return {"ids": ids, "len": len(ids)}


# Tokenize datasets and save as .bin files
def save_tokenized_dataset(ds, split):
    # Create shards dir
    base_dir = Path("shards")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize dataset
    num_workers = os.cpu_count() or 1
    tokenized = ds.map(
        tokenize,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {split} dataset",
        num_proc=num_workers,
    )

    # Save dataset as split.bin
    arr_len = np.sum(tokenized["len"])
    filename = base_dir / f"{split}.bin"
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
    num_shards = 1024

    idx = 0
    for shard_idx in tqdm(range(num_shards), desc=f"writing {str(filename)}"):
        shard = tokenized.shard(
            num_shards=num_shards, index=shard_idx, contiguous=True
        ).with_format("numpy")
        shard_ids = np.concatenate(shard["ids"])
        arr[idx : idx + len(shard_ids)] = shard_ids
        idx += len(shard_ids)
    arr.flush()
    del arr


save_tokenized_dataset(train_dataset, "train")
save_tokenized_dataset(val_dataset, "val")
