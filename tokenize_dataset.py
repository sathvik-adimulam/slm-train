import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from tokenizers import Tokenizer
from tqdm.auto import tqdm

TOKENS_PER_BIN = 100_000_000  # cap per output .bin file
NUM_SHARDS = os.cpu_count() or 1  # one process per shard
NUM_SUBSHARDS = 64  # granularity of streaming within a single worker
DTYPE = np.uint16  # requires vocab_size <= 65536; use np.uint32 otherwise

tokenizer: Tokenizer  # set in __main__; module-global so `tokenize` can see it

# Module-globals for worker processes (initialized via ProcessPoolExecutor initializer)
_file_counter = None
_counter_lock = None


def init_worker(counter, lock):
    global _file_counter, _counter_lock
    _file_counter = counter
    _counter_lock = lock


def tokenize(sample):
    ids = tokenizer.encode(sample["content"]).ids
    return {"ids": ids, "len": len(ids)}


def _compute_file_sizes(total_tokens, tokens_per_bin):
    sizes = []
    remaining = total_tokens
    while remaining > 0:
        sizes.append(min(tokens_per_bin, remaining))
        remaining -= sizes[-1]
    return sizes


def write_shard_bins(shard_idx, shard, out_dir):
    global _file_counter, _counter_lock

    total_tokens = sum(shard["len"])
    if total_tokens == 0:
        return

    file_sizes = _compute_file_sizes(total_tokens, TOKENS_PER_BIN)

    file_idx = 0
    pos_in_file = 0

    def open_memmap(idx: int):
        # Safely increment and fetch the global file counter
        with _counter_lock:
            _file_counter.value += 1

        path = out_dir / f"{_file_counter.value:05d}.bin"
        arr = np.memmap(path, dtype=DTYPE, mode="w+", shape=(file_sizes[idx],))
        return arr

    current_arr = open_memmap(file_idx)

    pbar = tqdm(
        total=total_tokens,
        desc=f"shard {shard_idx:02d}",
        position=shard_idx,
        leave=False,
    )

    for sub_idx in range(NUM_SUBSHARDS):
        sub = shard.shard(
            num_shards=NUM_SUBSHARDS, index=sub_idx, contiguous=True
        ).with_format("numpy")
        if len(sub) == 0:
            continue

        ids = np.concatenate(sub["ids"]).astype(DTYPE)
        ids_pos = 0

        while ids_pos < len(ids):
            space_left = file_sizes[file_idx] - pos_in_file
            take = min(space_left, len(ids) - ids_pos)

            current_arr[pos_in_file : pos_in_file + take] = ids[
                ids_pos : ids_pos + take
            ]
            pos_in_file += take
            ids_pos += take
            pbar.update(take)

            if pos_in_file == file_sizes[file_idx]:
                current_arr.flush()
                del current_arr
                file_idx += 1
                pos_in_file = 0
                if file_idx < len(file_sizes):
                    current_arr = open_memmap(file_idx)

    pbar.close()


def save_tokenized_dataset(ds, split, num_shards):
    base_dir = Path("shards") / split
    base_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize dataset (parallelized internally via num_proc), kept in memory.
    num_workers = os.cpu_count() or 1
    tokenized = (
        ds.map(
            tokenize,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {split} dataset",
            num_proc=num_workers,
        )
        .shuffle(seed=42)
        .flatten_indices()
    )

    shards = [
        tokenized.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in tqdm(
            range(num_shards), desc=f"Preparing {num_shards} shards for processing"
        )
    ]

    print(
        f"Writing {split} bins with {num_shards} parallel processes "
        f"({TOKENS_PER_BIN:,} tokens/bin)"
    )

    # Create shared memory counter and lock across processes
    file_counter = multiprocessing.Value("i", 0)
    counter_lock = multiprocessing.Lock()

    with ProcessPoolExecutor(
        max_workers=NUM_SHARDS,
        initializer=init_worker,
        initargs=(file_counter, counter_lock),
    ) as executor:
        futures = [
            executor.submit(
                write_shard_bins,
                i,
                shards[i],
                base_dir,
            )
            for i in range(num_shards)
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("tokenizer.json")

    ds_list = os.listdir("data")
    print(f"Datasets: {ds_list}")

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

    print("Finished train_test_split")

    save_tokenized_dataset(val_dataset, "val", 1)
    save_tokenized_dataset(train_dataset, "train", NUM_SHARDS)
