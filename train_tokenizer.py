from datasets import load_from_disk
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

regex_pattern = r"’(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s~"
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(pattern=Regex(regex_pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ]
)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=50304,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)


ds_py = load_from_disk("data/python")
ds_jupyter_scripts = load_from_disk("data/jupyter-scripts-dedup-filtered/")
ds_jupyter_struct = load_from_disk("data/jupyter-structured-clean-dedup")
ds_text = load_from_disk("data/text")
ds_math = load_from_disk("data/math")


def fetch_subsets(datasets, byte_limit, prop):
    assert sum(prop) == 1
    subset_byte_limits = [p * byte_limit for p in prop]
    for subset_byte_limit, ds in zip(subset_byte_limits, datasets):
        ds = ds.shuffle()
        b = 0
        for sample in ds:
            if b >= subset_byte_limit:
                break
            content = sample.get("content", "")
            content_bytes = len(content.encode())
            b += content_bytes
            yield content


def train_tokenizer():
    ds_list = [ds_py, ds_jupyter_scripts, ds_jupyter_struct, ds_text, ds_math]
    iterator = fetch_subsets(ds_list, 6e9, [0.35, 0.175, 0.175, 0.2, 0.1])
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))],
    )
    tokenizer.save("tokenizer.json")


if __name__ == "__main__":
    train_tokenizer()
