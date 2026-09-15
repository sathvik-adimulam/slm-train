import ast
import io
import os
import re
import tokenize
import warnings
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from jupytext import reads
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# HuggingFace authentication
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)



def _protected_lines(source: str) -> set[int]:
    """
    Return 1-indexed line numbers that fall inside a multi-line string token
    (e.g. a triple-quoted string body). These lines must never be treated as
    magic lines, even if their stripped text starts with '%' or '!'.
    """
    protected: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.STRING:
                start_row, _ = tok.start
                end_row, _ = tok.end
                if end_row > start_row:
                    # start_row is not protected since it also contains real
                    # code before the string literal begins (e.g. `s = """`).
                    protected.update(range(start_row + 1, end_row + 1))
    except tokenize.TokenError:
        # Tokenizing failed; fall back to no protection and let ast.parse's
        # error surface normally.
        pass
    return protected


def _is_magic_line(line: str) -> bool:
    """Check whether a line looks like a Jupyter magic (%, %%, or !)."""
    stripped = line.lstrip()
    return (
        stripped.startswith("%%")
        or stripped.startswith("%")
        or stripped.startswith("!")
    )

#Remove magic methods from jupyter cell
def strip_magics(source: str) -> str:
    lines = source.splitlines(keepends=True)
    if not lines:
        return source

    # A '%%' on the first non-blank line is a *cell* magic: by Jupyter
    # convention it consumes the entire remainder of the cell as its
    # argument, which is not Python source at all.
    for line in lines:
        if line.strip() == "":
            continue
        if line.lstrip().startswith("%%"):
            return ""
        break

    protected = _protected_lines(source)

    while True:
        candidate = "".join(lines)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError as err:
            lineno = err.lineno
            if lineno is None or lineno > len(lines):
                raise
            idx = lineno - 1
            line = lines[idx]

            if lineno in protected or not _is_magic_line(line):
                # Not attributable to a magic: a real syntax error.
                raise SyntaxError

            # Preserve the trailing newline so subsequent line numbers
            # are unaffected.
            lines[idx] = "\n" if line.endswith("\n") else ""

#validate syntax with AST
def validate_cell(source: str) -> tuple[bool, str]:
    try:
        cleaned = strip_magics(source)
        return True, cleaned
    except SyntaxError:
        return False, None

#Clean jupyter cell
def clean_cell(source: str) -> str | None:
    is_valid, cleaned = validate_cell(source)
    if not is_valid:
        return None
    if cleaned.strip() == "":
        return None
    return cleaned



cell_pattern = re.compile(
    r"(<jupyter_start>|<jupyter_text>|<jupyter_code>|<jupyter_output>|<empty_output>)"  # Group 1 (Tag)
    r"(.*?)"  # Group 2 (Content)
    r"(?=<jupyter_start>|<jupyter_text>|<jupyter_code>|<jupyter_output>|<empty_output>|$)",  # Lookahead
    re.DOTALL,
)


def process_jupyter_structured(file):
    """Extract and clean code cells from a tagged jupyter-structured sample."""
    notebook = file.get("content", "")
    if not notebook:
        return {"content": "", "status": 0}

    cells = cell_pattern.finditer(notebook)
    chunks = []
    for cell in cells:
        cell_type = cell.group(1)
        cell_content = cell.group(2)

        if cell_type == "<jupyter_code>":
            cleaned_code = clean_cell(cell_content)
            if cleaned_code is not None:
                chunks.append(cleaned_code)
            else:
                return {"content": "", "status": 0}

    return {"content": "\n\n".join(chunks), "status": 1}


def process_jupyter_scripts(file):
    """Extract and clean code cells from a jupytext percent-format script."""
    if file.get("lang") != "python":
        return {"content": "", "status": 0}
    content = file["content"]
    if not content:
        return {"content": "", "status": 0}
    try:
        chunks = []
        nb = reads(content, fmt="py:percent")

        for cell in nb.cells:
            if cell.cell_type == "code":
                cleaned = clean_cell(cell.source)
                if cleaned is not None:
                    chunks.append(cleaned)
                else:
                    return {"content": "", "status": 0}

        return {"content": "\n\n".join(chunks), "status": 1}
    except Exception:
        return {"content": "", "status": 0}


def process_py(file):
    """Validate a plain .py file's syntax with ast.parse."""
    content = file.get("content", "")
    if not content:
        return {"content": "", "status": 0}
    try:
        ast.parse(content)
        return {"content": content, "status": 1}
    except Exception:
        return {"content": "", "status": 0}


# =============================================================================
# Streaming subset builder
# =============================================================================


def gen(ds, target_size, output_dir):
    """
    Yield {"content": text} samples from a streaming dataset `ds` until the
    cumulative byte size reaches `target_size`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def gen_():
        text_size = 0
        with tqdm(ds, desc="Processed 0 bytes") as pbar:
            for sample in pbar:
                text = sample.get("text", "")
                sample_size = len(text.encode())

                if text_size + sample_size > target_size:
                    break

                text_size += sample_size
                pbar.set_description(f"Processed {text_size} bytes")
                yield {"content": text}

    return gen_


# =============================================================================
# Pipeline
# =============================================================================

if __name__ == "__main__":
    num_workers = os.cpu_count() or 1

    # Process jupyter notebooks
    ds_jupyter_structured = load_dataset(
        "bigcode/starcoderdata",
        data_dir="jupyter-structured-clean-dedup",
        split="train",
        num_proc=num_workers,
    )

    output_dir_jupyter_structured = Path("data/jupyter-structured-clean-dedup")
    output_dir_jupyter_structured.mkdir(parents=True, exist_ok=True)

    processed_ds_jupyter_structured = ds_jupyter_structured.map(
        process_jupyter_structured, num_proc=num_workers
    ).filter(lambda x: x.get("status"), num_proc=num_workers)
    processed_ds_jupyter_structured.save_to_disk(output_dir_jupyter_structured)

    ds_jupyter_scripts = load_dataset(
        "bigcode/starcoderdata",
        data_dir="jupyter-scripts-dedup-filtered",
        split="train",
        num_proc=num_workers,
    )
    output_dir_jupyter_scripts = Path("data/jupyter-scripts-dedup-filtered")
    output_dir_jupyter_scripts.mkdir(parents=True, exist_ok=True)
    processed_ds_jupyter_scripts = ds_jupyter_scripts.map(
        process_jupyter_scripts, num_proc=num_workers
    ).filter(lambda x: x.get("status"), num_proc=num_workers)
    processed_ds_jupyter_scripts.save_to_disk(output_dir_jupyter_scripts)

    # Process python files
    ds_py = load_dataset(
        "bigcode/starcoderdata", data_dir="python", split="train", num_proc=num_workers
    )
    output_dir_py = Path("data/python")
    output_dir_py.mkdir(parents=True, exist_ok=True)

    processed_ds_py = ds_py.map(process_py, num_proc=num_workers).filter(
        lambda x: x.get("status"), num_proc=num_workers
    )

    processed_ds_py.save_to_disk(output_dir_py)

    target_size_text = 9 * 1024**3
    target_size_math = 4.5 * 1024**3

    # Download and subset the text corpus (FineWeb-Edu) up to target_size_text
    print("Processing text dataset...")
    output_dir_text = Path("data/text")
    output_dir_text.mkdir(parents=True, exist_ok=True)
    ds_text = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    text_subset = Dataset.from_generator(gen(ds_text, target_size_text, "data/text"))
    text_subset.save_to_disk(output_dir_text)

    # Download and subset the math corpus (OpenWebMath) up to target_size_math
    print("Processing math dataset...")
    ds_math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    math_subset = Dataset.from_generator(gen(ds_math, target_size_math, "data/math"))
    text_subset.save_to_disk("data/math")
