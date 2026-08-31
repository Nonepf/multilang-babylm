"""
BabyLM 2026 — Data Pipeline.
Stream from HuggingFace → cache raw text → tokenize → cache tokenized Dataset.

Three-tier caching:
  1. Raw text:     {cache_dir}/{lang}_{N}M.txt
  2. Tokenized:    {cache_dir}/{lang}_{N}M_tokenized/   (HuggingFace Dataset)
  3. Training:     loads directly from tokenized cache

Subsequent seeds reuse the same cached data.
"""
import os
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from .config import LANG_CONFIG, MAX_SEQ_LENGTH


def count_units(text: str, lang: str) -> int:
    """
    Count native units in text.
    English/Dutch: whitespace-delimited words.
    Chinese: characters (excluding spaces and newlines).
    """
    if lang == "zh":
        return len(text.replace(" ", "").replace("\n", "").replace("\r", ""))
    return len(text.split())


def _stream_and_cache_raw(
    lang: str,
    native_units: int,
    cache_dir: str,
) -> str:
    """
    Stream from HuggingFace, concatenate, cache raw text.
    Returns path to cached raw text file.
    """
    cfg = LANG_CONFIG[lang]
    raw_path = os.path.join(cache_dir, f"{lang}_{native_units // 1_000_000}M.txt")

    if os.path.exists(raw_path):
        print(f"  [{cfg['name']}] Raw cache found ({os.path.getsize(raw_path)/1024/1024:.1f} MB)")
        return raw_path

    print(f"  [{cfg['name']}] Streaming {native_units:,} {cfg['unit']} from {cfg['dataset']}...")
    ds = load_dataset(cfg["dataset"], split="train", streaming=True)

    texts = []
    total = 0
    for sample in ds:
        text = sample["text"]
        if not text or len(text.strip()) < 3:
            continue
        n = count_units(text, lang)
        if n == 0:
            continue
        texts.append(text)
        total += n
        if total >= native_units:
            break

    full_text = "\n\n".join(texts)
    os.makedirs(cache_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    actual = count_units(full_text, lang)
    size_mb = os.path.getsize(raw_path) / 1024 / 1024
    print(f"  [{cfg['name']}] Cached: {actual:,} {cfg['unit']} ({size_mb:.1f} MB, {len(texts)} docs)")
    return raw_path


def _tokenize_and_cache(
    lang: str,
    native_units: int,
    raw_path: str,
    tokenizer: PreTrainedTokenizer,
    cache_dir: str,
) -> Dataset:
    """
    Tokenize raw text, slice into MAX_SEQ_LENGTH chunks, cache as Dataset.
    Returns the tokenized Dataset.
    """
    cfg = LANG_CONFIG[lang]
    tok_cache = os.path.join(cache_dir, f"{lang}_{native_units // 1_000_000}M_tokenized")

    if os.path.exists(tok_cache):
        print(f"  [{cfg['name']}] Tokenized cache found")
        return Dataset.load_from_disk(tok_cache)

    print(f"  [{cfg['name']}] Reading raw text...")
    with open(raw_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"  [{cfg['name']}] Tokenizing ({len(full_text):,} chars)...")
    tokenized = tokenizer(
        full_text, return_tensors="np", truncation=False, add_special_tokens=False
    )["input_ids"][0]
    n_tokens = len(tokenized)
    print(f"  [{cfg['name']}] {n_tokens:,} tokens")

    # Slice into non-overlapping MAX_SEQ_LENGTH chunks
    chunks = []
    for i in range(0, n_tokens - MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        chunk = tokenized[i : i + MAX_SEQ_LENGTH].tolist()
        chunks.append({"input_ids": chunk, "labels": chunk})

    ds = Dataset.from_list(chunks)
    os.makedirs(tok_cache, exist_ok=True)
    ds.save_to_disk(tok_cache)
    print(f"  [{cfg['name']}] {len(chunks):,} chunks — cached")
    return ds


def get_tokenized_dataset(
    lang: str,
    native_units: int,
    tokenizer: PreTrainedTokenizer,
    cache_dir: str,
) -> Dataset:
    """
    Full pipeline: stream → cache raw → tokenize → cache tokenized → return Dataset.

    Args:
        lang: "en" | "nl" | "zh"
        native_units: number of words (EN/NL) or chars (ZH)
        tokenizer: HuggingFace tokenizer (Qwen2.5)
        cache_dir: directory for raw + tokenized caches

    Returns:
        HuggingFace Dataset with "input_ids" and "labels" columns.
        Each row is one MAX_SEQ_LENGTH chunk.
    """
    raw_path = _stream_and_cache_raw(lang, native_units, cache_dir)
    ds = _tokenize_and_cache(lang, native_units, raw_path, tokenizer, cache_dir)
    return ds


def get_eval_split(ds: Dataset, n: int = 50) -> Dataset:
    """Return a small subset for eval during training."""
    return ds.select(range(min(n, len(ds))))
