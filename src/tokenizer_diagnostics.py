"""Measure corpus TWR and observed token-ID overlap on explicit text files.

This utility does not manufacture a paper result: users must supply the exact
sample files, whose paths and SHA-256 hashes are recorded in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path


def parse_text_arg(value: str) -> tuple[str, Path]:
    try:
        lang, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LANG=PATH") from exc
    if lang not in {"en", "nl", "zh"}:
        raise argparse.ArgumentTypeError("LANG must be en, nl, or zh")
    return lang, Path(path)


def count_units(text: str, lang: str) -> int:
    """Match the paper's counting convention without importing data tooling."""
    if lang in {"en", "nl"}:
        return len(text.split())
    return sum(1 for character in text if not character.isspace())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", type=parse_text_arg, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/tokenizer_diagnostics.json"))
    args = parser.parse_args()

    supplied = dict(args.text)
    if len(supplied) != len(args.text):
        parser.error("each language may be supplied only once")

    # Import model dependencies only after argument parsing so ``--help`` works
    # in a lightweight audit environment without PyTorch/Transformers installed.
    from .model import load_tokenizer

    tokenizer = load_tokenizer()
    report = {
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_length": len(tokenizer),
        "languages": {},
        "pairwise_observed_token_id_overlap": {},
        "warning": "Overlap is sample-dependent and is not a causal estimate of typological distance.",
    }
    observed: dict[str, set[int]] = {}
    for lang, path in supplied.items():
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        units = count_units(text, lang)
        observed[lang] = set(token_ids)
        report["languages"][lang] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "counting_units": units,
            "tokens": len(token_ids),
            "twr": len(token_ids) / units if units else None,
            "unique_token_ids": len(observed[lang]),
        }

    for left, right in combinations(sorted(observed), 2):
        intersection = observed[left] & observed[right]
        union = observed[left] | observed[right]
        report["pairwise_observed_token_id_overlap"][f"{left}-{right}"] = {
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": len(intersection) / len(union) if union else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
