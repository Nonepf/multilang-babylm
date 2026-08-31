"""Recompute the camera-ready TI/nTI/TB tables from per-seed BLiMP scores.

The default input is the compact transcription of Appendix Table 8. Raw
lm-eval files should first be reduced to the same explicit CSV schema; keeping
that boundary visible prevents a manuscript transcription from being mistaken
for independently reproduced raw output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from scipy import stats
except ImportError:  # Arithmetic remains runnable in a lightweight environment.
    stats = None

from .config import SEEDS


DIRECTIONS = {
    "nl2en": {"source": "nl", "target": "en", "label": "NL->EN"},
    "zh2en": {"source": "zh", "target": "en", "label": "ZH->EN"},
    "en2nl": {"source": "en", "target": "nl", "label": "EN->NL"},
    "zh2nl": {"source": "zh", "target": "nl", "label": "ZH->NL"},
}
CONTRASTS = {"en": ("nl2en", "zh2en"), "nl": ("en2nl", "zh2nl")}


def load_scores(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["phase"], row["condition"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"Duplicate row: {key}")
            rows[key] = {lang: float(row[f"{lang}_blimp"]) for lang in ("en", "nl", "zh")}
    expected = {
        *(("P1", lang, seed) for lang in ("en", "nl", "zh") for seed in SEEDS),
        *(("P2", direction, seed) for direction in DIRECTIONS for seed in SEEDS),
    }
    missing, extra = expected - rows.keys(), rows.keys() - expected
    if missing or extra:
        raise ValueError(f"Score grid mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return rows


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values)}


def paired_test(near: list[float], far: list[float]) -> dict[str, float]:
    delta = [left - right for left, right in zip(near, far)]
    delta_mean = statistics.fmean(delta)
    sd = statistics.stdev(delta)
    se = sd / math.sqrt(len(delta))
    if se == 0:
        ci_low = ci_high = delta_mean
    else:
        # The experiment always has five paired seeds (df=4). Use SciPy when
        # available and the exact tabulated 0.975 critical value otherwise.
        critical = stats.t.ppf(0.975, len(delta) - 1) if stats else 2.7764451051977987
        ci_low, ci_high = delta_mean - critical * se, delta_mean + critical * se
    t_value = delta_mean / se if se else math.inf
    p_value = float(stats.ttest_rel(near, far).pvalue) if stats else None
    return {
        "mean_difference": delta_mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": t_value,
        "df": len(delta) - 1,
        "p": p_value,
        "cohens_dz": delta_mean / sd if sd else math.inf,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def analyze(rows: dict[tuple[str, str, int], dict[str, float]]) -> dict:
    per_direction: dict[str, dict] = {}
    for condition, spec in DIRECTIONS.items():
        source, target = spec["source"], spec["target"]
        per_seed = []
        for seed in SEEDS:
            terminal = rows[("P2", condition, seed)][target]
            zero_shot = rows[("P1", source, seed)][target]
            target_only = rows[("P1", target, seed)][target]
            ti = terminal - zero_shot
            per_seed.append({
                "seed": seed,
                "zero_shot": zero_shot,
                "terminal": terminal,
                "target_only": target_only,
                "ti": ti,
                "normalized_ti": ti / (1.0 - zero_shot),
                "tb": terminal - target_only,
            })
        per_direction[condition] = {
            **spec,
            "per_seed": per_seed,
            "summary": {
                metric: mean_sd([row[metric] for row in per_seed])
                for metric in (
                    "zero_shot",
                    "terminal",
                    "target_only",
                    "ti",
                    "normalized_ti",
                    "tb",
                )
            },
        }

    contrasts = {}
    for target, (near_name, far_name) in CONTRASTS.items():
        tests = {}
        for metric in ("zero_shot", "terminal", "ti", "normalized_ti", "tb"):
            near = [row[metric] for row in per_direction[near_name]["per_seed"]]
            far = [row[metric] for row in per_direction[far_name]["per_seed"]]
            tests[metric] = paired_test(near, far)
        raw_p = [tests[m]["p"] for m in tests]
        adjusted = holm_adjust(raw_p) if all(p is not None for p in raw_p) else [None] * len(raw_p)
        for metric, p_holm in zip(tests, adjusted):
            tests[metric]["p_holm_within_target"] = p_holm
        contrasts[target] = {
            "near": near_name,
            "far": far_name,
            "tests": tests,
            "note": "Exploratory paired tests; the manuscript reports descriptive results.",
        }

    native = {
        lang: mean_sd([rows[("P1", lang, seed)][lang] for seed in SEEDS])
        for lang in ("en", "nl", "zh")
    }
    return {
        "scale": "proportion (multiply by 100 for percentage points)",
        "seeds": SEEDS,
        "native_phase1": native,
        "directions": per_direction,
        "contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("results/paper_blimp_scores.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_analysis.json"))
    args = parser.parse_args()

    report = analyze(load_scores(args.scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Camera-ready arithmetic check")
    for item in report["directions"].values():
        summary = item["summary"]
        print(
            f"  {item['label']}: ZS={summary['zero_shot']['mean']*100:.2f}, "
            f"terminal={summary['terminal']['mean']*100:.2f}, "
            f"TI={summary['ti']['mean']*100:.2f} pp, "
            f"nTI={summary['normalized_ti']['mean']*100:.2f}%, "
            f"TB={summary['tb']['mean']*100:.2f} pp"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
