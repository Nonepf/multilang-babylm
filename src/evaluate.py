"""
BabyLM 2026 — Evaluation.  Wrap lm-eval, parse results from log, save structured JSON.
"""
import argparse, json, os, re, subprocess, sys, time, warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Respect an explicit HF_DATASETS_OFFLINE setting. Forcing offline mode here
# made a first evaluation fail unless every benchmark artifact was pre-cached.

from .config import EVAL_LANG_GROUPS, BLIMP_TASKS, PROJECT_ROOT

BABYLM_EVAL_DIR = os.path.join(PROJECT_ROOT, "babylm-eval", "multilingual")
TASKS_DIR = os.path.join(BABYLM_EVAL_DIR, "tasks")


def run_lm_eval(model_path: str, task: str, output_dir: str,
                batch_size: int = 16, limit: int = None) -> dict:
    """Run lm-eval, parse results from log output, return dict of scores."""

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "lm_eval_output.log")

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", task,
        "--device", "cuda",
        "--output_path", output_dir,
        "--batch_size", str(batch_size),
        "--num_fewshot", "0",
        "--include_path", TASKS_DIR,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"  [{task}] Running...")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, cwd=BABYLM_EVAL_DIR,
                                stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  [{task}] FAILED (exit={result.returncode})")
        # show last errors
        with open(log_file) as f:
            for line in f.readlines()[-5:]:
                if any(kw in line for kw in ("Error", "Traceback", "FAILED")):
                    print(f"    {line.rstrip()[:200]}")
        return None

    # Parse results from the log
    scores = _parse_scores_from_log(log_file)
    if scores:
        for k, v in scores.items():
            if isinstance(v, dict):
                acc = v.get("acc", {})
                if isinstance(acc, dict):
                    print(f"  [{k}] acc={acc.get('value', '?'):.4f}")
                else:
                    print(f"  [{k}] {v}")

    # Save parsed JSON ourselves (no reliance on lm_eval output)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  [{task}] Saved -> {results_path}")

    return {"task": task, "status": "ok", "scores": scores}


def _parse_scores_from_log(log_file: str) -> dict:
    """Parse lm_eval markdown table from log. Extracts task name + metrics."""
    scores = {}
    with open(log_file) as f:
        lines = f.readlines()

    # Find the metric table: lines look like
    # |task_name|version|filter|n-shot|metric|   |value|   |stderr|
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("| "):
            continue
        # Skip separator lines like |---|---|
        if re.match(r'^\|[-: |]+\|$', line):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        # Expected: ['', 'task', 'version', 'filter', 'n-shot', 'metric',
        #            '', 'value', '', 'stderr', '']
        task_name = parts[1]
        metric = parts[5]
        value_str = parts[7]
        stderr_str = parts[9] if len(parts) > 9 else ""
        # Skip header rows
        if task_name in ("Tasks", "Groups") or metric == "Metric":
            continue
        try:
            val = float(value_str) if value_str and value_str != "" else None
        except ValueError:
            val = value_str
        try:
            se = float(stderr_str) if stderr_str and stderr_str not in ("", "N/A") else None
        except ValueError:
            se = None

        if task_name not in scores:
            scores[task_name] = {}
        scores[task_name][metric] = {"value": val, "stderr": se}

    return scores


def _extract_run_name(model_path: str) -> str:
    m = re.search(r'(phase[12].+)', model_path)
    if m:
        return m.group(1).replace("/", "_")
    return os.path.basename(model_path.rstrip("/"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--langs", default="eng")
    parser.add_argument("--output_dir", default="./results/eval")
    parser.add_argument("--blimp_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (for quick smoke tests)")

    args = parser.parse_args()
    langs = [l.strip() for l in args.langs.split(",")]
    args.model_path = os.path.abspath(args.model_path)

    if not os.path.exists(args.model_path):
        print(f"ERROR: not found: {args.model_path}")
        sys.exit(1)

    name = _extract_run_name(args.model_path)
    eval_dir = os.path.join(args.output_dir, name)

    REVERSE = {"eng": "en", "nld": "nl", "zho": "zh"}

    print(f"\n{'='*60}")
    print(f"  EVAL  {name}")
    print(f"{'='*60}\n")

    results = {}
    t0 = time.time()
    for lc in langs:
        short = REVERSE.get(lc, lc)
        task = BLIMP_TASKS.get(short) if args.blimp_only else EVAL_LANG_GROUPS.get(short)
        if not task:
            continue
        lang_dir = os.path.join(eval_dir, lc)
        results[lc] = run_lm_eval(args.model_path, task, lang_dir, args.batch_size, args.limit)

    elapsed = time.time() - t0
    summary = {"model_path": args.model_path, "model_name": name,
               "date": datetime.now().isoformat(),
               "elapsed_seconds": elapsed, "results": results}
    sp = os.path.join(eval_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Done in {elapsed/60:.1f} min  -> {sp}")


if __name__ == "__main__":
    main()
