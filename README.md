# Does Typological Distance Affect Cross-lingual Transfer?

[Paper](paper.pdf) · [Reported scores](results/paper_blimp_scores.csv)

Camera-ready research code for a BabyLM 2026 Workshop paper by Pengfei Ren.
The project studies sequential L1-to-L2 training with English, Dutch, and
Chinese under a 50M-effective-word budget per phase.

> **Main finding.** The existing runs do not show a uniform,
> target-independent advantage for the coarse “near-L1” condition. English and
> Dutch behave differently across Transfer Increment (TI) and terminal
> Transfer Benefit (TB); a headroom-normalized TI sensitivity analysis does
> not remove that target dependence. This is a descriptive near/far comparison, not a
> causal estimate of typological distance: source script, lexical/token
> overlap, corpus properties, tokenization, update count, and Chinese data
> allocation remain confounded.

## Before We Start 

Here's a bit about my motivations, the struggles I had, and some thoughts about the work. Feel free to check it out.

(Originally written in Chinese, translated to English by ChatGPT)

- [before-we-start (zh)](prelude-zh.md)
- [before-we-start (en)](prelude-en.md)

## Experiment at a glance

```text
Phase 1: random init -> 50M effective units of EN / NL / ZH -> L1 checkpoint
                                      |
                                      +-> zero-shot BLiMP on unseen L2
                                      |
Phase 2: matching L1 checkpoint -> 50M effective units of EN or NL -> terminal BLiMP

TI = terminal score - same-seed, source-L1 zero-shot score
nTI = TI / (1 - zero-shot score)
TB = terminal score - same-seed, target-only Phase-1 score
```

- Model: GPT-2 Small architecture (12 layers, 768 hidden, 12 heads) with a
  Qwen2.5-0.5B tokenizer; 201.9M parameters, including a 116.5M-parameter
  tied token embedding.
- Conditions: 3 monolingual Phase-1 conditions and 4 transfer conditions.
- Seeds: `42, 123, 456, 789, 1024` (35 runs total).
- Evaluation: `blimp_babylm_filtered`, `blimp_nl`, and `zhoblimp` through the
  BabyLM multilingual evaluation harness.
- Fixed-target comparisons: NL->EN vs ZH->EN, and EN->NL vs ZH->NL.

## Results summary

Means across five seeds. Zero-shot, terminal, and nTI are percentages; TI and
TB are percentage-point differences. TB uses the same-seed target-only
Phase-1 model.

| Target | Condition | Zero-shot | Terminal | TI | nTI | TB |
|---|---|---:|---:|---:|---:|---:|
| English | NL->EN (coarse near) | 54.56 | 64.93 | +10.37 | 22.81 | +4.79 |
| English | ZH->EN (coarse far) | 50.46 | 61.27 | +10.81 | 21.59 | +1.13 |
| Dutch | EN->NL (coarse near) | 49.40 | 73.07 | +23.67 | 46.75 | +2.04 |
| Dutch | ZH->NL (coarse far) | 51.93 | 72.22 | +20.29 | 42.15 | +1.19 |

For English, TI is nearly unchanged across the two source conditions while
terminal/TB differs more. For Dutch, the larger TI contrast is strongly
affected by the opposite zero-shot starting-point difference; terminal scores
differ by only 0.85 points. See the paper for the complete interpretation and
limitations.

## Reproduce the reported arithmetic

The repository includes a transparent transcription of all 35 per-seed scores
from Appendix Table 8. Recompute TI, nTI, TB, same-language baselines, and exploratory
paired diagnostics with:

```bash
python -m src.analyze
# or: bash scripts/analyze.sh
```

This writes `results/paper_analysis.json`. The committed CSV is a manuscript
snapshot, **not raw evaluation output**; it audits the arithmetic but does not
independently reproduce training or evaluation.

## Train and evaluate

The scripts assume Linux, a CUDA GPU, and network access to Hugging Face and
the BabyLM evaluation repository.

```bash
git clone https://github.com/Nonepf/multilang-babylm.git
cd multilang-babylm

bash scripts/setup.sh
bash scripts/train_phase1.sh
bash scripts/train_phase2.sh
bash scripts/evaluate.sh --blimp_only
```

Before a full rerun, record the exact dataset revisions, the
`babylm-eval` commit, package versions, CUDA/PyTorch versions, and generated
`run_meta.json` files. These provenance artifacts are required for a strict
replication but are not currently committed.

### Data allocation and update counts

| Language | Byte Premium used | Counting units per phase | Estimated Qwen tokens | Updates |
|---|---:|---:|---:|---:|
| English | 1.0000 | 50.0M words | ~60.0M | 916 |
| Dutch | 1.0516 | 47.5M words | ~80.8M | 1,234 |
| Chinese | 0.9894 | 50.5M characters | ~35.4M | 540 |

The code reproduces the allocation actually used in the paper. It is not a
dimensionally exact byte-budget match: the manuscript estimates that an
approximately aligned Chinese allocation would have been 82.5M characters,
not 50.5M. This exposure and update-count mismatch is an unresolved confound.

### Tokenizer diagnostics

TWR constants in `src/config.py` are reported Phase-0 diagnostics, not values
recomputed by setup. To measure TWR and observed token-ID overlap on explicit,
hash-recorded text samples:

```bash
python -m src.tokenizer_diagnostics \
  --text en=path/to/en.txt \
  --text nl=path/to/nl.txt \
  --text zh=path/to/zh.txt
```

No token-overlap result is claimed in the paper or committed here. Any overlap
number is corpus-sample-dependent and cannot isolate typological causality.

### Historical vocabulary caveat

Repository history shows that the reported checkpoints were originally
trained with 151,643 embedding rows and later resized to 151,665 rows for full
Qwen tokenizer compatibility. Current fresh-training code uses 151,665 rows
from initialization. The 22-row difference is small but means the current code
is not a bit-for-bit reconstruction of the historical training run. The legacy
`scripts/fix_vocab.sh` mutates checkpoints and should only be used on a copy of
the affected historical artifacts, never as a routine setup step.

## Repository layout

```text
src/config.py                 experiment constants and task definitions
src/data.py                   streaming, counting, tokenization, caching
src/model.py                  GPT-2/Qwen model and tokenizer construction
src/train.py                  Phase-1 and Phase-2 training entry point
src/evaluate.py               BabyLM/lm-eval wrapper
src/analyze.py                TI/nTI/TB and per-seed arithmetic
src/tokenizer_diagnostics.py  sample-hashed TWR/token-overlap diagnostics
scripts/                      setup, orchestration, evaluation, analysis
results/paper_blimp_scores.csv  Appendix Table 8 score snapshot
paper.pdf                     camera-ready manuscript
```

## Citation

```bibtex
@inproceedings{ren2026does,
  title={Does Typological Distance Affect Cross-lingual Transfer?},
  author={Pengfei Ren},
  booktitle={EMNLP 2026 Workshop: BabyLM},
  year={2026},
  url={https://openreview.net/forum?id=Mftctkavxf}
}
```

## Acknowledgments and reuse

The paper acknowledges the reviewers and the BabyLM community. Training data,
tokenizers, and the evaluation harness retain their respective upstream terms.
Repository code is released under the [MIT License](LICENSE).
