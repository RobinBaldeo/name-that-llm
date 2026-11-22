# name-that-llm

Kaggle "Name That LLM": 18-class authorship attribution, metric = balanced accuracy.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt && pip install -e .
```

Put `train.csv`, `test.csv`, `sample_submission.parquet` in `data/` (gitignored).
W&B credentials go in `.env` (gitignored) as `WANDB_API_KEY=...` (`WB_API_KEY` also works).
Without a key, runs fall back to `WANDB_MODE=offline` — sync later with `wandb sync wandb/offline-run-*`.

## SVM member (4-view TF-IDF + LinearSVC) — see `spec/SPEC_svm.md`

CV is GroupKFold(5) on `prompt_id` (`data/folds.csv`, created on first run by `ntl.data`).
Vectorizers are fit inside each fold. Score columns always follow `src/ntl/labels.py`.

| view | text | analyzer | n-grams | weight |
|---|---|---|---|---|
| `char` | raw | char | 3–5 | 1.0 |
| `word` | raw | word (+punct tokens) | 1–3 | 0.7 |
| `shape` | `X, x x—x x d x—x.` | char | 2–6 | 0.5 |
| `func` | `however , the C — C on C C — C .` | word | 1–3 | 0.5 |

```bash
python -m ntl.svm.cv --config configs/svm.yaml                      # one CV run -> one W&B run (group svm)
python -m ntl.svm.cv --config configs/svm.yaml --override C=0.25 --override views=char,word
python scripts/run_svm_experiments.py --stage all                    # SPEC §5 plan (baseline, default, C sweep, ablations)
python scripts/wandb_summary.py                                      # compare runs without the browser
python scripts/select_best_svm.py                                    # -> configs/svm_best.yaml, alias OOF artifact `best`
python -m ntl.svm.full_fit --config configs/svm_best.yaml            # fit on all train, log model + test predictions
python scripts/select_best_svm.py --full-fit-run-id <id>             # alias model/test artifacts `best` too
python kaggle/svm_submit.py                                          # self-contained Kaggle script (no wandb)
pytest
```

Per-run outputs land in `artifacts/svm/<run_id>/` (gitignored) and are logged as W&B artifacts:
`svm-oof-<run_id>` (`oof_scores.npy`, `oof_ids.csv`, `labels.json`), and for full-fit runs
`svm-model-<run_id>` and `svm-test-<run_id>`. `artifacts/svm/runs.jsonl` mirrors every run locally
(`python scripts/wandb_summary.py --local`).

Pseudo-probabilities for blending: `softmax(scores / T)` with `T = temperature` from `configs/svm_best.yaml`.
