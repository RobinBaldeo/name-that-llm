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

## SVM member (4-view TF-IDF + LinearSVC)

CV is GroupKFold(5) on `prompt_id` (`data/folds.csv`, created on first run by `ntl.data`).
Vectorizers are fit inside each fold. Score columns always follow `src/ntl/labels.py`.

| view | text | analyzer | n-grams | weight (default → best) |
|---|---|---|---|---|
| `char` | raw | char | 3–5 | 1.0 |
| `word` | raw | word (+punct tokens) | 1–3 | 0.7 |
| `shape` | `X, x x—x x d x—x.` | char | 2–6 | 0.5 → **0.3** |
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

### Results

5-fold prompt-grouped CV on train (2,699 texts, 150 prompts). All runs are in W&B group `svm`.

| run | C | CV balanced acc |
|---|---|---|
| 2-view baseline (char 1–5 + word 1–2) | 0.1 | 0.340 ± 0.010 |
| 4-view default | 0.5 | 0.363 ± 0.007 |
| **4-view, `shape` weight 0.3 (best → `configs/svm_best.yaml`)** | **0.5** | **0.370 ± 0.009** |

**C sweep** (4-view default weights): 0.1 → 0.353, 0.25 → 0.359, **0.5 → 0.363**, 1.0 → 0.360.

**View ablations** (C = 0.5). Every view helps; `char` matters most:

| views | CV | | views | CV |
|---|---|---|---|---|
| all four | 0.363 | | | |
| minus `char` | 0.331 | | `char` only | 0.312 |
| minus `func` | 0.355 | | `word` only | 0.323 |
| minus `word` | 0.356 | | `func` only | 0.305 |
| minus `shape` | 0.359 | | `shape` only | 0.213 |

**`shape` weight** (C = 0.5): 0 → 0.359, **0.3 → 0.370**, 0.5 → 0.363, 0.7 → 0.351. The +0.007 over the
default is about one fold-std and was picked from ~15 configs, so part of it may be selection noise.
Other weights between 0.1 and 0.4 haven't been tried.

**Per-class recall** (best run): easiest are gemini-3.1-pro-preview 0.82, claude-opus-5 0.69 and gpt-6-astra 0.62;
hardest are claude-fable-5 0.11, glm-5.3 0.15, glm-5.3-flash 0.15 and gpt-5.6-terra 0.17.
Most errors are between models from the same family (425 of the 569 errors in the top-20 confusion pairs),
led by gemini-3.8-flash ↔ gemini-3.7-flash, gpt-5.6-sol/terra → gpt-5.6-luna and glm-5.3-flash → glm-5.3.

Pseudo-probabilities use T = 0.1 (OOF log-loss 1.951).
