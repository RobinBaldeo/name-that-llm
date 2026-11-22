"""Data loading and the prompt-grouped CV folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from ntl.labels import LABEL_TO_ID

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_train(data_dir: str | Path = "data") -> pd.DataFrame:
    df = pd.read_csv(resolve(data_dir) / "train.csv")
    unknown = set(df["model"]) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"train.csv has labels not in labels.py: {sorted(unknown)}")
    df["text"] = df["text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_test(data_dir: str | Path = "data") -> pd.DataFrame:
    df = pd.read_csv(resolve(data_dir) / "test.csv")
    df["text"] = df["text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def make_folds(train: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """GroupKFold on prompt_id: every prompt's answers land in exactly one fold."""
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=train.index, dtype=int)
    for k, (_, va_idx) in enumerate(gkf.split(train, groups=train["prompt_id"])):
        fold.iloc[va_idx] = k
    return pd.DataFrame({"id": train["id"], "fold": fold.values})


def load_or_create_folds(
    train: pd.DataFrame, folds_path: str | Path, n_splits: int = 5, seed: int = 42
) -> tuple[pd.Series, str]:
    """Return a fold number per train row (aligned to `train`) and where it came from.

    Reuses an existing folds file; otherwise creates it once. The file is data
    (gitignored); this function is the code that makes it.
    """
    path = resolve(folds_path)
    if path.exists():
        folds = pd.read_csv(path)
        source = f"existing:{folds_path}"
    else:
        folds = make_folds(train, n_splits=n_splits, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(path, index=False)
        source = f"created:{folds_path}(GroupKFold({n_splits}),prompt_id,seed={seed})"
        print(f"[folds] created {path} ({n_splits} prompt-grouped folds)")

    fold = train[["id"]].merge(folds, on="id", how="left")["fold"]
    if fold.isna().any():
        raise ValueError(f"{fold.isna().sum()} train ids missing from {path}; delete it to rebuild")
    fold = fold.astype(int)

    # A prompt split across folds would leak topic into validation.
    per_prompt = pd.DataFrame({"p": train["prompt_id"], "f": fold}).groupby("p")["f"].nunique()
    if (per_prompt > 1).any():
        raise ValueError(f"{path}: some prompt_ids span several folds; it is not prompt-grouped")
    return fold, source


if __name__ == "__main__":
    # python -m ntl.data  -> (re)creates data/folds.csv
    import argparse

    ap = argparse.ArgumentParser(description="Create prompt-grouped CV folds")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/folds.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tr = load_train(args.data_dir)
    f = make_folds(tr, args.n_splits, args.seed)
    resolve(args.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(resolve(args.out), index=False)
    print(f.merge(tr[["id", "prompt_id"]], on="id").groupby("fold").agg(rows=("id", "size"), prompts=("prompt_id", "nunique")))
