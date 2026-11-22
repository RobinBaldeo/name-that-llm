"""Name That LLM - multi-view TF-IDF + LinearSVC submission (self-contained, internet off).

Fits the best CV config on all of train at runtime, predicts test, writes
`submission.parquet` (id, model) and `svm_test_scores.npy` (n_test x 18 decision
scores, LABELS column order) to the working directory.

No wandb, no repo imports: config values are inlined from configs/svm_best.yaml and the
text transforms are copied from src/ntl/svm/views.py (tests/ check both stay in sync).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Config - copied from configs/svm_best.yaml
# (W&B run zllgpx3v "svm-char+word+shape+func-C0.5-shapew0.3", CV balanced acc 0.3700)
# ---------------------------------------------------------------------------
SEED = 42
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
USE_VIEWS = ["char", "word", "shape", "func"]
VIEWS = {
    "char": dict(transform="raw", analyzer="char", ngram_range=(3, 5), weight=1.0, lowercase=False, max_features=300_000),
    "word": dict(transform="raw", analyzer="word", ngram_range=(1, 3), weight=0.7, lowercase=True, token_pattern=TOKEN_PATTERN),
    "shape": dict(transform="shape", analyzer="char", ngram_range=(2, 6), weight=0.3, lowercase=False),
    "func": dict(transform="func", analyzer="word", ngram_range=(1, 3), weight=0.5, lowercase=True, token_pattern=TOKEN_PATTERN),
}
MIN_DF = 2
SUBLINEAR_TF = True
C = 0.5
CLASS_WEIGHT = "balanced"
MAX_ITER = 5000
TEMPERATURE = 0.1  # softmax(scores / T) pseudo-probabilities; picked on OOF log-loss

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]

# ---------------------------------------------------------------------------
# Text transforms - verbatim copy of src/ntl/svm/views.py
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(TOKEN_PATTERN)
_WORD_RE = re.compile(r"\w+")
_SHAPE_RE = re.compile(r"(\d+)|([^\W\d_]+)")
DISCOURSE_MARKERS = frozenset(
    {
        "however", "moreover", "furthermore", "additionally", "thus",
        "therefore", "indeed", "notably", "essentially", "ultimately",
    }
)
FUNCTION_WORDS = frozenset(ENGLISH_STOP_WORDS) | DISCOURSE_MARKERS


def _shape_token(m: re.Match) -> str:
    if m.group(1) is not None:
        return "d"
    return "X" if m.group(2)[0].isupper() else "x"


def shape_transform(text: str) -> str:
    return _SHAPE_RE.sub(_shape_token, text)


def function_word_transform(text: str) -> str:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in FUNCTION_WORDS or not _WORD_RE.fullmatch(tok):
            out.append(tok)
        else:
            out.append("C")
    return " ".join(out)


TRANSFORMS = {"raw": lambda t: t, "shape": shape_transform, "func": function_word_transform}


def apply_transform(texts, transform):
    fn = TRANSFORMS[transform]
    return [fn(t) for t in texts]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_vectorizer() -> FeatureUnion:
    parts, weights = [], {}
    for name in USE_VIEWS:
        v = VIEWS[name]
        kw = dict(analyzer=v["analyzer"], ngram_range=tuple(v["ngram_range"]), lowercase=v["lowercase"],
                  min_df=MIN_DF, sublinear_tf=SUBLINEAR_TF, max_features=v.get("max_features"), dtype=np.float32)
        if v["analyzer"] == "word" and v.get("token_pattern"):
            kw["token_pattern"] = v["token_pattern"]
        steps = []
        if v["transform"] != "raw":
            steps.append(("prep", FunctionTransformer(apply_transform, kw_args={"transform": v["transform"]})))
        steps.append(("tfidf", TfidfVectorizer(**kw)))
        parts.append((name, Pipeline(steps)))
        weights[name] = float(v["weight"])
    return FeatureUnion(parts, transformer_weights=weights)


def find_data_dir() -> Path:
    for d in (Path("/kaggle/input/competitions/name-that-llm"), Path("/kaggle/input/name-that-llm"),
              Path("./data"), Path(__file__).resolve().parent.parent / "data"):
        if (d / "train.csv").exists() and (d / "test.csv").exists():
            return d
    raise FileNotFoundError("train.csv/test.csv not found in /kaggle/input/... or ./data")


def validate(sub: pd.DataFrame, test_ids: pd.Series) -> None:
    assert list(sub.columns) == ["id", "model"], list(sub.columns)
    assert sub["id"].notna().all() and sub["model"].notna().all(), "null values"
    assert not sub["id"].duplicated().any(), "duplicate ids"
    bad = set(sub["model"]) - set(LABELS)
    assert not bad, f"invalid labels {bad}"
    assert len(sub) == test_ids.nunique() and set(sub["id"]) == set(test_ids), "test ids missing / extra"


def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)
    data_dir = find_data_dir()
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"data: {data_dir}  train={len(train)}  test={len(test)}")

    label_id = {m: i for i, m in enumerate(LABELS)}
    unknown = set(train["model"]) - set(label_id)
    assert not unknown, f"train labels not in LABELS: {unknown}"
    y = train["model"].map(label_id).to_numpy()

    vec = build_vectorizer()
    X = vec.fit_transform(train["text"].fillna("").astype(str).tolist())
    print(f"features: {X.shape[1]:,}  ({time.time() - t0:.0f}s)")
    clf = LinearSVC(C=C, class_weight=CLASS_WEIGHT, max_iter=MAX_ITER, random_state=SEED).fit(X, y)

    raw = clf.decision_function(vec.transform(test["text"].fillna("").astype(str).tolist()))
    scores = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    scores[:, clf.classes_.astype(int)] = raw  # columns in LABELS order

    sub = pd.DataFrame({"id": test["id"].astype(str), "model": [LABELS[i] for i in scores.argmax(1)]})
    validate(sub, test["id"].astype(str))
    sub.to_parquet(out_dir / "submission.parquet", index=False)
    np.save(out_dir / "svm_test_scores.npy", scores)
    print(f"wrote {out_dir / 'submission.parquet'} and svm_test_scores.npy  ({time.time() - t0:.0f}s total)")
    print(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
