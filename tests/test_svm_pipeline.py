import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ntl.labels import LABELS
from ntl.svm.config import apply_overrides, load_config, run_name
from ntl.svm.features import build_vectorizer

REPO = Path(__file__).resolve().parents[1]


def test_overrides_and_shorthands():
    cfg = load_config("configs/svm.yaml")
    out = apply_overrides(cfg, ["C=0.25", "views=char,word", "views.shape.weight=0.7"])
    assert out["model"]["C"] == 0.25
    assert out["use_views"] == ["char", "word"]
    assert out["views"]["shape"]["weight"] == 0.7
    assert cfg["model"]["C"] == 0.5  # original untouched
    assert run_name(out) == "svm-char+word-C0.25"


def test_view_weights_scale_blocks():
    texts = ["The cat sat on the mat.", "A dog ran in the park!", "The cat ran.", "A dog sat."] * 3
    cfg = load_config("configs/svm.yaml")
    w1 = apply_overrides(cfg, ["views=word,func"])
    w2 = apply_overrides(w1, ["views.word.weight=1.4"])  # word: 0.7 -> 1.4 (x2)
    X1, X2 = build_vectorizer(w1).fit_transform(texts), build_vectorizer(w2).fit_transform(texts)
    n_word = build_vectorizer(w1).transformer_list[0][1].fit_transform(texts).shape[1]
    np.testing.assert_allclose(X2[:, :n_word].toarray(), 2 * X1[:, :n_word].toarray(), rtol=1e-6)
    np.testing.assert_allclose(X2[:, n_word:].toarray(), X1[:, n_word:].toarray(), rtol=1e-6)


@pytest.fixture
def tiny_data(tmp_path):
    """18 classes x 10 prompts, each class with a distinctive verbal tic."""
    rng = random.Random(0)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    rows = []
    for p in range(10):
        for c, m in enumerate(LABELS):
            body = " ".join(rng.choice(words) for _ in range(30))
            rows.append({"id": f"{p:02d}{c:02d}", "prompt_id": f"p{p}", "text": f"tic{c} {body} tic{c}.", "model": m})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": ["t1", "t2"], "text": ["tic3 alpha beta.", "tic7 gamma."]}).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def test_cv_end_to_end_disabled_wandb(tiny_data):
    from ntl.svm.cv import run_cv

    cfg = apply_overrides(load_config("configs/svm.yaml"), [
        f"data_dir={tiny_data.as_posix()}", f"folds_path={(tiny_data / 'folds.csv').as_posix()}",
        f"output_dir={(tiny_data / 'out').as_posix()}", "views.char.max_features=5000",
    ])
    rec = run_cv(cfg, wandb_mode="disabled", verbose=False)
    out = Path(rec["out_dir"])
    oof = np.load(out / "oof_scores.npy")
    assert oof.shape == (180, 18)
    assert rec["cv/balanced_acc"] > 0.9  # the tic makes it trivially separable
    assert json.loads((out / "labels.json").read_text()) == LABELS
    ids = pd.read_csv(out / "oof_ids.csv")
    assert (ids.groupby("prompt_id")["fold"].nunique() == 1).all()  # prompt-grouped
    assert ids["fold"].nunique() == 5


def test_full_fit_disabled_wandb(tiny_data):
    from ntl.svm.full_fit import run_full_fit

    cfg = apply_overrides(load_config("configs/svm.yaml"), [
        f"data_dir={tiny_data.as_posix()}", f"output_dir={(tiny_data / 'out').as_posix()}",
    ])
    rec = run_full_fit(cfg, wandb_mode="disabled")
    sub = pd.read_parquet(Path(rec["out_dir"]) / "submission.parquet")
    assert sub["model"].tolist() == ["claude-opus-5", "gemini-3.7-flash"]  # tic3, tic7


def test_kaggle_constants_match_best_config():
    best_path = REPO / "configs" / "svm_best.yaml"
    if not best_path.exists():
        pytest.skip("configs/svm_best.yaml not written yet")
    best = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("svm_submit", REPO / "kaggle" / "svm_submit.py")
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)

    assert k.LABELS == LABELS
    assert k.USE_VIEWS == best["use_views"]
    assert k.C == pytest.approx(best["model"]["C"])
    assert k.MAX_ITER == best["model"]["max_iter"]
    assert k.CLASS_WEIGHT == best["model"]["class_weight"]
    assert k.SEED == best["seed"]
    assert k.TEMPERATURE == pytest.approx(best["temperature"])
    for name in best["use_views"]:
        b, kv = best["views"][name], k.VIEWS[name]
        assert kv["transform"] == b["transform"]
        assert kv["analyzer"] == b["analyzer"]
        assert tuple(kv["ngram_range"]) == tuple(b["ngram_range"])
        assert kv["weight"] == pytest.approx(b["weight"])
        assert kv["lowercase"] == b["lowercase"]
        assert kv.get("max_features") == b.get("max_features")
        assert kv.get("token_pattern") == b.get("token_pattern")
        assert b.get("min_df", 2) == k.MIN_DF and b.get("sublinear_tf", True) == k.SUBLINEAR_TF
