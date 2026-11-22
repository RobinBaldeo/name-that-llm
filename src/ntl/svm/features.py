"""Build the weighted multi-view TF-IDF matrix.

Each view = (text transform) -> TfidfVectorizer. Views are combined with a
FeatureUnion whose `transformer_weights` multiply each view's matrix by its weight
before the sparse hstack. The returned object is a plain sklearn estimator: fit it on
the training part of a fold only, then `.transform` the validation / test texts.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from ntl.svm.views import TRANSFORMS, apply_transform

VIEW_DEFAULTS = {
    "transform": "raw",
    "analyzer": "char",
    "ngram_range": (1, 1),
    "weight": 1.0,
    "lowercase": True,
    "min_df": 2,
    "sublinear_tf": True,
    "max_features": None,
    "token_pattern": None,
}


def view_spec(spec: dict) -> dict:
    s = {**VIEW_DEFAULTS, **spec}
    s["ngram_range"] = tuple(s["ngram_range"])
    if s["transform"] not in TRANSFORMS:
        raise ValueError(f"unknown transform {s['transform']!r}; choose from {list(TRANSFORMS)}")
    return s


def build_view(spec: dict) -> Pipeline:
    s = view_spec(spec)
    tfidf_kw = dict(
        analyzer=s["analyzer"],
        ngram_range=s["ngram_range"],
        lowercase=s["lowercase"],
        min_df=s["min_df"],
        sublinear_tf=s["sublinear_tf"],
        max_features=s["max_features"],
        dtype=np.float32,
    )
    if s["analyzer"] == "word" and s["token_pattern"]:
        tfidf_kw["token_pattern"] = s["token_pattern"]
    steps = []
    if s["transform"] != "raw":
        steps.append(("prep", FunctionTransformer(apply_transform, kw_args={"transform": s["transform"]})))
    steps.append(("tfidf", TfidfVectorizer(**tfidf_kw)))
    return Pipeline(steps)


def build_vectorizer(cfg: dict) -> FeatureUnion:
    """FeatureUnion over cfg['use_views'] (in that order), weighted by each view's weight."""
    names = list(cfg["use_views"])
    if not names:
        raise ValueError("use_views is empty")
    views = cfg["views"]
    missing = [n for n in names if n not in views]
    if missing:
        raise ValueError(f"use_views references undefined views: {missing}")
    return FeatureUnion(
        [(n, build_view(views[n])) for n in names],
        transformer_weights={n: float(view_spec(views[n])["weight"]) for n in names},
        n_jobs=cfg.get("n_jobs"),
    )


def view_sizes(vec: FeatureUnion) -> dict[str, int]:
    """Vocabulary size per fitted view."""
    return {name: len(pipe.named_steps["tfidf"].vocabulary_) for name, pipe in vec.transformer_list}
