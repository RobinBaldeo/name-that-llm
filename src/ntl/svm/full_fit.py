"""Fit the SVM on all of train, predict test, log model + predictions to W&B.

    python -m ntl.svm.full_fit --config configs/svm_best.yaml

Writes artifacts/svm/<run_id>/{model.joblib, test_scores.npy, test_ids.csv, submission.parquet}
and logs W&B artifacts `svm-model-{run_id}` (model) and `svm-test-{run_id}` (predictions).
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from ntl.data import load_test, load_train, resolve
from ntl.labels import LABELS, encode
from ntl.svm.config import apply_overrides, load_config, run_name
from ntl.svm.cv import resolved_config
from ntl.svm.features import build_vectorizer, view_sizes
from ntl.svm.model import balanced_acc, make_svc, scores_in_label_order
from ntl.wandb_utils import init_run


def validate_submission(sub: pd.DataFrame, test_ids) -> None:
    assert list(sub.columns) == ["id", "model"], f"bad columns {list(sub.columns)}"
    assert sub["id"].notna().all() and sub["model"].notna().all(), "nulls in submission"
    assert not sub["id"].duplicated().any(), "duplicate ids in submission"
    bad = set(sub["model"]) - set(LABELS)
    assert not bad, f"invalid labels: {bad}"
    missing = set(test_ids) - set(sub["id"])
    assert not missing and len(sub) == len(set(test_ids)), f"{len(missing)} test ids missing"


def run_full_fit(cfg: dict, wandb_mode: str | None = None) -> dict:
    train = load_train(cfg.get("data_dir", "data"))
    test = load_test(cfg.get("data_dir", "data"))
    y = np.asarray(encode(train["model"]), dtype=int)

    log_cfg = resolved_config(cfg)
    name = run_name(cfg) + "-full"
    run, mode = init_run(log_cfg, job_type="full-fit", name=name, mode=wandb_mode)
    print(f"[full-fit] {name}  (wandb={mode}, run_id={run.id}); train={len(train)} test={len(test)}")

    t0 = time.perf_counter()
    vec = build_vectorizer(cfg)
    X = vec.fit_transform(train["text"].tolist())
    clf = make_svc(cfg).fit(X, y)
    fit_secs = time.perf_counter() - t0
    test_scores = scores_in_label_order(clf, clf.decision_function(vec.transform(test["text"].tolist())))
    train_acc = balanced_acc(y, scores_in_label_order(clf, clf.decision_function(X)).argmax(1))

    sub = pd.DataFrame({"id": test["id"].astype(str), "model": [LABELS[i] for i in test_scores.argmax(1)]})
    validate_submission(sub, test["id"].astype(str))

    out_dir = resolve(cfg.get("output_dir", "artifacts/svm")) / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "svc": clf, "labels": LABELS, "config": log_cfg}, out_dir / "model.joblib", compress=3)
    np.save(out_dir / "test_scores.npy", test_scores)
    test[["id"]].to_csv(out_dir / "test_ids.csv", index=False)
    (out_dir / "labels.json").write_text(json.dumps(LABELS, indent=1))
    sub.to_parquet(out_dir / "submission.parquet", index=False)

    run.summary.update({
        "full/fit_seconds": fit_secs,
        "full/n_features": X.shape[1],
        "full/train_balanced_acc": train_acc,  # in-sample, sanity only
        "full/n_test": len(test),
        "full/temperature": cfg.get("temperature"),
        **{f"full/n_features_{v}": s for v, s in view_sizes(vec).items()},
    })
    if mode != "disabled":
        import wandb

        model_art = wandb.Artifact(f"svm-model-{run.id}", type="model",
                                   description=f"joblib of fitted vectorizers + LinearSVC ({name})",
                                   metadata={"labels": LABELS, "temperature": cfg.get("temperature")})
        model_art.add_file(str(out_dir / "model.joblib"))
        model_art.add_file(str(out_dir / "labels.json"))
        run.log_artifact(model_art)
        pred_art = wandb.Artifact(f"svm-test-{run.id}", type="predictions",
                                  description=f"test decision scores (labels.py order) + submission ({name})",
                                  metadata={"n_test": len(test), "temperature": cfg.get("temperature")})
        for f in ("test_scores.npy", "test_ids.csv", "labels.json", "submission.parquet"):
            pred_art.add_file(str(out_dir / f))
        run.log_artifact(pred_art)
    run.finish()
    print(f"[full-fit] done in {fit_secs:.1f}s; n_features={X.shape[1]:,}; outputs in {out_dir}")
    print(sub["model"].value_counts().to_string())
    return {"run_id": run.id, "out_dir": str(out_dir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/svm_best.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = ap.parse_args(argv)
    run_full_fit(apply_overrides(load_config(args.config), args.override), wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
