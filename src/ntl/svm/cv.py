"""Prompt-grouped 5-fold CV for the multi-view TF-IDF + LinearSVC model, logged to W&B.

    python -m ntl.svm.cv --config configs/svm.yaml
    python -m ntl.svm.cv --config configs/svm.yaml --override C=0.25 --override views=char,word

Vectorizers and the SVC are fit on the training part of each fold only.
Outputs (also logged as W&B artifact `svm-oof-{run_id}`) go to artifacts/svm/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ntl.data import load_or_create_folds, load_train, resolve
from ntl.labels import LABELS, encode
from ntl.svm.config import active_views, apply_overrides, dump_config, load_config, run_name, views_tag
from ntl.svm.features import build_vectorizer, view_sizes
from ntl.svm.model import (
    balanced_acc,
    make_svc,
    per_class_recall,
    pseudo_proba,
    scores_in_label_order,
    select_temperature,
    worst_confusions,
)
from ntl.wandb_utils import git_commit, init_run


def resolved_config(cfg: dict, folds_source: str | None = None) -> dict:
    """Config dict logged to W&B: only active views, plus provenance."""
    out = {k: v for k, v in cfg.items() if k != "views"}
    out["views"] = active_views(cfg)
    out["views_tag"] = views_tag(cfg)
    out["view_weights"] = {n: v.get("weight", 1.0) for n, v in out["views"].items()}
    out["git_commit"] = git_commit()
    if folds_source:
        out["folds_source"] = folds_source
    return out


def fit_predict_fold(cfg, texts_tr, y_tr, texts_va):
    vec = build_vectorizer(cfg)
    X_tr = vec.fit_transform(texts_tr)
    X_va = vec.transform(texts_va)
    clf = make_svc(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_tr, y_tr)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    scores = scores_in_label_order(clf, clf.decision_function(X_va))
    return scores, X_tr.shape[1], view_sizes(vec), n_conv


def run_cv(cfg: dict, wandb_mode: str | None = None, verbose: bool = True) -> dict:
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    train = load_train(cfg.get("data_dir", "data"))
    fold, folds_source = load_or_create_folds(train, cfg["folds_path"], int(cfg.get("n_folds", 5)), seed)
    texts = train["text"].to_numpy(dtype=object)
    y = np.asarray(encode(train["model"]), dtype=int)
    n_folds = int(fold.max()) + 1

    log_cfg = resolved_config(cfg, folds_source)
    name = run_name(cfg)
    run, mode = init_run(log_cfg, job_type="cv", name=name, mode=wandb_mode)
    if verbose:
        print(f"[cv] {name}  (wandb={mode}, run_id={run.id})")

    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    fold_accs, fold_rows = [], []
    for k in range(n_folds):
        tr = np.flatnonzero(fold.to_numpy() != k)
        va = np.flatnonzero(fold.to_numpy() == k)
        t0 = time.perf_counter()
        scores, n_feat, sizes, n_conv = fit_predict_fold(cfg, texts[tr], y[tr], texts[va])
        secs = time.perf_counter() - t0
        oof[va] = scores
        acc = balanced_acc(y[va], scores.argmax(1))
        fold_accs.append(acc)
        row = {
            f"fold/{k}/balanced_acc": acc,
            f"fold/{k}/fit_seconds": secs,
            f"fold/{k}/n_features": n_feat,
            f"fold/{k}/convergence_warnings": n_conv,
        }
        row.update({f"fold/{k}/n_features_{v}": s for v, s in sizes.items()})
        run.log(row)
        fold_rows.append({"fold": k, "balanced_acc": acc, "fit_seconds": secs, "n_features": n_feat, **sizes})
        if verbose:
            print(f"  fold {k}: bal_acc={acc:.4f}  n_feat={n_feat:,}  {secs:.1f}s" + (f"  ({n_conv} conv. warnings)" if n_conv else ""))

    pred = oof.argmax(1)
    cv_acc = balanced_acc(y, pred)
    cv_std = float(np.std(fold_accs))
    best_T, losses = select_temperature(oof, y, cfg["temperature_grid"])
    recalls = per_class_recall(y, pred)
    confusions = worst_confusions(y, pred, top=20)
    counts = train["model"].value_counts().to_dict()

    summary = {
        "cv/balanced_acc": cv_acc,
        "cv/balanced_acc_std": cv_std,
        "cv/logloss": losses[best_T],
        "cv/best_T": best_T,
        "cv/accuracy": float((pred == y).mean()),
    }
    summary.update({f"cv/logloss_T{T}": l for T, l in losses.items()})
    summary.update({f"recall/{m}": r for m, r in recalls.items()})
    run.summary.update(summary)

    if mode != "disabled":
        import wandb

        recall_rows = sorted(([m, r, int(counts.get(m, 0))] for m, r in recalls.items()), key=lambda r: r[1])
        run.log({
            "cv/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y.tolist(), preds=pred.tolist(), class_names=LABELS, title="OOF confusion (svm)"
            ),
            "cv/recall_table": wandb.Table(columns=["model", "recall", "n"], data=recall_rows),
            "cv/worst_confusions": wandb.Table(
                columns=["true", "pred", "count", "same_family"],
                data=[[c["true"], c["pred"], c["count"], c["same_family"]] for c in confusions],
            ),
            "cv/fold_table": wandb.Table(dataframe=pd.DataFrame(fold_rows)),
        })

    # ---- local outputs + OOF artifact ----
    out_dir = resolve(cfg.get("output_dir", "artifacts/svm")) / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_scores.npy", oof)
    pd.DataFrame({"id": train["id"], "prompt_id": train["prompt_id"], "fold": fold, "model": train["model"]}).to_csv(
        out_dir / "oof_ids.csv", index=False
    )
    (out_dir / "labels.json").write_text(json.dumps(LABELS, indent=1))
    metrics = {**summary, "fold_balanced_acc": fold_accs, "worst_confusions": confusions, "temperature_losses": losses}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    dump_config({**log_cfg, "temperature": best_T}, out_dir / "config.yaml")

    if mode != "disabled":
        import wandb

        art = wandb.Artifact(
            f"svm-oof-{run.id}",
            type="oof",
            description=f"OOF decision scores ({len(train)}x{len(LABELS)}, labels.py order) for {name}",
            metadata={"cv/balanced_acc": cv_acc, "best_T": best_T, "run_name": name},
        )
        for f in ("oof_scores.npy", "oof_ids.csv", "labels.json"):
            art.add_file(str(out_dir / f))
        run.log_artifact(art)

    record = {
        "run_id": run.id,
        "name": name,
        "job_type": "cv",
        "wandb_mode": mode,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "C": cfg["model"]["C"],
        "views": list(cfg["use_views"]),
        "weights": log_cfg["view_weights"],
        "cv/balanced_acc": cv_acc,
        "cv/balanced_acc_std": cv_std,
        "cv/logloss": losses[best_T],
        "cv/best_T": best_T,
        "lowest_recalls": sorted(recalls.items(), key=lambda kv: kv[1])[:5],
        "out_dir": str(out_dir),
        "config": {**log_cfg, "temperature": best_T},
    }
    if mode != "disabled":  # local run log = offline-friendly mirror of the W&B group
        with open(resolve(cfg.get("output_dir", "artifacts/svm")) / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    run.finish()
    if verbose:
        print(f"[cv] {name}: balanced_acc={cv_acc:.4f} +/- {cv_std:.4f}  logloss={losses[best_T]:.4f} @ T={best_T}")
        print("     hardest: " + ", ".join(f"{m}={r:.2f}" for m, r in record["lowest_recalls"]))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/svm.yaml")
    ap.add_argument("--override", action="append", default=[], help="key=value (repeatable), e.g. C=0.25, views=char,word")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    run_cv(cfg, wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
