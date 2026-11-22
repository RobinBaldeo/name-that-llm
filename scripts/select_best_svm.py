"""Pick the best SVM CV run, write configs/svm_best.yaml, alias its OOF artifact `best`.

    python scripts/select_best_svm.py                 # best by cv/balanced_acc (from W&B)
    python scripts/select_best_svm.py --run-id abc123 # force a specific run
    python scripts/select_best_svm.py --local         # select from artifacts/svm/runs.jsonl, no W&B aliasing

The `best` alias is moved: removed from every other svm-oof-* artifact first.
After a full-fit run, pass --full-fit-run-id to also alias svm-model-* / svm-test-* as `best`.
"""

from __future__ import annotations

import argparse
import json

import yaml

from ntl.data import resolve

KEEP_KEYS = ["seed", "data_dir", "folds_path", "n_folds", "output_dir", "n_jobs", "tag", "name_suffix",
             "use_views", "views", "model", "temperature_grid", "wandb"]


def best_config(raw_cfg: dict, temperature: float, source: dict) -> dict:
    cfg = {k: raw_cfg[k] for k in KEEP_KEYS if k in raw_cfg and raw_cfg[k] is not None}
    cfg["temperature"] = float(temperature)
    cfg["selected_from"] = source
    return cfg


def move_alias(a, project: str, art_type: str, prefix: str, run_id: str, alias: str = "best") -> None:
    target = f"{prefix}-{run_id}"
    for coll in a.artifact_collections(project, type_name=art_type):
        if not coll.name.startswith(prefix + "-"):
            continue
        for art in coll.artifacts():
            if alias in art.aliases and coll.name != target:
                art.aliases.remove(alias)
                art.save()
                print(f"  removed '{alias}' from {coll.name}:{art.version}")
    art = a.artifact(f"{project}/{target}:latest")
    if alias not in art.aliases:
        art.aliases.append(alias)
        art.save()
    print(f"  aliased {target}:{art.version} as '{alias}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--full-fit-run-id", help="also alias svm-model-/svm-test-<id> as best")
    ap.add_argument("--out", default="configs/svm_best.yaml")
    args = ap.parse_args()

    if args.local:
        runs = [json.loads(l) for l in resolve("artifacts/svm/runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        runs = [r for r in runs if r["job_type"] == "cv" and (not args.run_id or r["run_id"] == args.run_id)]
        best = max(runs, key=lambda r: r["cv/balanced_acc"])
        raw_cfg, run_id, name, acc, T = best["config"], best["run_id"], best["name"], best["cv/balanced_acc"], best["cv/best_T"]
        a = project = None
    else:
        from ntl.wandb_utils import api, project_path

        a = api()
        project = project_path(a)
        if args.run_id:
            run = a.run(f"{project}/{args.run_id}")
        else:
            runs = [r for r in a.runs(project, filters={"group": "svm", "jobType": "cv", "state": "finished"})
                    if r.summary.get("cv/balanced_acc") is not None and "superseded" not in r.tags]
            run = max(runs, key=lambda r: r.summary["cv/balanced_acc"])
        raw_cfg, run_id, name = dict(run.config), run.id, run.name
        acc, T = run.summary["cv/balanced_acc"], run.summary["cv/best_T"]

    cfg = best_config(raw_cfg, T, {"run_id": run_id, "run_name": name, "cv/balanced_acc": round(float(acc), 5)})
    out = resolve(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Best SVM config, selected from W&B run {run_id} ({name}), CV balanced acc {acc:.4f}.\n")
        fh.write("# Written by scripts/select_best_svm.py - kaggle/svm_submit.py inlines these values.\n")
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"best CV run: {name} ({run_id})  balanced_acc={acc:.4f}  T={T}  -> {args.out}")

    if a is not None:
        move_alias(a, project, "oof", "svm-oof", run_id)
        if args.full_fit_run_id:
            move_alias(a, project, "model", "svm-model", args.full_fit_run_id)
            move_alias(a, project, "predictions", "svm-test", args.full_fit_run_id)


if __name__ == "__main__":
    main()
