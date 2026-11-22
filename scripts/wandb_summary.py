"""Print all W&B runs in group `svm` as a comparison table (no browser needed).

    python scripts/wandb_summary.py               # CV runs, best first
    python scripts/wandb_summary.py --all         # include full-fit runs and runs tagged `superseded`
    python scripts/wandb_summary.py --local       # read artifacts/svm/runs.jsonl instead of the W&B API
"""

from __future__ import annotations

import argparse
import json

from ntl.data import resolve


def _lowest(recalls: dict, k: int = 5) -> str:
    items = sorted(recalls.items(), key=lambda kv: kv[1])[:k]
    return ", ".join(f"{m}={r:.2f}" for m, r in items)


def rows_from_wandb(include_all: bool) -> list[dict]:
    from ntl.wandb_utils import api, project_path

    a = api()
    filters = {"group": "svm"} if include_all else {"group": "svm", "jobType": "cv"}
    rows = []
    for r in a.runs(project_path(a), filters=filters, order="-created_at"):
        if "superseded" in r.tags and not include_all:
            continue
        s = r.summary._json_dict
        cfg = r.config
        rows.append({
            "name": r.name,
            "id": r.id,
            "job": r.job_type,
            "state": r.state,
            "acc": s.get("cv/balanced_acc"),
            "std": s.get("cv/balanced_acc_std"),
            "T": s.get("cv/best_T"),
            "C": (cfg.get("model") or {}).get("C"),
            "views": ",".join(cfg.get("use_views") or []),
            "lowest": _lowest({k.split("/", 1)[1]: v for k, v in s.items() if k.startswith("recall/")}),
        })
    return rows


def rows_from_local() -> list[dict]:
    path = resolve("artifacts/svm/runs.jsonl")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({
            "name": r["name"], "id": r["run_id"], "job": r["job_type"], "state": r["wandb_mode"],
            "acc": r["cv/balanced_acc"], "std": r["cv/balanced_acc_std"], "T": r["cv/best_T"],
            "C": r["C"], "views": ",".join(r["views"]), "lowest": _lowest(dict(r["lowest_recalls"])),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include full-fit runs")
    ap.add_argument("--local", action="store_true", help="read the local runs.jsonl mirror")
    args = ap.parse_args()
    rows = rows_from_local() if args.local else rows_from_wandb(args.all)
    if not rows:
        print("no runs found in group 'svm'")
        return
    rows.sort(key=lambda r: (r["acc"] is None, -(r["acc"] or 0)))

    def fmt(v, p=4):
        if v is None:
            return "-"
        return f"{v:.{p}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    w = max(len(r["name"]) for r in rows)
    print(f"{'name':<{w}}  {'run':<8}  {'job':<8}  {'bal_acc':>7}  {'std':>6}  {'C':>5}  {'T':>5}  {'views':<22}  5 lowest recalls")
    print("-" * (w + 140))
    for r in rows:
        print(f"{r['name']:<{w}}  {r['id']:<8}  {(r['job'] or '-'):<8}  {fmt(r['acc']):>7}  {fmt(r['std']):>6}  "
              f"{fmt(r['C'], 2):>5}  {fmt(r['T'], 2):>5}  {r['views']:<22}  {r['lowest']}")


if __name__ == "__main__":
    main()
