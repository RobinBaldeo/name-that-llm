"""Run the SPEC_svm §5 experiment plan; each experiment is one W&B run (group `svm`).

    python scripts/run_svm_experiments.py --stage baseline     # 1. repro 2-view baseline
    python scripts/run_svm_experiments.py --stage default      # 2. 4-view default (C=0.5)
    python scripts/run_svm_experiments.py --stage C            # 3. C sweep {0.1, 0.25, 1.0} (+ 0.5 from stage 2)
    python scripts/run_svm_experiments.py --stage ablation     # 4. each view alone + all-minus-one (8 runs)
    python scripts/run_svm_experiments.py --stage weights --view shape --delta 0.2   # 5. optional
    python scripts/run_svm_experiments.py --stage all          # 1-4 in order

Ablations / weight tweaks use the best C found so far among full 4-view runs
(read from artifacts/svm/runs.jsonl) unless --C is given.
"""

from __future__ import annotations

import argparse
import json

from ntl.data import resolve
from ntl.svm.config import apply_overrides, load_config
from ntl.svm.cv import run_cv

ALL_VIEWS = ["char", "word", "shape", "func"]
C_GRID = [0.1, 0.25, 0.5, 1.0]


def _runs() -> list[dict]:
    path = resolve("artifacts/svm/runs.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_full_view_C(default: float = 0.5) -> float:
    full = [r for r in _runs() if r.get("job_type") == "cv" and sorted(r["views"]) == sorted(ALL_VIEWS)
            and r["config"].get("tag") is None and not r["config"].get("name_suffix")]
    if not full:
        return default
    return float(max(full, key=lambda r: r["cv/balanced_acc"])["C"])


def run(base: dict, overrides: list[str], mode: str | None) -> dict:
    return run_cv(apply_overrides(base, overrides), wandb_mode=mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["baseline", "default", "C", "ablation", "weights", "all"])
    ap.add_argument("--config", default="configs/svm.yaml")
    ap.add_argument("--C", type=float, default=None, help="C for ablation/weights (default: best 4-view C so far)")
    ap.add_argument("--view", help="view to re-weight (weights stage)")
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    stages = ["baseline", "default", "C", "ablation"] if args.stage == "all" else [args.stage]
    results = []

    for stage in stages:
        if stage == "baseline":
            results.append(run_cv(load_config("configs/svm_baseline.yaml"), wandb_mode=args.wandb_mode))
        elif stage == "default":
            results.append(run(base, [], args.wandb_mode))
        elif stage == "C":
            done = {r["name"] for r in _runs()}
            for C in C_GRID:
                if f"svm-char+word+shape+func-C{C}" in done:
                    print(f"[C sweep] C={C} already logged (default run); skipping")
                    continue
                results.append(run(base, [f"C={C}"], args.wandb_mode))
        elif stage in ("ablation", "weights"):
            C = args.C if args.C is not None else best_full_view_C(float(base["model"]["C"]))
            print(f"[{stage}] using C={C}")
            if stage == "ablation":
                for v in ALL_VIEWS:  # each view alone
                    results.append(run(base, [f"C={C}", f"views={v}"], args.wandb_mode))
                for v in ALL_VIEWS:  # all minus one
                    rest = ",".join(x for x in ALL_VIEWS if x != v)
                    results.append(run(base, [f"C={C}", f"views={rest}", f"name_suffix=minus-{v}"], args.wandb_mode))
            else:
                if not args.view:
                    ap.error("--stage weights needs --view")
                w0 = float(base["views"][args.view]["weight"])
                for w in (round(w0 - args.delta, 3), round(w0 + args.delta, 3)):
                    results.append(run(base, [f"C={C}", f"views.{args.view}.weight={w}", f"name_suffix={args.view}w{w}"], args.wandb_mode))

    print("\n=== results ===")
    for r in sorted(results, key=lambda r: -r["cv/balanced_acc"]):
        print(f"{r['cv/balanced_acc']:.4f} +/- {r['cv/balanced_acc_std']:.4f}  {r['name']}  (run {r['run_id']})")


if __name__ == "__main__":
    main()
