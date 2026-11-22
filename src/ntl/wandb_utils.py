"""W&B setup shared by the CV / full-fit CLIs and the summary scripts.

Credentials come from the environment or the repo's `.env` file, never from code.
`.env` may name the key `WANDB_API_KEY` or `WB_API_KEY`; either works.
"""

from __future__ import annotations

import os
import subprocess

from ntl.data import REPO_ROOT

PROJECT = "name-that-llm"
_KEY_ALIASES = ("WB_API_KEY", "WANDB_KEY", "WANDB_TOKEN")
_ENTITY_ALIASES = ("WB_ENTITY",)


def load_env() -> None:
    """Load `.env` (without overriding real env vars) and normalise W&B variable names."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is optional; plain env vars still work
        pass
    if not os.environ.get("WANDB_API_KEY"):
        for alias in _KEY_ALIASES:
            if os.environ.get(alias):
                os.environ["WANDB_API_KEY"] = os.environ[alias]
                break
    if not os.environ.get("WANDB_ENTITY"):
        for alias in _ENTITY_ALIASES:
            if os.environ.get(alias):
                os.environ["WANDB_ENTITY"] = os.environ[alias]
                break


def resolve_mode(requested: str | None = None) -> str:
    """online / offline / disabled. Falls back to offline when no API key is available."""
    load_env()
    mode = requested or os.environ.get("WANDB_MODE") or "online"
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WARNING: WANDB_API_KEY not found in env/.env -> running offline (sync later with `wandb sync wandb/offline-run-*`)")
        mode = "offline"
    os.environ["WANDB_MODE"] = mode
    return mode


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def init_run(cfg: dict, job_type: str, name: str, mode: str | None = None):
    """Start a W&B run in group `svm`. Returns (run, mode)."""
    mode = resolve_mode(mode)
    import wandb

    wb = cfg.get("wandb", {})
    run = wandb.init(
        project=wb.get("project", PROJECT),
        entity=os.environ.get("WANDB_ENTITY") or None,
        group=wb.get("group", "svm"),
        job_type=job_type,
        name=name,
        tags=list(wb.get("tags", ["svm", "tfidf", "cpu"])),
        config=cfg,
        dir=str(REPO_ROOT),
        mode=mode,
        reinit="finish_previous",
    )
    return run, mode


def api():
    """A wandb.Api() with credentials loaded from env/.env."""
    load_env()
    import wandb

    return wandb.Api()


def project_path(api_obj) -> str:
    entity = os.environ.get("WANDB_ENTITY") or api_obj.default_entity
    return f"{entity}/{PROJECT}"
