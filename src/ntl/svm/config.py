"""Load YAML configs and apply `--override key=value` edits.

Override keys are dotted paths into the config (`views.shape.weight=0.7`), with two
shorthands from the spec:
    C=0.25                -> model.C = 0.25
    views=char,word       -> use_views = [char, word]
Values are parsed as YAML (`0.25` -> float, `[1,3]` -> list, `true` -> bool).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from ntl.data import resolve

SHORTHANDS = {"C": "model.C", "views": "use_views", "T": "temperature"}


def load_config(path: str | Path) -> dict:
    with open(resolve(path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["config_path"] = str(path)
    return cfg


def _parse_value(key: str, raw: str):
    if key == "use_views" and not raw.strip().startswith("["):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = SHORTHANDS.get(key.strip(), key.strip())
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(key, raw)
    return cfg


def views_tag(cfg: dict) -> str:
    return cfg.get("tag") or "+".join(cfg["use_views"])


def run_name(cfg: dict) -> str:
    """svm-{views_tag}-C{C}[-{suffix}]"""
    name = f"svm-{views_tag(cfg)}-C{cfg['model']['C']}"
    if cfg.get("name_suffix"):
        name += f"-{cfg['name_suffix']}"
    return name


def active_views(cfg: dict) -> dict:
    """Only the views actually used (keeps the logged config readable)."""
    return {n: cfg["views"][n] for n in cfg["use_views"]}


def dump_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
