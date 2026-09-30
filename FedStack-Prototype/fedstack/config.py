"""Configuration loading and path resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def _norm_class(name: str) -> str:
    """Normalise a class-folder name (the dataset has one folder with a stray space)."""
    return "".join(name.split())


class Config:
    def __init__(self, path: Path | str = CONFIG_PATH):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.raw: dict[str, Any] = yaml.safe_load(f) or {}

    # ---- helpers -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        q = Path(os.path.expandvars(os.path.expanduser(str(p))))
        return q if q.is_absolute() else (ROOT / q)

    # ---- typed accessors ----------------------------------------------
    @property
    def mode(self) -> str:
        return str(self.get("mode", "auto")).lower()

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return list(self.get("model.nodes", []))

    def node_checkpoint(self, node: dict[str, Any]) -> Path | None:
        return self.resolve(node.get("checkpoint"))

    @property
    def stacker_path(self) -> Path | None:
        return self.resolve(self.get("model.stacker_file"))

    @property
    def partition(self) -> dict[str, list[str]]:
        part = self.get("federation.partition", {}) or {}
        return {k: [_norm_class(c) for c in v] for k, v in part.items()}

    @property
    def training_dir(self) -> Path | None:
        return self.resolve(self.get("data.training_images_dir"))

    @property
    def zero_day_dir(self) -> Path | None:
        return self.resolve(self.get("data.zero_day_dir"))

    @property
    def cache_path(self) -> Path:
        return ROOT / "data" / "cache" / f"thumbs{int(self.get('data.thumbnail_size', 64))}.npz"

    @property
    def samples_dir(self) -> Path:
        return ROOT / "data" / "samples"

    def summary(self) -> dict[str, Any]:
        """Sanitised view for the UI."""
        nodes = []
        for n in self.nodes:
            ck = self.node_checkpoint(n)
            nodes.append({
                "name": n.get("name"), "label": n.get("label", ""),
                "checkpoint": str(ck) if ck else None,
                "checkpoint_exists": bool(ck and ck.exists()),
                "families": self.partition.get(n.get("name"), []),
            })
        st = self.stacker_path
        return {
            "config_file": str(self.path),
            "mode_requested": self.mode,
            "backbone": self.get("model.backbone"),
            "image_size": self.get("model.image_size"),
            "accuracy_weighting": bool(self.get("model.accuracy_weighting", True)),
            "nodes": nodes,
            "stacker": {"kind": self.get("model.stacker", "rf"), "path": str(st) if st else None, "exists": bool(st and st.exists())},
            "training_images_dir": str(self.training_dir) if self.training_dir else None,
            "training_images_dir_exists": bool(self.training_dir and self.training_dir.exists()),
            "zero_day_dir": str(self.zero_day_dir) if self.zero_day_dir else None,
            "zero_day_dir_exists": bool(self.zero_day_dir and self.zero_day_dir.exists()),
            "federation": {"rounds": self.get("federation.rounds", 20), "local_epochs": self.get("federation.local_epochs", 5)},
            "open_set": self.get("open_set", {}),
            "watch": {**(self.get("watch", {}) or {}), "dir": str(self.resolve(self.get("watch.dir", "watch_inbox")))},
        }


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    return Config(path)
