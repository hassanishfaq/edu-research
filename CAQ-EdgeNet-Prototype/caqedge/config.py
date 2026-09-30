"""Configuration loading and path resolution (same pattern as the Chapter 3 and 4 prototypes)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


class Config:
    def __init__(self, path: Path | str = CONFIG_PATH):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.raw: dict[str, Any] = yaml.safe_load(f) or {}

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

    @property
    def cic_dir(self) -> Path | None:
        return self.resolve(self.get("data.cic_iot_dir"))

    @property
    def checkpoint(self) -> Path:
        return self.resolve(self.get("model.checkpoint", "models/caq_edgenet.pt"))

    @property
    def policy_path(self) -> Path:
        return self.resolve(self.get("model.policy", "models/policy.pt"))

    @property
    def cache_path(self) -> Path:
        return ROOT / "data" / "cache" / "dataset.npz"

    def summary(self) -> dict[str, Any]:
        cic = self.cic_dir
        return {
            "config_file": str(self.path),
            "cic_iot_dir": str(cic) if cic else None, "cic_iot_dir_exists": bool(cic and cic.exists() and any(cic.glob("*.csv"))),
            "context_csv": self.get("data.context_csv") or None,
            "checkpoint": str(self.checkpoint), "checkpoint_exists": self.checkpoint.exists(),
            "policy": str(self.policy_path), "policy_exists": self.policy_path.exists(),
            "model": self.get("model", {}), "training": self.get("training", {}), "quantization": self.get("quantization", {}),
            "device_profile": self.get("device_profile", {}), "network": self.get("network", {}), "controller": self.get("controller", {}),
            "simulation": {**(self.get("simulation", {}) or {}), "telemetry_file": str(self.resolve(self.get("simulation.telemetry_file", "data/telemetry.jsonl")))},
            "class_counts": self.get("data.class_counts", {}),
        }


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    return Config(path)
