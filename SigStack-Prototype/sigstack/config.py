"""Configuration loading and path resolution (same pattern as the Chapter 3 prototype)."""
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
        return str(self.get("encoder.mode", "auto")).lower()

    @property
    def engines(self) -> list[dict[str, str]]:
        out = []
        for e in self.get("data.engines", []) or []:
            out.append({"folder": str(e.get("folder")), "display": str(e.get("display", e.get("folder")))})
        return out

    @property
    def engine_folders(self) -> list[str]:
        return [e["folder"] for e in self.engines]

    def display_name(self, folder: str) -> str:
        for e in self.engines:
            if e["folder"] == folder:
                return e["display"]
        return folder

    @property
    def archive(self) -> Path | None:
        return self.resolve(self.get("data.archive"))

    @property
    def reports_dir(self) -> Path | None:
        return self.resolve(self.get("data.reports_dir"))

    @property
    def cache_path(self) -> Path:
        return ROOT / "data" / "cache" / "reports.pkl"

    @property
    def state_path(self) -> Path:
        return ROOT / "data" / "cache" / "state_surrogate.pkl"

    @property
    def samples_dir(self) -> Path:
        return ROOT / "data" / "samples"

    @property
    def category_map_path(self) -> Path:
        return self.resolve(self.get("labels.category_map", "category_map.yaml"))

    def roberta_dir(self, slot: str) -> Path | None:
        """slot = 'category' or an engine folder name."""
        if slot == "category":
            return self.resolve(self.get("encoder.roberta.category_model_dir"))
        pat = str(self.get("encoder.roberta.engine_model_dir_pattern", "models/roberta_{engine}"))
        return self.resolve(pat.format(engine=slot))

    def summary(self) -> dict[str, Any]:
        slots = []
        for slot in ["category"] + self.engine_folders:
            d = self.roberta_dir(slot)
            slots.append({"slot": slot, "display": "direct category model" if slot == "category" else f"{self.display_name(slot)} family model",
                          "dir": str(d) if d else None, "exists": bool(d and (d / "config.json").exists())})
        return {
            "config_file": str(self.path), "mode_requested": self.mode,
            "roberta": {"base_model": self.get("encoder.roberta.base_model"), "max_length": self.get("encoder.roberta.max_length"),
                        "int8_dynamic": bool(self.get("encoder.roberta.int8_dynamic", False))},
            "model_slots": slots,
            "archive": str(self.archive) if self.archive else None, "archive_exists": bool(self.archive and self.archive.exists()),
            "reports_dir": str(self.reports_dir) if self.reports_dir else None,
            "engines": self.engines,
            "normalizer": self.get("normalizer", {}), "labels": {k: v for k, v in (self.get("labels", {}) or {}).items() if k != "report_groups"},
            "train": self.get("train", {}), "open_set": self.get("open_set", {}),
            "watch": {**(self.get("watch", {}) or {}), "dir": str(self.resolve(self.get("watch.dir", "watch_inbox")))},
            "acquisition": {"virustotal_configured": bool(self.get("acquisition.virustotal_api_key")), "cache_dir": str(self.resolve(self.get("acquisition.cache_dir", "data/cache/reports")))},
        }


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    return Config(path)
