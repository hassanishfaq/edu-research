"""Label service -- Section 4.1 (Eq. 4.4) and the versioned category map of Table 4.3 / Section 4.4.4.

    Score(f) = av_b + av_k + av_s          Eq. (4.4): number of engines that flag the file
    map(engine, raw string) -> category    first matching regex rule of category_map.yaml
    consensus(f)                           majority of the mapped categories; ties broken by engine priority
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

UNMAPPED = "Unmapped"


class CategoryMap:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.version: str = str(raw.get("version", "v?"))
        self.categories: list[str] = list(raw.get("categories", []))
        self.rules: dict[str, list[tuple[re.Pattern, str, str]]] = {}
        for engine, rules in (raw.get("rules") or {}).items():
            compiled = []
            for r in rules or []:
                compiled.append((re.compile(r["match"], re.IGNORECASE), str(r["category"]), str(r["match"])))
            self.rules[engine] = compiled
        self._memo: dict[tuple[str, str], tuple[str, str | None]] = {}

    def map(self, engine: str, raw: str) -> tuple[str, str | None]:
        """-> (category, rule that matched or None)."""
        key = (engine, raw)
        if key in self._memo:
            return self._memo[key]
        out: tuple[str, str | None] = (UNMAPPED, None)
        for pat, cat, src in self.rules.get(engine, []):
            if pat.search(raw):
                out = (cat, src)
                break
        self._memo[key] = out
        return out

    def rules_table(self) -> list[dict[str, str]]:
        return [{"engine": e, "match": src, "category": cat} for e, rs in self.rules.items() for _, cat, src in rs]


def consensus(mapped: dict[str, str], priority: list[str]) -> tuple[str, bool]:
    """Majority category over the engines that flagged the file (Unmapped ignored).  -> (category, disputed)."""
    cats = [c for c in mapped.values() if c and c != UNMAPPED]
    if not cats:
        return UNMAPPED, False
    cnt = Counter(cats)
    best = cnt.most_common()
    top = [c for c, n in best if n == best[0][1]]
    disputed = len(cnt) > 1
    if len(top) == 1:
        return top[0], disputed
    for eng in priority:                      # tie: engine priority
        c = mapped.get(eng)
        if c in top:
            return c, True
    return sorted(top)[0], True


def score(mapped_or_raw: dict[str, Any]) -> int:
    """Eq. (4.4): aggregated detection score = number of engines with a verdict."""
    return sum(1 for v in mapped_or_raw.values() if v)


def label_report(engine_strings: dict[str, str | None], cmap: CategoryMap, priority: list[str]) -> dict[str, Any]:
    """Everything the label service returns for one file."""
    raw = {e: s for e, s in engine_strings.items() if s}
    mapped = {}
    rules = {}
    for e, s in raw.items():
        c, r = cmap.map(e, s)
        mapped[e] = c
        rules[e] = r
    cat, disputed = consensus(mapped, priority)
    agree = len({c for c in mapped.values() if c != UNMAPPED}) <= 1
    return {"raw": raw, "mapped": mapped, "rules": rules, "score": score(raw), "consensus": cat, "disputed": disputed,
            "agreement": agree if len(raw) > 1 else None, "map_version": cmap.version}
