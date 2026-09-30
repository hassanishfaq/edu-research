"""Build the report cache and the demo sample folders from the behaviour-report archive in config.yaml.

Usage:  python scripts/prepare_data.py            (re-run after changing data.* or normalizer.* in config.yaml)

Reads the archive in place (no extraction).  For every unique file hash it
  * collects the raw verdict string of each engine that flagged the file        (label service input),
  * normalises the nested report: flatten -> filter -> align to the schema -> serialise   (Eqs. 4.5-4.10),
  * counts the RoBERTa tokens of the serialised row                                  (n = 512 policy),
and writes
  data/cache/reports.pkl       hashes, texts, token counts, event counts, per-engine strings, schema version
  data/samples/<category>/     a few raw reports per standardised category (+ index.json with their verdicts)
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sigstack import flatten as fl                     # noqa: E402
from sigstack.config import ROOT, load_config          # noqa: E402
from sigstack.labels import UNMAPPED, CategoryMap, label_report   # noqa: E402
from sigstack.tokenizer import TokenCounter            # noqa: E402


def iter_reports(cfg):
    """Yield (engine_folder, family, hash, bytes-loader) for every report in the archive or folder."""
    engines = set(cfg.engine_folders)
    arc = cfg.archive
    if arc and arc.exists():
        z = zipfile.ZipFile(arc)
        for info in z.infolist():
            p = info.filename.split("/")
            if len(p) >= 4 and p[-1].endswith(".json") and p[-3] in engines:
                yield p[-3], p[-2], p[-1][:-5].lower(), (lambda i=info: z.read(i))
        return
    d = cfg.reports_dir
    if d and d.exists():
        for eng in sorted(engines):
            for fam in sorted((d / eng).glob("*")):
                if fam.is_dir():
                    for f in fam.glob("*.json"):
                        yield eng, fam.name, f.stem.lower(), (lambda f=f: f.read_bytes())
        return
    raise SystemExit(f"No data: set data.archive or data.reports_dir in {cfg.path}")


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    engines = cfg.engine_folders
    print(f"scanning reports of {engines} ...")
    strings: dict[str, dict[str, str]] = defaultdict(dict)
    loaders: dict[str, object] = {}
    n_reports = 0
    for eng, fam, h, load in iter_reports(cfg):
        strings[h][eng] = fam
        loaders.setdefault(h, load)
        n_reports += 1
    hashes = sorted(strings)
    print(f"  {n_reports} reports -> {len(hashes)} unique hashes in {time.time() - t0:.0f}s")
    max_h = int(cfg.get("data.max_hashes", 0) or 0)
    if max_h and max_h < len(hashes):
        rnd = random.Random(42)
        rnd.shuffle(hashes)
        hashes = sorted(hashes[:max_h])
        print(f"  deterministic sample of {len(hashes)} hashes (data.max_hashes)")

    schema = fl.SCHEMAS[str(cfg.get("normalizer.schema_version", "v1"))]
    drop = list(cfg.get("normalizer.drop_keys", ["success", "pid"]))
    order = list(cfg.get("normalizer.category_order", fl.DEFAULT_ORDER))
    max_items = int(cfg.get("normalizer.max_items_per_key", 40))
    mask = bool(cfg.get("normalizer.mask_sample_hash", True))
    max_chars = int(cfg.get("data.text_max_chars", 3000))

    texts: list[str] = []
    n_values = np.zeros(len(hashes), dtype=np.int32)
    present = np.zeros(len(hashes), dtype=np.int16)
    cat_counts = np.zeros((len(hashes), len(fl.DEFAULT_ORDER)), dtype=np.int32)
    bad = 0
    t1 = time.time()
    for i, h in enumerate(hashes):
        try:
            rep = json.loads(loaders[h]())
        except Exception:
            rep, bad = {}, bad + 1
        norm = fl.normalise_report(rep, schema, drop, order, max_items=max_items, mask_hash=mask, max_chars=max_chars)
        texts.append(norm["text"])
        n_values[i] = norm["stats"]["n_values"]
        present[i] = norm["stats"]["present"]
        cc = norm["category_counts"]
        cat_counts[i] = [cc.get(c, 0) for c in fl.DEFAULT_ORDER]
        if (i + 1) % 10000 == 0:
            print(f"  normalised {i + 1}/{len(hashes)} ({time.time() - t1:.0f}s)")
    print(f"  normalised {len(hashes)} reports in {time.time() - t1:.0f}s ({bad} unreadable)")

    tc = TokenCounter(cfg.get("encoder.tokenizer", "roberta-base"))
    t2 = time.time()
    n_tokens = np.array(tc.count_many(texts), dtype=np.int32)
    print(f"  token counts ({tc.kind}) in {time.time() - t2:.0f}s: median {int(np.median(n_tokens))}, "
          f"p90 {int(np.percentile(n_tokens, 90))}, share > 512: {(n_tokens > 512).mean() * 100:.1f}%")

    eng_strings = {e: [strings[h].get(e) for h in hashes] for e in engines}
    cache = {"hashes": hashes, "texts": texts, "n_tokens": n_tokens, "n_values": n_values, "present": present, "cat_counts": cat_counts,
             "cat_order": fl.DEFAULT_ORDER, "engines": engines, "engine_strings": eng_strings, "schema_version": str(cfg.get("normalizer.schema_version", "v1")),
             "text_max_chars": max_chars, "built": time.time(), "source": str(cfg.archive or cfg.reports_dir)}
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cache written: {cfg.cache_path} ({cfg.cache_path.stat().st_size / 1e6:.0f} MB)")

    # ---- demo samples per standardised category (raw reports + their verdicts)
    cmap = CategoryMap(cfg.category_map_path)
    prio = list(cfg.get("labels.engine_priority", engines))
    per_cat: dict[str, list[str]] = defaultdict(list)
    rnd = random.Random(7)
    order_h = list(hashes)
    rnd.shuffle(order_h)
    k = int(cfg.get("data.samples_per_category", 5))
    for h in order_h:
        lab = label_report(strings[h], cmap, prio)
        c = lab["consensus"]
        if c == UNMAPPED or len(per_cat[c]) >= k:
            continue
        i = hashes.index(h) if len(hashes) < 200000 else None
        per_cat[c].append(h)
        if all(len(v) >= k for v in per_cat.values()) and len(per_cat) >= len(cmap.categories):
            break
    sdir = cfg.samples_dir
    if sdir.exists():
        import shutil
        shutil.rmtree(sdir)
    index = []
    for c, hs in sorted(per_cat.items()):
        d = sdir / c.replace(" ", "_").replace("/", "-")
        d.mkdir(parents=True, exist_ok=True)
        for h in hs:
            (d / f"{h}.json").write_bytes(loaders[h]())
            index.append({"category": c, "hash": h, "path": f"{d.name}/{h}.json", "engines": strings[h]})
    (sdir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"  {len(index)} sample reports written to {sdir} ({len(per_cat)} categories)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
