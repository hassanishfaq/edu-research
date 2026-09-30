"""Build the compact image cache and the demo sample folders from the dataset folders in config.yaml.

Usage:  python scripts/prepare_data.py            (re-run after changing data.* in config.yaml)

Produces
  data/cache/thumbs<N>.npz   X (n,N,N) uint8 thumbnails, y labels, split codes, class names,
                             plus the zero-day families as X_zd / y_zd / zd_classes
  data/samples/known/<family>/*.png      a few full-resolution demo images per family
  data/samples/zero_day/<family>/*.png   a few images per unseen family
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedstack.config import load_config, _norm_class  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def list_classes(root: Path) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            out.append((_norm_class(d.name), d))
    return out


def load_thumb(p: Path, size: int) -> np.ndarray:
    im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def main() -> None:
    cfg = load_config()
    size = int(cfg.get("data.thumbnail_size", 64))
    split = cfg.get("data.split", {"train": 350, "val": 75, "test": 75})
    n_samples = int(cfg.get("data.samples_per_class", 4))
    rng = np.random.default_rng(int(cfg.get("surrogate.seed", 42)))

    train_dir = cfg.training_dir
    if not train_dir or not train_dir.exists():
        sys.exit(f"data.training_images_dir not found: {train_dir}")

    t0 = time.time()
    X, y, s, files, classes = [], [], [], [], []
    samples_known = cfg.samples_dir / "known"
    for ci, (cname, cdir) in enumerate(list_classes(train_dir)):
        imgs = sorted(p for p in cdir.iterdir() if p.suffix.lower() in IMG_EXT)
        order = rng.permutation(len(imgs))
        n_tr, n_va, n_te = split["train"], split["val"], split["test"]
        codes = np.full(len(imgs), 2, dtype=np.int8)
        codes[order[:n_tr]] = 0
        codes[order[n_tr:n_tr + n_va]] = 1
        codes[order[n_tr + n_va:n_tr + n_va + n_te]] = 2
        keep = order[:n_tr + n_va + n_te]
        classes.append(cname)
        for k in keep:
            X.append(load_thumb(imgs[k], size)); y.append(ci); s.append(codes[k]); files.append(imgs[k].relative_to(train_dir).as_posix())
        dest = samples_known / cname
        dest.mkdir(parents=True, exist_ok=True)
        for k in order[n_tr + n_va:n_tr + n_va + n_samples]:  # samples come from the test split
            shutil.copy2(imgs[k], dest / imgs[k].name)
        print(f"  [{ci + 1:2d}/28] {cname:40s} {len(keep)} images", flush=True)

    X = np.stack(X); y = np.asarray(y, dtype=np.int16); s = np.asarray(s, dtype=np.int8)

    # unseen families
    Xz, yz, fz, zclasses = [], [], [], []
    zd = cfg.zero_day_dir
    samples_zd = cfg.samples_dir / "zero_day"
    if zd and zd.exists():
        for zi, (zname, zdir) in enumerate(list_classes(zd)):
            imgs = sorted(p for p in zdir.iterdir() if p.suffix.lower() in IMG_EXT)
            imgs.sort(key=lambda p: p.stat().st_size)
            zclasses.append(zname)
            for p in imgs[:120]:
                Xz.append(load_thumb(p, size)); yz.append(zi); fz.append(p.relative_to(zd).as_posix())
            dest = samples_zd / zname
            dest.mkdir(parents=True, exist_ok=True)
            for p in imgs[:max(n_samples, 6)]:
                shutil.copy2(p, dest / p.name)
            print(f"  [zero-day] {zname:20s} {min(len(imgs), 120)} images", flush=True)
    Xz = np.stack(Xz) if Xz else np.zeros((0, size, size), np.uint8)

    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.cache_path, X=X, y=y, split=s, files=np.asarray(files), classes=np.asarray(classes),
                        X_zd=Xz, y_zd=np.asarray(yz, dtype=np.int16), zd_files=np.asarray(fz), zd_classes=np.asarray(zclasses),
                        size=size)
    mb = os.path.getsize(cfg.cache_path) / 1e6
    print(f"cache written: {cfg.cache_path}  ({mb:.1f} MB, {len(X)} known + {len(Xz)} unseen images, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
