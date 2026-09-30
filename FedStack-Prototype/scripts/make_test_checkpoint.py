"""Create RANDOMLY INITIALISED node checkpoints in exactly the format train_node.py writes,
so the ResNet code path (loading, alignment, payload accounting, scanning) can be exercised
before the real trained models arrive.  They are NOT trained models — predictions are noise.

Usage:  python scripts/make_test_checkpoint.py [--backbone resnet18|resnet50] [--out models/_test]
Then point config.yaml -> model.nodes[*].checkpoint at models/_test/node_X_model.pt  (or copy them to ./models).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedstack.config import load_config  # noqa: E402
from fedstack.models import build_embedding_net  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--out", default="models/_test")
    args = ap.parse_args()
    import torch
    cfg = load_config()
    out = cfg.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fams in cfg.partition.items():
        model = build_embedding_net(args.backbone, True, len(fams), pretrained=False, freeze=True)
        path = out / f"{name}_model.pt"
        torch.save({"state_dict": model.state_dict(), "class_to_idx": {c: i for i, c in enumerate(fams)},
                    "backbone": args.backbone, "grayscale": True, "n_classes": len(fams)}, path)
        print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(fams)} classes, RANDOM weights)")


if __name__ == "__main__":
    main()
