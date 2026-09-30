"""Write a tiny random-weight RoBERTa folder in the format the demo loads, to exercise the RoBERTa
model slot without a GPU or a download:

    python scripts/make_test_checkpoint.py                # -> models/roberta_category (2 layers, d = 64)
    python scripts/make_test_checkpoint.py --slot Baidu   # -> models/roberta_Baidu (family labels of that engine)
    python run.py                                         # the slot now reports kind = roberta

The predictions of such a model are meaningless (random weights); delete the folder to return to the
surrogate head, or replace it with the folder written by scripts/colab_finetune_roberta.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sigstack.config import load_config      # noqa: E402
from sigstack.labels import CategoryMap, UNMAPPED   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="category")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--full-size", action="store_true", help="use the real roberta-base geometry (12 layers, d = 768; 500 MB, still random weights)")
    args = ap.parse_args()
    from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

    cfg = load_config()
    cmap = CategoryMap(cfg.category_map_path)
    if args.slot == "category":
        classes = list(cmap.categories)
    else:
        c = pickle.load(open(cfg.cache_path, "rb"))
        cnt = Counter(s for s in c["engine_strings"][args.slot] if s and cmap.map(args.slot, s)[0] != UNMAPPED)
        classes = sorted(f for f, k in cnt.items() if k >= int(cfg.get("train.min_family_support", 30)))
    tok = AutoTokenizer.from_pretrained(str(cfg.get("encoder.roberta.base_model", "roberta-base")))
    if args.full_size:
        rc = RobertaConfig.from_pretrained("roberta-base", num_labels=len(classes))
    else:
        rc = RobertaConfig(vocab_size=tok.vocab_size, hidden_size=args.hidden, num_hidden_layers=args.layers, num_attention_heads=4, intermediate_size=args.hidden * 4,
                           max_position_embeddings=514, type_vocab_size=1, num_labels=len(classes), pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    rc.id2label = {i: c for i, c in enumerate(classes)}
    rc.label2id = {c: i for i, c in enumerate(classes)}
    model = RobertaForSequenceClassification(rc)
    out = cfg.roberta_dir(args.slot)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(classes), encoding="utf-8")
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote random-weight test model with {len(classes)} labels ({n / 1e6:.1f} M params) to {out}")


if __name__ == "__main__":
    main()
