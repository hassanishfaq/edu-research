"""Fine-tune the Chapter 4 RoBERTa models on Google Colab (or the RTX 3090) and download the folders
this demo loads as its model slots.

Colab cell 1  (upload the archive and this script to Drive first):
    !pip -q install transformers datasets accelerate scikit-learn pyyaml
    !cp "/content/drive/MyDrive/multi_av_behaviour_reports.zip" /content/
    !python /content/drive/MyDrive/colab_finetune_roberta.py --archive /content/multi_av_behaviour_reports.zip \
        --demo /content/drive/MyDrive/SigStack-Prototype --slot category --epochs 3 --max-rows 40000

What this script does
  1. rebuilds the serialised rows exactly as the demo does (same normaliser, same category map, same schema),
  2. fine-tunes `roberta-base` (n = 512, batch 8, 500 warm-up steps, weight decay 0.01 -- the Section 4.3 recipe;
     the thesis used 100 epochs with 5-fold stratified CV, `--epochs` and `--folds` reproduce that) on
        --slot category            consensus category labels           -> models/roberta_category
        --slot Baidu|Kaspersky|Symantec   that engine's family labels  -> models/roberta_<engine>
  3. saves the folder with save_pretrained() plus labels.json and metrics.json.
Copy the folder(s) into  SigStack-Prototype/models/  and start  python run.py  (encoder.mode: auto picks them up).
The demo's own package is imported from --demo, so the normaliser used here is the one that serves.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--demo", required=True, help="path of the SigStack-Prototype folder (for sigstack/ and category_map.yaml)")
    ap.add_argument("--slot", default="category", help="category | Baidu | Kaspersky | Symantec")
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5, help="the thesis text prints 0.01; 2e-5 is the usual fine-tuning value")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=1, help=">1 runs stratified k-fold CV and reports the mean; the final model is trained on everything")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--min-family-support", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, args.demo)
    from sigstack import flatten as fl
    from sigstack.config import load_config
    from sigstack.labels import UNMAPPED, CategoryMap, label_report

    cfg = load_config(Path(args.demo) / "config.yaml")
    cmap = CategoryMap(Path(args.demo) / cfg.get("labels.category_map", "category_map.yaml"))
    engines = cfg.engine_folders
    prio = list(cfg.get("labels.engine_priority", engines))
    ncfg = cfg.get("normalizer", {}) or {}
    schema = fl.SCHEMAS[str(ncfg.get("schema_version", "v1"))]

    # ---- rows
    z = zipfile.ZipFile(args.archive)
    strings: dict[str, dict[str, str]] = {}
    loaders: dict[str, zipfile.ZipInfo] = {}
    for info in z.infolist():
        p = info.filename.split("/")
        if len(p) >= 4 and p[-1].endswith(".json") and p[-3] in engines:
            h = p[-1][:-5].lower()
            strings.setdefault(h, {})[p[-3]] = p[-2]
            loaders.setdefault(h, info)
    hashes = sorted(strings)
    texts, labels = [], []
    for h in hashes:
        lab = label_report(strings[h], cmap, prio)
        if args.slot == "category":
            y = lab["consensus"]
            if y == UNMAPPED:
                continue
        else:
            y = strings[h].get(args.slot)
            if not y or cmap.map(args.slot, y)[0] == UNMAPPED:
                continue
        rep = json.loads(z.read(loaders[h]))
        norm = fl.normalise_report(rep, schema, list(ncfg.get("drop_keys", ["success"])), list(ncfg.get("category_order", fl.DEFAULT_ORDER)),
                                   max_items=int(ncfg.get("max_items_per_key", 20)), mask_hash=bool(ncfg.get("mask_sample_hash", True)), max_chars=int(cfg.get("data.text_max_chars", 3000)))
        texts.append(norm["text"])
        labels.append(y)
    cnt = Counter(labels)
    if args.slot != "category":
        keep = {f for f, k in cnt.items() if k >= args.min_family_support}
        texts, labels = zip(*[(t, y) for t, y in zip(texts, labels) if y in keep])
        texts, labels = list(texts), list(labels)
    classes = sorted(set(labels))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[v] for v in labels])
    if args.max_rows and args.max_rows < len(texts):
        from sklearn.model_selection import StratifiedShuffleSplit
        idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=args.max_rows, random_state=42).split(texts, y))
        texts, y = [texts[i] for i in idx], y[idx]
    print(f"slot {args.slot}: {len(texts)} rows, {len(classes)} classes")

    # ---- fine-tuning
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.base)

    def encode(ts, ys):
        ds = Dataset.from_dict({"text": ts, "label": [int(v) for v in ys]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_length), batched=True, remove_columns=["text"])

    def train_one(tr_t, tr_y, ev_t=None, ev_y=None, out_dir="tmp"):
        model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(classes), id2label={i: c for i, c in enumerate(classes)}, label2id=cidx)
        targs = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 4, num_train_epochs=args.epochs,
                                  learning_rate=args.lr, warmup_steps=args.warmup, weight_decay=args.weight_decay, fp16=torch.cuda.is_available(), logging_steps=100,
                                  save_strategy="no", eval_strategy="no", report_to=[])
        trainer = Trainer(model=model, args=targs, train_dataset=encode(tr_t, tr_y), data_collator=DataCollatorWithPadding(tok))
        trainer.train()
        metrics = {}
        if ev_t is not None:
            pred = trainer.predict(encode(ev_t, ev_y)).predictions.argmax(1)
            metrics = {"accuracy": float((pred == ev_y).mean()), "f1_macro": float(f1_score(ev_y, pred, average="macro"))}
        return trainer, metrics

    cv = []
    if args.folds > 1:
        for k, (tr, te) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(texts, y), start=1):
            _, m = train_one([texts[i] for i in tr], y[tr], [texts[i] for i in te], y[te], out_dir=f"tmp_fold{k}")
            print(f"fold {k}: {m}")
            cv.append(m)
    t = time.time()
    trainer, _ = train_one(texts, y, out_dir="tmp_final")
    out = Path(args.out or (Path(args.demo) / "models" / (f"roberta_{args.slot}")))
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(classes), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"slot": args.slot, "rows": len(texts), "classes": len(classes), "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                                                  "warmup": args.warmup, "weight_decay": args.weight_decay, "max_length": args.max_length, "cv": cv,
                                                  "cv_mean_f1": float(np.mean([m["f1_macro"] for m in cv])) if cv else None, "train_seconds": round(time.time() - t)}, indent=1), encoding="utf-8")
    print(f"\nDone. Copy {out} into SigStack-Prototype/models/ (encoder.mode: auto picks it up).")


if __name__ == "__main__":
    main()
