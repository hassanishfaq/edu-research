# Fine-tuned models

This folder is intentionally empty in the repository. Fine-tuned models are not versioned.

| Folder | Slot | Written by |
|---|---|---|
| `roberta_category/` | direct category model (9 standardised categories) | `scripts/colab_finetune_roberta.py --slot category` |
| `roberta_Baidu/` | Baidu family model | `scripts/colab_finetune_roberta.py --slot Baidu` |
| `roberta_Kaspersky/` | Kaspersky family model | `scripts/colab_finetune_roberta.py --slot Kaspersky` |
| `roberta_Symantec/` | Symantec family model | `scripts/colab_finetune_roberta.py --slot Symantec` |

Each folder is a Hugging Face `save_pretrained()` directory:

```
roberta_category/
├── config.json
├── model.safetensors
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, special_tokens_map.json
├── labels.json        # class names in output order
└── metrics.json       # training settings and cross-validation results
```

With `encoder.mode: auto` in `config.yaml`, every slot whose folder exists switches to RoBERTa at the next start. The other slots keep the CPU stand-in.

To test the loader without fine-tuning (random weights, predictions are noise):

```bash
python scripts/make_test_checkpoint.py                 # tiny 2-layer model in models/roberta_category
python scripts/make_test_checkpoint.py --slot Baidu    # family labels of that engine
python scripts/make_test_checkpoint.py --full-size     # real roberta-base geometry (≈ 500 MB)
```
