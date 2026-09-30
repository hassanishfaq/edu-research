# Configuration reference

Behaviour is driven by [`config.yaml`](../config.yaml) and the category rules in [`category_map.yaml`](../category_map.yaml). Paths may be absolute or relative to the repository folder. Start the server with another file using `python run.py --config my_config.yaml`. Keep API keys in such a local copy and never commit them.

## `encoder`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `auto` | `auto`: each slot uses RoBERTa if its folder exists (`config.json` present), otherwise the stand-in. `roberta`: same, but warns about missing folders. `surrogate`: stand-in for every slot. |
| `roberta.base_model` | `roberta-base` | Informational; the base of the fine-tuned folders. |
| `roberta.max_length` | `512` | Sequence length $n$: truncation at inference and the token-meter cut. |
| `roberta.category_model_dir` | `models/roberta_category` | Folder of the direct category model. |
| `roberta.engine_model_dir_pattern` | `models/roberta_{engine}` | Folder of each engine's family model; `{engine}` is the engine folder name. |
| `roberta.int8_dynamic` | `false` | Apply PyTorch dynamic INT8 quantisation to the linear layers (for CPU hosts). |
| `roberta.batch_size` | `8` | Inference batch size of the RoBERTa slots. |
| `surrogate.n_features` | `131072` | Hashing space ($2^{17}$) of the stand-in feature extractor. |
| `surrogate.ngram_max` | `2` | Word n-gram range 1…`ngram_max`. |
| `surrogate.svd_components` | `256` | Dimension of the shared latent space (truncated SVD). |
| `surrogate.head` | `logreg` | Head of each stand-in slot: `logreg` or `mlp`. |
| `surrogate.logreg_C` / `surrogate.mlp_hidden` | `4.0` / `256` | Head hyper-parameters. |
| `surrogate.seed` | `42` | Seed of the SVD and the heads. |
| `tokenizer` | `roberta-base` | Tokenizer used for token counting in every mode. An empty string selects an approximate splitter (no download). |

## `data`

| Key | Default | Meaning |
|---|---|---|
| `archive` | `datasets/multi_av_behaviour_reports.zip` | Zip archive `<root>/<engine>/<family>/<md5>.json`, read in place. |
| `reports_dir` | `""` | Alternative: a folder with the same layout (used when `archive` is empty or missing). |
| `engines[*].folder` | `Baidu`, `Kaspersky`, `Symantec` | Engine folder names in the archive. |
| `engines[*].display` | same | Name shown on the dashboard. |
| `max_hashes` | `0` | 0 uses every unique hash. A smaller number takes a deterministic sample (seed 42). |
| `text_max_chars` | `3000` | Characters of serialised text kept per report (more than 512 tokens). |
| `samples_per_category` | `5` | Demo reports copied to `data/samples/` per category. |

Changing `data.*` or `normalizer.*` requires re-running `python scripts/prepare_data.py`.

## `normalizer`

| Key | Default | Meaning |
|---|---|---|
| `schema_version` | `v1` | Key set $C$ (`sigstack/flatten.py: SCHEMAS`). It is recorded with every verdict. |
| `drop_keys` | `[success]` | $K_{\text{unnecessary}}$ of Eq. 4.7: last key components removed after flattening. |
| `max_items_per_key` | `20` | Distinct values kept per key, in order of appearance. |
| `category_order` | hooking … hosts_file | Serialisation order, which decides what survives truncation at $n$ tokens. |
| `mask_sample_hash` | `true` | Replace the sample's own SHA-256 name by `<sample>`. System and user path prefixes are always canonicalised. |

## `labels`

| Key | Default | Meaning |
|---|---|---|
| `category_map` | `category_map.yaml` | Versioned rule file (see below). |
| `consensus` | `majority` | Majority of the mapped engine categories (the only rule implemented). |
| `engine_priority` | `[Kaspersky, Symantec, Baidu]` | Tie-breaking order of the consensus. |
| `min_engines` | `1` | Minimum Eq. 4.4 score for a file to enter the labelled corpus (3 = unanimous detection). |
| `report_groups` | nine → six | Grouping of the categories into the six families of thesis Table 4.10 (reporting only). |

## `train`

| Key | Default | Meaning |
|---|---|---|
| `folds` | `5` | Stratified $k$ of the cross-validation. |
| `max_samples` | `30000` | Stratified cap on the labelled corpus for the live CPU run (0 = all 75,926). |
| `min_family_support` | `30` | Minimum labelled reports for a vendor family to become a class of an engine's family model. |
| `meta_learner` | `logreg` | Stacking meta-learner: `logreg` or `rf`. |
| `logreg_C` | `1.0` | Regularisation of the logistic-regression meta-learner. |
| `roberta_eval_rows` | `600` | Test rows scored per fold when a slot runs RoBERTa on the CPU. |
| `holdout_rows` | `12000` | Stratified sample used by the category hold-out study. |
| `seed` | `42` | Seed of the sample, the folds and the hold-out split. |

## `open_set`

| Key | Default | Meaning |
|---|---|---|
| `method` | `max_prob` | Confidence score: maximum probability, or `entropy` ($1 - H(\hat p)/\log K$). |
| `threshold` | `auto` | A fixed number, or `auto` to calibrate τ₄ out-of-fold on known data. |
| `known_reject_budget` | `0.10` | The false-alarm budget $\beta$ of Eq. 4.11. |

## `acquisition`

| Key | Default | Meaning |
|---|---|---|
| `virustotal_api_key` | `""` | Optional. When set, an unknown `<hash>.txt` is fetched from the VirusTotal v3 API and cached. Keep keys out of version control. |
| `cache_dir` | `data/cache/reports` | Report cache by hash; a report is never fetched twice. |

## `watch` and `server`

| Key | Default | Meaning |
|---|---|---|
| `watch.dir` | `watch_inbox` | Folder watched by the intake agent. |
| `watch.poll_seconds` | `1.0` | Polling interval. A file is analysed once its size has stopped changing. |
| `watch.extensions` | `.json .txt` | File types picked up. |
| `server.host` / `server.port` | `127.0.0.1` / `8766` | Bind address and port. |
| `server.open_browser` | `true` | Open the dashboard at start-up (`--no-browser` overrides it). |

## `category_map.yaml`

```yaml
version: v1                      # bump on every edit; stored with every verdict
categories: [Adware, Backdoor Trojans, Worms, Downloaders, PUA, Infostealers, Trojans, Viruses, Fake Security Tools]
rules:
  Symantec:                      # engine folder name
    - { match: '^Adware\.|^Download\.Adware', category: Adware }
    - { match: '^PUA\.',                      category: PUA }
    # ... first matching rule wins; no match -> Unmapped (shown as a map gap)
```

- Rules are case-insensitive regular expressions, evaluated in order.
- Category names must be members of `categories`.
- After an edit, restart the server. The corpus labels are re-derived from the stored vendor strings, and a saved model trained under another map version is discarded automatically.
