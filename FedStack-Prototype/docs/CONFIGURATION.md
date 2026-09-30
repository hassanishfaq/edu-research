# Configuration reference

All behaviour is driven by [`config.yaml`](../config.yaml). Paths may be absolute or relative to the repository folder. Start the server with a different file using `python run.py --config my_config.yaml`.

## `mode`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `auto` | `auto` uses the ResNet checkpoints when **every** checkpoint listed under `model.nodes` exists, and the stand-in nodes otherwise. `resnet` forces the checkpoints (it falls back to stand-ins with a warning if they cannot be loaded). `surrogate` always uses the stand-ins. |

## `model`

| Key | Default | Meaning |
|---|---|---|
| `backbone` | `resnet50` | Backbone of the reference rows in the payload table: `alexnet`, `resnet18` or `resnet50`. The loaded checkpoints carry their own backbone name. |
| `image_size` | `224` | Input resolution of the backbone. |
| `grayscale` | `true` | Informational. The research code uses a one-channel first convolution, and the loader reads this flag from each checkpoint. |
| `accuracy_weighting` | `true` | Use the accuracy-aware node weights of Eq. 3.15; `false` gives uniform weights. |
| `quantize_int8_for_transfer` | `true` | Payload accounting counts INT8 weights per round (`false` counts FP32). |
| `nodes[*].name` | `node_1` … `node_4` | Node identifier. It must match a key of `federation.partition`. |
| `nodes[*].label` | text | Human-readable specialisation shown on the dashboard. |
| `nodes[*].checkpoint` | `models/node_k_model.pt` | Checkpoint written by `train_node.py`. |
| `stacker` | `rf` | Server meta-learner: `rf` (random forest) or `logreg` (logistic regression, as in the thesis and the lowest latency). |
| `rf_trees` | `300` | Number of trees of the random-forest stacker. |
| `logreg_C` | `10.0` | Inverse regularisation strength of the logistic-regression stacker. |
| `stacker_file` | `models/stacker.joblib` | Optional pre-fitted stacker `{"stacker", "vocab", "metrics"}` written by `server_stack.py`. |

## `data`

| Key | Default | Meaning |
|---|---|---|
| `training_images_dir` | `datasets/Short_Dataset` | Folder with one sub-folder of images per known family. |
| `zero_day_dir` | `datasets/External` | Folder with one sub-folder per family never seen by any node (optional). |
| `samples_per_class` | `4` | Demo images copied to `data/samples/` per family. |
| `thumbnail_size` | `64` | Resolution of the thumbnail cache used by the stand-in nodes. |
| `split.train` / `split.val` / `split.test` | `350` / `75` / `75` | Images per family in each split (seed 42). Validation images are never used to train a node; the server fits the meta-learner on them. |

Changing `data.*` requires re-running `python scripts/prepare_data.py`.

## `federation`

| Key | Default | Meaning |
|---|---|---|
| `rounds` | `20` | Communication rounds $T$. |
| `local_epochs` | `5` | Local epochs $E$ per round (stand-in nodes). |
| `partition` | 4 × 7 families | Class-specialist assignment of the families to the nodes. Folder names are compared with whitespace removed. |

## `surrogate`

Settings of the CPU stand-in nodes, which are used until checkpoints are present.

| Key | Default | Meaning |
|---|---|---|
| `pca_components` | `96` | Size of each node's private PCA feature space. |
| `hidden_units` | `256` | Width of the node's neural head. |
| `learning_rate` | `0.0005` | Adam learning rate of the head. |
| `seed` | `42` | Seed for the data split, PCA and head. |

## `open_set`

| Key | Default | Meaning |
|---|---|---|
| `method` | `max_prob` | Confidence score: the maximum meta-learner probability, or `entropy` ($1 - H(\hat p)/\log C$). |
| `threshold` | `auto` | A fixed number, or `auto` to calibrate $\tau$ out-of-fold on known validation data. |
| `known_reject_budget` | `0.10` | The false-alarm budget $\beta$: the share of known validation samples that $\tau$ is allowed to reject. |

## `watch`

| Key | Default | Meaning |
|---|---|---|
| `dir` | `watch_inbox` | Folder watched by the endpoint agent. |
| `poll_seconds` | `1.0` | Polling interval. A file is scanned once its size has stopped changing. |
| `quarantine` | `false` | Move every scanned file to `quarantine/`. Files are moved, never deleted. |
| `extensions` | `.exe .dll .sys .scr .pdf .bin .png .jpg` | File types picked up by the watcher. |

## `server`

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Bind address. Keep the loopback address unless the dashboard must be reachable from other machines. |
| `port` | `8765` | HTTP port. |
| `open_browser` | `true` | Open the dashboard in the default browser at start-up (`--no-browser` overrides it). |
