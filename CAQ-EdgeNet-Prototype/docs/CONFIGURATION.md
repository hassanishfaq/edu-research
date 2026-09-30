# Configuration reference

All behaviour is driven by [`config.yaml`](../config.yaml). Paths may be absolute or relative to the repository folder. Start the server with another file using `python run.py --config my_config.yaml`. Every device, network and power value is a **simulator parameter**. A field prototype replaces these values with measurements from the target hardware.

## `data`

| Key | Default | Meaning |
|---|---|---|
| `cic_iot_dir` | `datasets/CIC-IoT-2023` | Folder of CIC-IoT-2023 CSV files. If it holds no CSV, the synthetic stand-in is generated. |
| `max_rows_per_file` | `150000` | Rows read per CSV file. |
| `context_csv` | `""` | Optional CSV with the seven context columns (and optionally `busy_next`) that replaces the simulator. |
| `class_counts` | Table 5.2 | Samples per class after mapping (sub-sampling for CIC-IoT-2023, generation for the stand-in). |
| `split` | `0.70 / 0.15 / 0.15` | Stratified train / validation / test fractions. |
| `seed` | `42` | Seed of generation, sampling and split. |

Delete `data/cache/dataset.npz` after changing `data.*`.

## `model`

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `models/caq_edgenet.pt` | Detector written by *Train detector* and reloaded at start. |
| `policy` | `models/policy.pt` | Controller written by *Train controller* and reloaded at start. |
| `h_flow` / `h_ctx` | `64` / `16` | Widths of the flow and context encoders (Eq. 5.2). |
| `d_z` | `64` | Fused embedding size (Eq. 5.3). This is also what the partial path transmits. |
| `h_cls` | `128` | Classifier hidden width (two hidden layers, `h_cls` and `h_cls/2`). |
| `conditional_attention` | `true` | Device-state-keyed attention gate (FusionCAB). |
| `qae` / `qae_bits` | `true` / `8` | Quantisation-aware compression of the embedding and its bit width $b$ (Eq. 5.4). |
| `busy_head` | `true` | Busy-state forecasting head (multi-task). |
| `lambda_busy` | `1.0` | Weight of the busy-state loss. |

## `training`

| Key | Default | Meaning |
|---|---|---|
| `epochs` | `12` | Epochs of *Train detector*. |
| `batch_size` | `256` | Minibatch size. |
| `lr` / `weight_decay` | `0.002` / `0.0001` | Adam settings, with cosine annealing. |
| `ablation_epochs` | `5` | Epochs per ablation variant. |

## `quantization`

| Key | Default | Meaning |
|---|---|---|
| `sensitivity_threshold_pts` | `0.30` | A layer whose 8-bit quantisation costs more than this many accuracy points keeps 16 bits. |
| `protected_layers` | `[attn, cls_out]` | Layers always kept at 16 bits (attention gate, final classifier). |
| `latency_repeats` | `400` | Batch-1 forward passes averaged per precision when measuring latency. |

## `device_profile`

| Key | Default | Meaning |
|---|---|---|
| `name` | IIoT gateway class | Label shown on the dashboard. |
| `reference_fp32_ms` | `12.6` | On-device FP32 inference time of the full model (thesis Table 5.7). |
| `compute_factor` | `auto` | Scaling from this machine's measured FP32 time to the device. `auto` anchors it to `reference_fp32_ms`; a number sets it directly. |
| `collect_ms` | `35.0` | Flow-window feature export and context sampling. |
| `serialize_ms` | `1.5` | Message serialisation. |
| `cpu_slowdown` | `25.0` | On-device work stretches by $1 + 25\,\text{CPU}^2$ under the device's own load. |
| `power_active_w` / `power_inference_w` / `power_idle_w` / `power_radio_w` | `2.4` / `3.5` / `0.3` / `1.6` | Power in each activity state (W). |
| `battery_wh` | `7.4` | Battery capacity (2,000 mAh at 3.7 V). |

## `network`

| Key | Default | Meaning |
|---|---|---|
| `base_rtt_ms` / `rtt_jitter_ms` | `25` / `6` | Nominal round-trip time to the edge and its jitter. |
| `congestion_rtt_ms` | `140` | RTT during a congestion episode (the bandwidth drops to a quarter). |
| `bandwidth_mbps` | `4.0` | Nominal uplink. |
| `message_overhead_ms` | `2.0` | Fixed per-message transmission overhead. |
| `edge_service_full_ms` / `edge_service_heads_ms` | `28` / `3` | Edge service time at zero load for a full-model or heads-only request. It is multiplied by $1 + 3\,\text{load}$. |
| `link_down_prob` / `outage_steps` | `0.03` / `[3, 12]` | Probability that an outage starts at a step, and its duration range. |

## `controller`

| Key | Default | Meaning |
|---|---|---|
| `lambda1` / `lambda2` / `lambda3` | `1.0` / `0.6` / `0.6` | Reward weights of accuracy, latency and energy (Eq. 5.7). These are the operator's policy knobs. |
| `latency_ref_ms` / `energy_ref_j` | `150` / `0.15` | Normalisation of the latency and energy terms. |
| `gamma` | `0.95` | Discount factor (Eq. 5.8). |
| `episodes` / `steps_per_episode` | `200` / `120` | DQN training length. |
| `epsilon_start` / `epsilon_end` | `1.0` / `0.05` | Linear exploration schedule. |
| `hysteresis` | `2` | Consecutive decisions a new action must persist before the fleet switches to it. |
| `eval_episodes` | `12` | Held-out episodes of the policy comparison. |

## `simulation` and `server`

| Key | Default | Meaning |
|---|---|---|
| `simulation.n_devices` | `6` | Devices in the live fleet. |
| `simulation.tick_seconds` | `0.7` | Wall-clock interval between fleet steps. |
| `simulation.telemetry_file` | `data/telemetry.jsonl` | Append-only telemetry and reward log. |
| `server.host` / `server.port` | `127.0.0.1` / `8767` | Bind address and port. |
| `server.open_browser` | `true` | Open the dashboard at start-up (`--no-browser` overrides it). |
