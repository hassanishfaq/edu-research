# Trained models

This folder is intentionally empty in the repository. The files below are **written by the dashboard**, reloaded at every start and not versioned:

| File | Written by | Content |
|---|---|---|
| `caq_edgenet.pt` | *Detector → Train detector* (`POST /api/detector/train`) | state dict, architecture configuration, version `d…`, learning curve, standardisation statistics (≈ 130 KB) |
| `policy.pt` | *Controller → Train controller* (`POST /api/controller/train`) | Q-network, version `p…`, training history, last policy comparison (≈ 40 KB) |

Training both on a CPU takes under two minutes, so the files are not distributed. Delete them to start from scratch. If the architecture in `config.yaml` (`model.*`) changes, retrain the detector: the checkpoint stores its own architecture and is loaded as saved.
