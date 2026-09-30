# Trained models

This folder is intentionally empty in the repository. Trained checkpoints are not versioned.

| File | Written by | Required |
|---|---|---|
| `node_1_model.pt` … `node_4_model.pt` | `train_node.py` of the research code ([github.com/jhshah101/Zero-Day-Malware-Detection](https://github.com/jhshah101/Zero-Day-Malware-Detection)), or `scripts/colab_train_nodes.py` | for the ResNet path |
| `stacker.joblib` | `server_stack.py` of the research code | optional; otherwise the server fits the meta-learner on its validation pool |

Each node checkpoint is a `torch.save` dictionary with the following keys:

```python
{
    "state_dict":   ...,                  # EmbeddingNet weights (backbone + linear head)
    "class_to_idx": {"<family>": 0, ...}, # the node's seven families
    "backbone":     "resnet50",           # alexnet | resnet18 | resnet50
    "grayscale":    True,                 # single-channel first convolution
    "n_classes":    7,
}
```

With `mode: auto` in `config.yaml`, the server switches to these checkpoints as soon as all four files exist.

To test the loader without trained models:

```bash
python scripts/make_test_checkpoint.py --backbone resnet50 --out models
```

This writes checkpoints with **random weights**. They are useful for measuring latency, payload and memory, but their predictions are noise.
