"""Node model definitions — identical to the released thesis code (models.py) so that
checkpoints written by train_node.py load unchanged — plus payload-size helpers.

torch / torchvision are imported lazily: the surrogate path does not need them.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def _torch():
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    return torch, nn, tvm


def get_backbone(name: str = "resnet18", pretrained: bool = False, grayscale: bool = True):
    torch, nn, tvm = _torch()
    if name == "alexnet":
        m = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT if pretrained else None)
        if grayscale:
            m.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        backbone = nn.Sequential(*list(m.features.children()), nn.AdaptiveAvgPool2d((6, 6)))
        return backbone, 256 * 6 * 6
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if grayscale:
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return nn.Sequential(*list(m.children())[:-1]), m.fc.in_features
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        if grayscale:
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return nn.Sequential(*list(m.children())[:-1]), m.fc.in_features
    raise ValueError(f"Unknown backbone: {name}")


def build_embedding_net(backbone: str, grayscale: bool, n_classes: int, pretrained: bool = False, freeze: bool = True):
    torch, nn, _ = _torch()

    class LinearHead(nn.Module):
        def __init__(self, in_dim: int, n_classes: int):
            super().__init__()
            self.fc = nn.Linear(in_dim, n_classes)

        def forward(self, x):
            return self.fc(x)

    class EmbeddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor, emb_dim = get_backbone(backbone, pretrained=pretrained, grayscale=grayscale)
            self.emb_dim = emb_dim
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if freeze:
                for p in self.feature_extractor.parameters():
                    p.requires_grad = False
            self.head = LinearHead(emb_dim, n_classes)

        def forward(self, x):
            feats = self.feature_extractor(x)
            if feats.ndim == 4:
                feats = self.pool(feats)
                feats = feats.view(feats.size(0), -1)
            return self.head(feats), feats

    return EmbeddingNet()


def load_node_checkpoint(path: Path) -> tuple[Any, dict[str, int], dict[str, Any]]:
    """Load a checkpoint saved by train_node.py -> (model in eval mode, class_to_idx, meta)."""
    torch, _, _ = _torch()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = build_embedding_net(ckpt["backbone"], ckpt.get("grayscale", True), ckpt["n_classes"], pretrained=False, freeze=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sd = ckpt["state_dict"]
    meta = {
        "backbone": ckpt["backbone"], "grayscale": ckpt.get("grayscale", True), "n_classes": ckpt["n_classes"],
        "params": int(sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)),
        "bytes_fp32": state_dict_bytes(sd), "bytes_fp16": state_dict_bytes(sd, dtype=torch.float16), "bytes_int8": state_dict_bytes(sd, int8=True),
        "head_bytes_fp32": state_dict_bytes({k: v for k, v in sd.items() if k.startswith("head.")}),
    }
    return model, dict(ckpt["class_to_idx"]), meta


def state_dict_bytes(sd: dict, dtype=None, int8: bool = False) -> int:
    """Serialised size of a state-dict (what a node would actually upload)."""
    torch, _, _ = _torch()
    buf = io.BytesIO()
    out = {}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            out[k] = v
        elif int8:
            scale = v.abs().max() / 127.0 if v.numel() else torch.tensor(1.0)
            out[k] = (v / (scale + 1e-12)).round().clamp(-127, 127).to(torch.int8)
            out[k + ".scale"] = scale.to(torch.float32)
        else:
            out[k] = v.to(dtype) if dtype else v
    torch.save(out, buf)
    return buf.tell()


# Analytical fallbacks (used before checkpoints exist) — parameter counts of the backbones
BACKBONE_PARAMS = {"resnet50": 25_557_032, "resnet18": 11_689_512, "alexnet": 2_469_696}
BACKBONE_GMACS = {"resnet50": 4.1, "resnet18": 1.8, "alexnet": 0.7}
