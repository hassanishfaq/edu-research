"""CAQ-EdgeNet (Section 5.2) in PyTorch, plus the precision runtimes used for deployment measurements.

  Multi-modal encoder     h_s = sigma(W_s x_s + b_s),  h_d = sigma(W_d x_d + b_d)          Eq. (5.2)
  Fusion                  z   = sigma(W_f [h_s ; h_d] + b_f)                                Eq. (5.3)
  Conditional attention   a   = sigmoid(W_a [z ; S_t] + b_a),  z' = 2 a * z   (S_t = battery, CPU, latency; "FusionCAB")
  Quantisation-aware      s   = (max z' - min z') / (2^b - 1),  z_hat = round(z'/s) * s     Eq. (5.4)  (straight-through in training)
  Classifier              logits over the eight classes of Table 5.2
  Busy-state head         forecast of the busy state one step ahead (multi-task, Table 5.3)

The device runs the encoder part (local and partial paths), the edge runs the classifier on z_hat
(partial path) or the whole model (edge path) -- see simulator.py.

`NumpyRuntime` executes the trained weights at a per-layer precision (32 / 16 / 8 / 4 bits, Eq. 5.4
per tensor, integer accumulation for the 8-bit layers) so that size, latency and accuracy of FP32,
FP16, uniform INT8 and the hybrid scheme of Section 5.2.3 are measured, not assumed.
"""
from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_ORDER = ["enc_d", "enc_s", "fuse", "attn", "cls_h", "cls_h2", "cls_out", "busy_h", "busy_out"]
DEVICE_LAYERS = ["enc_d", "enc_s", "fuse", "attn"]          # what the partial path keeps on the device


class FakeQuant(torch.autograd.Function):
    """Eq. (5.4) with a straight-through estimator."""

    @staticmethod
    def forward(ctx, z, bits):
        s = (z.max() - z.min()) / (2 ** bits - 1) + 1e-8
        return torch.round(z / s) * s

    @staticmethod
    def backward(ctx, g):
        return g, None


class CAQEdgeNet(nn.Module):
    def __init__(self, d_flow: int = 46, d_ctx: int = 7, d_state: int = 3, h_flow: int = 64, h_ctx: int = 16, d_z: int = 64, h_cls: int = 32, n_classes: int = 8,
                 attention: bool = True, qae: bool = True, bits: int = 8, busy_head: bool = True, use_ctx: bool = True):
        super().__init__()
        self.cfg = dict(d_flow=d_flow, d_ctx=d_ctx, d_state=d_state, h_flow=h_flow, h_ctx=h_ctx, d_z=d_z, h_cls=h_cls, n_classes=n_classes, attention=attention, qae=qae, bits=bits, busy_head=busy_head, use_ctx=use_ctx)
        self.enc_d = nn.Linear(d_flow, h_flow)
        self.enc_s = nn.Linear(d_ctx, h_ctx)
        self.fuse = nn.Linear(h_flow + (h_ctx if use_ctx else 0), d_z)
        self.attn = nn.Linear(d_z + d_state, d_z)
        self.cls_h = nn.Linear(d_z, h_cls)
        self.cls_h2 = nn.Linear(h_cls, h_cls // 2)
        self.cls_out = nn.Linear(h_cls // 2, n_classes)
        self.busy_h = nn.Linear(d_z, 16)
        self.busy_out = nn.Linear(16, 1)
        self.attention, self.qae, self.bits, self.busy_head, self.use_ctx = attention, qae, bits, busy_head, use_ctx

    # ---- device side: encoders -> fusion -> conditional attention -> quantisation
    def encode(self, x_flow, x_ctx, state):
        h_d = F.relu(self.enc_d(x_flow))
        if self.use_ctx:
            h_s = F.relu(self.enc_s(x_ctx))
            z = F.relu(self.fuse(torch.cat([h_d, h_s], dim=1)))
        else:
            z = F.relu(self.fuse(h_d))
        if self.attention:
            gate = torch.sigmoid(self.attn(torch.cat([z, state], dim=1)))
            zp = 2.0 * gate * z
        else:
            gate = torch.full_like(z, 0.5)
            zp = z
        zq = FakeQuant.apply(zp, self.bits) if self.qae else zp
        return zq, gate, z

    # ---- edge / device side: classifier on the (quantised) embedding; busy-state head on the shared fused embedding z
    def heads(self, zq, z=None):
        logits = self.cls_out(F.relu(self.cls_h2(F.relu(self.cls_h(zq)))))
        zb = z if z is not None else zq
        busy = torch.sigmoid(self.busy_out(F.relu(self.busy_h(zb)))).squeeze(1) if self.busy_head else torch.zeros(zq.shape[0], device=zq.device)
        return logits, busy

    def forward(self, x_flow, x_ctx, state):
        zq, gate, z = self.encode(x_flow, x_ctx, state)
        logits, busy = self.heads(zq, z)
        return logits, busy, zq, gate, z

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


# =============================================================================
# Training
# =============================================================================
def train_model(model: CAQEdgeNet, data: dict[str, Any], epochs: int, batch: int, lr: float, wd: float, lambda_busy: float, seed: int = 42,
                progress=None, state_sampler=None) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xf, Xc, y, yb, part = data["X_flow"], data["X_ctx"], data["y"], data["y_busy"], data["part"]
    tr, va = np.where(part == 0)[0], np.where(part == 1)[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    hist = []
    Xf_t, Xc_t, y_t, yb_t = torch.from_numpy(Xf), torch.from_numpy(Xc), torch.from_numpy(y), torch.from_numpy(yb)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = rng.permutation(tr)
        tot = tot_cls = tot_busy = 0.0
        nb = 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            st = torch.from_numpy(sample_states(len(idx), rng, Xc[idx]))
            logits, busy, *_ = model(Xf_t[idx], Xc_t[idx], st)
            l_cls = F.cross_entropy(logits, y_t[idx])
            l_busy = F.mse_loss(busy, yb_t[idx]) if model.busy_head else torch.zeros(())
            loss = l_cls + lambda_busy * l_busy
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); tot_cls += float(l_cls); tot_busy += float(l_busy); nb += 1
        sched.step()
        ev = evaluate(model, data, va, rng)
        row = {"epoch": ep, "loss": tot / nb, "loss_cls": tot_cls / nb, "loss_busy": tot_busy / nb, "val_accuracy": ev["accuracy"], "val_f1": ev["f1_macro"], "val_busy_mae": ev["busy_mae"],
               "ms": round((time.perf_counter() - t0) * 1000)}
        hist.append(row)
        if progress:
            progress(row)
    return hist


def sample_states(n: int, rng: np.random.RandomState, ctx: np.ndarray | None = None) -> np.ndarray:
    """Device states S_t = (battery, CPU, latency) seen during training: battery / CPU from the context row, latency random."""
    st = np.zeros((n, 3), dtype=np.float32)
    if ctx is not None:
        st[:, 0] = ctx[:, 0]
        st[:, 1] = ctx[:, 1]
    else:
        st[:, 0] = rng.uniform(0, 1, n)
        st[:, 1] = rng.uniform(0, 1, n)
    st[:, 2] = np.clip(rng.exponential(0.25, n), 0, 1)     # normalised RTT (1 = 250 ms or worse)
    return st


def predict(model: CAQEdgeNet, Xf: np.ndarray, Xc: np.ndarray, states: np.ndarray, batch: int = 4096):
    model.eval()
    out_p, out_b = [], []
    with torch.no_grad():
        for i in range(0, len(Xf), batch):
            logits, busy, *_ = model(torch.from_numpy(Xf[i:i + batch]), torch.from_numpy(Xc[i:i + batch]), torch.from_numpy(states[i:i + batch]))
            out_p.append(torch.softmax(logits, 1).numpy())
            out_b.append(busy.numpy())
    return np.concatenate(out_p), np.concatenate(out_b)


def evaluate(model: CAQEdgeNet, data: dict[str, Any], idx: np.ndarray, rng: np.random.RandomState | None = None) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
    rng = rng or np.random.RandomState(0)
    st = sample_states(len(idx), rng, data["X_ctx"][idx])
    P, B = predict(model, data["X_flow"][idx], data["X_ctx"][idx], st)
    y, yb = data["y"][idx], data["y_busy"][idx]
    pred = P.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(P.shape[1])), zero_division=0)
    try:
        auc = float(roc_auc_score(y, P, multi_class="ovr"))
    except Exception:
        auc = None
    err = B - yb
    return {"accuracy": float((pred == y).mean()), "f1_macro": float(f1_score(y, pred, average="macro")), "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "auc": auc,
            "per_class": [{"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(len(p))],
            "confusion": confusion(y, pred, P.shape[1]), "busy_mae": float(np.abs(err).mean()), "busy_rmse": float(np.sqrt((err ** 2).mean())),
            "busy_mape": float((np.abs(err) / np.maximum(np.abs(yb), 0.05)).mean() * 100), "busy_r2": float(1 - (err ** 2).sum() / max(((yb - yb.mean()) ** 2).sum(), 1e-9)),
            "busy_evs": float(1 - err.var() / max(yb.var(), 1e-9)), "n": int(len(idx))}


def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> list[list[int]]:
    m = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m.tolist()


# =============================================================================
# Precision runtimes (numpy): FP32 / FP16 / INT8 / INT4 per layer, Eq. (5.4)
# =============================================================================
def quantise_tensor(w: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantisation with Eq. (5.4)'s scale: returns integer codes and the scale."""
    qmax = 2 ** (bits - 1) - 1
    s = float(np.abs(w).max()) / qmax + 1e-12
    q = np.clip(np.round(w / s), -qmax, qmax)
    return q.astype(np.int8 if bits <= 8 else np.int16), s


class NumpyRuntime:
    """Runs a trained CAQEdgeNet with a per-layer precision map, e.g. {"enc_d": 8, "attn": 32, ...}.
    32 -> float32 weights and activations; 16 -> float16 weights (float32 compute, as a CPU without half kernels);
    8 / 4 -> integer weight codes with integer accumulation over 8-bit-quantised activations, then rescale."""

    def __init__(self, model: CAQEdgeNet, precision: dict[str, int]):
        self.cfg = dict(model.cfg)
        self.precision = {k: int(precision.get(k, 32)) for k in LAYER_ORDER}
        self.layers: dict[str, dict[str, Any]] = {}
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        for name in LAYER_ORDER:
            W, b = sd[f"{name}.weight"], sd[f"{name}.bias"]
            bits = self.precision[name]
            if bits >= 32:
                self.layers[name] = {"W": W.astype(np.float32), "b": b.astype(np.float32), "bits": 32}
            elif bits == 16:
                self.layers[name] = {"W": W.astype(np.float16), "b": b.astype(np.float16), "bits": 16}
            else:
                q, s = quantise_tensor(W, bits)
                self.layers[name] = {"Wq": q, "s": s, "b": b.astype(np.float32), "bits": bits}
        self.attention, self.qae, self.bits, self.busy_head, self.use_ctx = model.attention, model.qae, model.bits, model.busy_head, model.use_ctx

    def _linear(self, name: str, x: np.ndarray) -> np.ndarray:
        L = self.layers[name]
        if L["bits"] >= 32:
            return x @ L["W"].T + L["b"]
        if L["bits"] == 16:
            return (x.astype(np.float16) @ L["W"].T).astype(np.float32) + L["b"].astype(np.float32)
        # integer path: quantise the activation to 8 bits, accumulate in int32, rescale
        xq, sx = quantise_tensor(x, 8)
        acc = xq.astype(np.int32) @ L["Wq"].astype(np.int32).T
        return acc.astype(np.float32) * (sx * L["s"]) + L["b"]

    def encode(self, x_flow: np.ndarray, x_ctx: np.ndarray, state: np.ndarray):
        h_d = np.maximum(self._linear("enc_d", x_flow), 0)
        if self.use_ctx:
            h_s = np.maximum(self._linear("enc_s", x_ctx), 0)
            z = np.maximum(self._linear("fuse", np.concatenate([h_d, h_s], 1)), 0)
        else:
            z = np.maximum(self._linear("fuse", h_d), 0)
        if self.attention:
            gate = 1 / (1 + np.exp(-self._linear("attn", np.concatenate([z, state], 1))))
            zp = 2.0 * gate * z
        else:
            gate = np.full_like(z, 0.5)
            zp = z
        if self.qae:
            s = (zp.max() - zp.min()) / (2 ** self.bits - 1) + 1e-8
            zq = np.round(zp / s) * s
        else:
            zq = zp
        return zq.astype(np.float32), gate, z

    def heads(self, zq: np.ndarray, z: np.ndarray | None = None):
        logits = self._linear("cls_out", np.maximum(self._linear("cls_h2", np.maximum(self._linear("cls_h", zq), 0)), 0))
        zb = z if z is not None else zq
        busy = 1 / (1 + np.exp(-self._linear("busy_out", np.maximum(self._linear("busy_h", zb), 0))))[:, 0] if self.busy_head else np.zeros(len(zq), dtype=np.float32)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), busy

    def forward(self, x_flow, x_ctx, state):
        zq, gate, z = self.encode(x_flow, x_ctx, state)
        P, busy = self.heads(zq, z)
        return P, busy, zq, gate, z

    # ---- deployment measurements
    def size_bytes(self, part: str = "all") -> int:
        names = LAYER_ORDER if part == "all" else (DEVICE_LAYERS if part == "device" else [n for n in LAYER_ORDER if n not in DEVICE_LAYERS])
        tot = 0
        for n in names:
            L = self.layers[n]
            bits = L["bits"]
            W = L["W"] if "W" in L else L["Wq"]
            tot += (W.size * bits + 7) // 8 + L["b"].size * 4 + (4 if "s" in L else 0)
        return int(tot)

    def latency_ms(self, x_flow, x_ctx, state, repeats: int = 300, part: str = "all") -> float:
        f = (lambda: self.forward(x_flow[:1], x_ctx[:1], state[:1])) if part == "all" else (lambda: self.encode(x_flow[:1], x_ctx[:1], state[:1])) if part == "device" else (lambda: self.heads(self._zq))
        if part == "edge_heads":
            self._zq = self.encode(x_flow[:1], x_ctx[:1], state[:1])[0]
        for _ in range(30):
            f()
        t = time.perf_counter()
        for _ in range(repeats):
            f()
        return (time.perf_counter() - t) / repeats * 1000

    def accuracy(self, Xf, Xc, states, y, batch: int = 4096) -> float:
        pred = np.concatenate([self.forward(Xf[i:i + batch], Xc[i:i + batch], states[i:i + batch])[0].argmax(1) for i in range(0, len(Xf), batch)])
        return float((pred == y).mean())

    def describe(self) -> dict[str, int]:
        return dict(self.precision)


def state_dict_bytes(model: nn.Module, dtype=None) -> int:
    buf = io.BytesIO()
    sd = {k: (v.to(dtype) if dtype and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, buf)
    return buf.tell()


def layer_table(model: CAQEdgeNet) -> list[dict[str, Any]]:
    out = []
    for name in LAYER_ORDER:
        lin = getattr(model, name)
        out.append({"layer": name, "role": LAYER_ROLE[name], "in": lin.in_features, "out": lin.out_features, "params": lin.weight.numel() + lin.bias.numel()})
    return out


LAYER_ROLE = {"enc_d": "dynamic (flow) encoder, Eq. (5.2)", "enc_s": "static / context encoder, Eq. (5.2)", "fuse": "fusion, Eq. (5.3)", "attn": "conditional attention gate (device state)",
              "cls_h": "classifier hidden 1", "cls_h2": "classifier hidden 2", "cls_out": "classifier output (8 classes)", "busy_h": "busy-state head hidden", "busy_out": "busy-state head output"}
