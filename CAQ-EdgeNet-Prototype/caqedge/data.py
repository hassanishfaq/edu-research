"""Feature aggregation (Section 5.2.1, Table 5.1): the two modalities of CAQ-EdgeNet.

  x_d  dynamic, network-level features: the 46 per-flow features of CIC-IoT-2023 (eight classes, Table 5.2)
  x_s  static / device-context features: the 7 context columns of the MDF-style context data (battery, CPU,
       memory, network activity, active applications, time of day), which also give the busy-state target

Real CIC-IoT-2023 CSVs are used when data.cic_iot_dir holds them; otherwise a synthetic stand-in with the
same 46 features and the class counts of Table 5.2 is generated and labelled as such everywhere in the UI.
The device context is produced by the DeviceContextSim below (or data.context_csv).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ["Benign", "DDoS", "DoS", "Reconnaissance", "Mirai", "Spoofing", "Brute Force", "Web Attack"]
CLASS_GROUP = {"Benign": "Normal Traffic", "DDoS": "Network Attack", "DoS": "Network Attack", "Reconnaissance": "Information Gathering", "Mirai": "IoT Malware",
               "Spoofing": "Identity-Based Attack", "Brute Force": "Credential Attack", "Web Attack": "Application-Layer Attack"}

# The 46 flow features of CIC-IoT-2023 (column names of the released CSVs)
FLOW_FEATURES = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
                 "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
                 "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
                 "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
CONTEXT_FEATURES = ["battery", "cpu", "memory", "net_activity", "active_apps", "hour_sin", "hour_cos"]
STATE_FEATURES = ["battery", "cpu", "latency"]          # S_t of Eq. (5.5)

# CIC-IoT-2023 label -> Table 5.2 class
LABEL_MAP = {
    "BenignTraffic": "Benign",
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS", "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS", "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    "Recon-PingSweep": "Reconnaissance", "Recon-OSScan": "Reconnaissance", "Recon-PortScan": "Reconnaissance", "VulnerabilityScan": "Reconnaissance", "Recon-HostDiscovery": "Reconnaissance",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    "DictionaryBruteForce": "Brute Force",
    "BrowserHijacking": "Web Attack", "Backdoor_Malware": "Web Attack", "CommandInjection": "Web Attack", "SqlInjection": "Web Attack", "XSS": "Web Attack", "Uploading_Attack": "Web Attack",
}


# =============================================================================
# Real CIC-IoT-2023 CSVs
# =============================================================================
def load_cic(cic_dir: Path, class_counts: dict[str, int], max_rows_per_file: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import pandas as pd
    files = sorted(cic_dir.glob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, nrows=max_rows_per_file)
        cols = {c.strip(): c for c in df.columns}
        missing = [c for c in FLOW_FEATURES if c not in cols]
        if missing or "label" not in cols:
            continue
        df = df.rename(columns={v: k for k, v in cols.items()})
        df["cls"] = df["label"].map(LABEL_MAP)
        frames.append(df.dropna(subset=["cls"])[FLOW_FEATURES + ["cls"]])
    if not frames:
        raise RuntimeError("no CIC-IoT-2023 CSV with the 46 flow features and a label column found")
    df = pd.concat(frames, ignore_index=True)
    rng = np.random.RandomState(seed)
    parts = []
    for c in CLASSES:
        sub = df[df["cls"] == c]
        n = min(int(class_counts.get(c, len(sub))), len(sub))
        parts.append(sub.iloc[rng.choice(len(sub), n, replace=False)] if n else sub.iloc[:0])
    df = pd.concat(parts, ignore_index=True)
    X = df[FLOW_FEATURES].to_numpy(dtype=np.float32)
    y = np.array([CLASSES.index(c) for c in df["cls"]], dtype=np.int64)
    return X, y, {"source": "cic-iot-2023", "files": [f.name for f in files], "rows_available": {c: int((df["cls"] == c).sum()) for c in CLASSES}}


# =============================================================================
# Synthetic stand-in flows (class-conditional, same 46 features)
# =============================================================================
# per class: log-normal centres for scale features, Bernoulli rates for indicator / flag features
_SCALE = ["flow_duration", "Header_Length", "Duration", "Rate", "Srate", "Drate", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
          "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
_IND = ["fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
        "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC"]
_BASE = {"flow_duration": 8.0, "Header_Length": 8000.0, "Duration": 64.0, "Rate": 60.0, "Srate": 60.0, "Drate": 0.02, "ack_count": 0.5, "syn_count": 0.5, "fin_count": 0.2, "urg_count": 0.1,
         "rst_count": 0.3, "Tot sum": 6000.0, "Min": 54.0, "Max": 400.0, "AVG": 150.0, "Std": 80.0, "Tot size": 150.0, "IAT": 8e7, "Number": 9.5, "Magnitue": 17.0, "Radius": 110.0,
         "Covariance": 8000.0, "Variance": 0.6, "Weight": 141.0}
_IND_BASE = {"fin_flag_number": 0.1, "syn_flag_number": 0.15, "rst_flag_number": 0.1, "psh_flag_number": 0.3, "ack_flag_number": 0.6, "ece_flag_number": 0.0, "cwr_flag_number": 0.0,
             "HTTP": 0.1, "HTTPS": 0.4, "DNS": 0.25, "Telnet": 0.0, "SMTP": 0.0, "SSH": 0.02, "IRC": 0.0, "TCP": 0.7, "UDP": 0.3, "DHCP": 0.02, "ARP": 0.03, "ICMP": 0.03, "IPv": 1.0, "LLC": 1.0}
_CLASS_MULT: dict[str, dict[str, float]] = {   # multiplicative deviations from the benign centre (scale features)
    "Benign": {},
    "DDoS": {"Rate": 400, "Srate": 400, "flow_duration": 0.02, "Header_Length": 20, "Tot sum": 0.4, "Max": 0.3, "AVG": 0.45, "Std": 0.15, "IAT": 0.6, "Number": 1.1, "Variance": 0.2, "syn_count": 6, "rst_count": 3, "Weight": 1.0},
    "DoS": {"Rate": 120, "Srate": 120, "flow_duration": 0.05, "Header_Length": 12, "Tot sum": 0.45, "Max": 0.35, "AVG": 0.5, "Std": 0.2, "IAT": 0.7, "Variance": 0.25, "syn_count": 4, "rst_count": 2},
    "Reconnaissance": {"Rate": 6, "Srate": 6, "flow_duration": 0.4, "Header_Length": 2.5, "Tot sum": 0.5, "Max": 0.4, "AVG": 0.5, "Std": 0.5, "syn_count": 3, "rst_count": 5, "fin_count": 2, "Number": 1.0, "Variance": 0.6},
    "Mirai": {"Rate": 250, "Srate": 250, "flow_duration": 0.03, "Header_Length": 15, "Tot sum": 1.4, "Max": 1.2, "AVG": 1.3, "Std": 0.4, "IAT": 0.65, "Variance": 0.3, "Drate": 0.2},
    "Spoofing": {"Rate": 0.5, "Srate": 0.5, "flow_duration": 3.0, "Header_Length": 0.6, "Tot sum": 0.4, "Max": 0.3, "AVG": 0.4, "Std": 0.3, "IAT": 1.2, "Duration": 1.5, "Variance": 0.4},
    "Brute Force": {"Rate": 2.5, "Srate": 2.5, "flow_duration": 2.0, "Header_Length": 3.0, "Tot sum": 1.6, "Max": 1.3, "AVG": 1.1, "Std": 1.2, "ack_count": 3, "fin_count": 1.5, "Duration": 1.0},
    "Web Attack": {"Rate": 1.2, "Srate": 1.2, "flow_duration": 1.5, "Header_Length": 2.0, "Tot sum": 3.0, "Max": 2.5, "AVG": 1.8, "Std": 2.0, "ack_count": 2.5, "Variance": 1.3},
}
_CLASS_IND: dict[str, dict[str, float]] = {    # indicator / flag rates per class
    "Benign": {},
    "DDoS": {"syn_flag_number": 0.7, "rst_flag_number": 0.35, "psh_flag_number": 0.15, "ack_flag_number": 0.35, "HTTP": 0.08, "HTTPS": 0.05, "DNS": 0.02, "TCP": 0.6, "UDP": 0.3, "ICMP": 0.15},
    "DoS": {"syn_flag_number": 0.6, "rst_flag_number": 0.25, "psh_flag_number": 0.15, "ack_flag_number": 0.4, "HTTP": 0.15, "HTTPS": 0.05, "DNS": 0.02, "TCP": 0.65, "UDP": 0.3, "ICMP": 0.05},
    "Reconnaissance": {"syn_flag_number": 0.55, "rst_flag_number": 0.5, "fin_flag_number": 0.3, "psh_flag_number": 0.05, "ack_flag_number": 0.3, "HTTP": 0.05, "HTTPS": 0.03, "DNS": 0.05, "TCP": 0.85, "UDP": 0.1, "ICMP": 0.2},
    "Mirai": {"syn_flag_number": 0.05, "rst_flag_number": 0.02, "psh_flag_number": 0.02, "ack_flag_number": 0.05, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0, "TCP": 0.05, "UDP": 0.95, "ICMP": 0.0},
    "Spoofing": {"syn_flag_number": 0.05, "rst_flag_number": 0.03, "psh_flag_number": 0.1, "ack_flag_number": 0.2, "HTTP": 0.03, "HTTPS": 0.05, "DNS": 0.55, "TCP": 0.2, "UDP": 0.35, "ARP": 0.7, "ICMP": 0.1, "LLC": 1.0},
    "Brute Force": {"syn_flag_number": 0.2, "rst_flag_number": 0.15, "psh_flag_number": 0.55, "ack_flag_number": 0.8, "HTTP": 0.05, "HTTPS": 0.02, "DNS": 0.02, "SSH": 0.55, "Telnet": 0.35, "TCP": 0.97, "UDP": 0.02},
    "Web Attack": {"syn_flag_number": 0.2, "rst_flag_number": 0.1, "psh_flag_number": 0.6, "ack_flag_number": 0.85, "HTTP": 0.8, "HTTPS": 0.15, "DNS": 0.05, "TCP": 0.98, "UDP": 0.02},
}


def synth_flows(class_counts: dict[str, int], seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    fidx = {f: i for i, f in enumerate(FLOW_FEATURES)}
    for ci, c in enumerate(CLASSES):
        n = int(class_counts.get(c, 0))
        if n <= 0:
            continue
        X = np.zeros((n, len(FLOW_FEATURES)), dtype=np.float32)
        mult, ind = _CLASS_MULT[c], _CLASS_IND[c]
        # a per-sample "intensity" factor gives correlated deviations inside a class (bursty vs quiet windows)
        intensity = np.exp(rng.normal(0, 0.35, n))
        for f in _SCALE:
            centre = _BASE[f] * mult.get(f, 1.0)
            sig = 0.55 if f in ("Rate", "Srate", "IAT", "Header_Length", "flow_duration") else 0.35
            X[:, fidx[f]] = centre * np.exp(rng.normal(0, sig, n)) * (intensity if f in ("Rate", "Srate", "Header_Length", "syn_count", "rst_count", "ack_count") else 1.0)
        for f in _IND:
            p = ind.get(f, _IND_BASE[f])
            # indicators are window fractions in CIC-IoT-2023: mostly 0/1 with some intermediate values
            frac = rng.binomial(1, p, n).astype(np.float32)
            mix = rng.rand(n) < 0.15
            frac[mix] = np.clip(p + rng.normal(0, 0.15, mix.sum()), 0, 1)
            X[:, fidx[f]] = frac
        proto = np.where(X[:, fidx["UDP"]] > 0.5, 17.0, np.where(X[:, fidx["ICMP"]] > 0.5, 1.0, 6.0)) + rng.normal(0, 0.8, n)
        X[:, fidx["Protocol Type"]] = proto
        X[:, fidx["Tot size"]] = X[:, fidx["AVG"]] * np.exp(rng.normal(0, 0.1, n))
        X[:, fidx["Magnitue"]] = np.sqrt(np.maximum(X[:, fidx["AVG"]], 1e-3)) * np.exp(rng.normal(0, 0.15, n))
        Xs.append(X)
        ys.append(np.full(n, ci, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    # label noise / overlap so the task is not trivially separable (DoS vs DDoS, Recon vs DoS confusions as in real captures)
    flip = rng.rand(len(y)) < 0.012
    y = y.copy()
    y[flip] = rng.randint(0, len(CLASSES), flip.sum())
    return X, y, {"source": "synthetic stand-in", "files": [], "rows_available": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}}


# =============================================================================
# Device context (MDF-style) simulator: also the busy-state target
# =============================================================================
class DeviceContextSim:
    """Autocorrelated device context: battery, CPU, memory, network activity, active applications, hour of day.
    busy state = CPU load in [0, 1]; the forecasting target is the busy state one step ahead."""

    def __init__(self, seed: int = 0, dt_minutes: float = 1.0):
        self.rng = np.random.RandomState(seed)
        self.t = self.rng.uniform(0, 24 * 60)
        self.dt = dt_minutes
        self.battery = self.rng.uniform(0.35, 1.0)
        self.cpu = self.rng.uniform(0.1, 0.5)
        self.memory = self.rng.uniform(0.3, 0.7)
        self.net = self.rng.uniform(0.05, 0.4)
        self.apps = float(self.rng.randint(1, 6))
        self.regime = 0        # 0 quiet, 1 pending burst (an application is starting), 2 busy burst
        self.burst_left = 0
        self.charging = False

    def step(self, extra_drain: float = 0.0) -> np.ndarray:
        r = self.rng
        self.t = (self.t + self.dt) % (24 * 60)
        hour = self.t / 60.0
        # regime switching drives the busy bursts.  A burst is announced one interval ahead by the leading
        # indicators an application launch produces (network activity, one more active application) -- the
        # signal the busy-state head can use and a "last value" forecaster cannot.
        if self.regime == 0 and r.rand() < 0.06:
            self.regime = 1
            self.apps = float(min(self.apps + 1, 9))
        elif self.regime == 1:
            self.regime, self.burst_left = 2, int(r.randint(3, 12))
        elif self.regime == 2:
            self.burst_left -= 1
            if self.burst_left <= 0:
                self.regime = 0
        busy = self.regime == 2
        target = 0.75 + 0.2 * np.sin(hour / 24 * 2 * np.pi) if busy else 0.2 + 0.1 * np.sin(hour / 24 * 2 * np.pi)
        self.cpu = float(np.clip(0.6 * self.cpu + 0.4 * target + r.normal(0, 0.05), 0, 1))
        self.memory = float(np.clip(0.9 * self.memory + 0.1 * (0.4 + 0.4 * self.cpu) + r.normal(0, 0.02), 0, 1))
        net_target = 0.7 if self.regime == 1 else (0.15 + 0.5 * self.cpu * r.rand())
        self.net = float(np.clip(0.5 * self.net + 0.5 * net_target + r.normal(0, 0.05), 0, 1))
        if r.rand() < 0.10:
            self.apps = float(np.clip(self.apps + r.choice([-1, 1]), 1, 9))
        if self.charging:
            self.battery = min(1.0, self.battery + 0.004)
            if self.battery >= 0.99 or r.rand() < 0.02:
                self.charging = False
        else:
            self.battery = max(0.0, self.battery - 0.0008 - 0.002 * self.cpu - extra_drain)
            if self.battery < 0.12 and r.rand() < 0.2:
                self.charging = True
        return self.vector()

    def vector(self) -> np.ndarray:
        hour = self.t / 60.0
        return np.array([self.battery, self.cpu, self.memory, self.net, self.apps / 9.0, np.sin(hour / 24 * 2 * np.pi), np.cos(hour / 24 * 2 * np.pi)], dtype=np.float32)

    def busy(self) -> float:
        return float(self.cpu)


def context_series(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n context vectors and the busy state one step ahead (targets), from several simulated devices."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, len(CONTEXT_FEATURES)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    per_dev = 400
    i = 0
    dev = 0
    while i < n:
        sim = DeviceContextSim(seed=seed * 1000 + dev)
        for _ in range(rng.randint(20, 60)):
            sim.step()
        for _ in range(per_dev):
            if i >= n:
                break
            X[i] = sim.vector()
            sim.step()
            y[i] = sim.busy()
            i += 1
        dev += 1
    return X, y


# =============================================================================
# Dataset assembly + cache
# =============================================================================
def build_dataset(cfg, log=print) -> dict[str, Any]:
    t = time.time()
    seed = int(cfg.get("data.seed", 42))
    counts = {k: int(v) for k, v in (cfg.get("data.class_counts", {}) or {}).items()}
    cic = cfg.cic_dir
    if cic and cic.exists() and any(cic.glob("*.csv")):
        log(f"loading CIC-IoT-2023 CSVs from {cic} ...")
        Xf, y, info = load_cic(cic, counts, int(cfg.get("data.max_rows_per_file", 150000)), seed)
    else:
        log(f"CIC-IoT-2023 folder not found ({cic}); generating the synthetic stand-in flows with the class counts of Table 5.2")
        Xf, y, info = synth_flows(counts, seed)
    n = len(y)
    ctx_csv = cfg.resolve(cfg.get("data.context_csv"))
    if ctx_csv and ctx_csv.exists():
        import pandas as pd
        df = pd.read_csv(ctx_csv)
        Xc = df[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
        yb = df["busy_next"].to_numpy(dtype=np.float32) if "busy_next" in df else np.roll(Xc[:, 1], -1)
        idx = np.random.RandomState(seed).randint(0, len(Xc), n)
        Xc, yb = Xc[idx], yb[idx]
        info["context"] = str(ctx_csv)
    else:
        Xc, yb = context_series(n, seed)
        info["context"] = "simulated device context (MDF-style)"
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    Xf, Xc, y, yb = Xf[perm], Xc[perm], y[perm], yb[perm]
    # stratified split
    split = cfg.get("data.split", {}) or {}
    tr, va = float(split.get("train", 0.7)), float(split.get("val", 0.15))
    part = np.zeros(n, dtype=np.int8)
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        a, b = int(len(idx) * tr), int(len(idx) * (tr + va))
        part[idx[a:b]] = 1
        part[idx[b:]] = 2
    # standardisation fitted on the training split (log1p on heavy-tailed scale features)
    Xf_t = np.log1p(np.maximum(Xf, 0))
    mu, sd = Xf_t[part == 0].mean(0), Xf_t[part == 0].std(0) + 1e-6
    log(f"dataset ready: {n} flows ({info['source']}), {len(CLASSES)} classes, split {int((part == 0).sum())}/{int((part == 1).sum())}/{int((part == 2).sum())} in {time.time() - t:.1f} s")
    return {"X_flow_raw": Xf, "X_flow": ((Xf_t - mu) / sd).astype(np.float32), "X_ctx": Xc, "y": y, "y_busy": yb, "part": part, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
            "classes": CLASSES, "source": info["source"], "info": info, "built": time.time()}


def save_dataset(d: dict[str, Any], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_flow_raw=d["X_flow_raw"], X_flow=d["X_flow"], X_ctx=d["X_ctx"], y=d["y"], y_busy=d["y_busy"], part=d["part"], mu=d["mu"], sd=d["sd"],
                        meta=np.array(json.dumps({"source": d["source"], "info": d["info"], "built": d["built"]})))


def load_dataset(path: Path) -> dict[str, Any] | None:
    import json
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {k: z[k] for k in ("X_flow_raw", "X_flow", "X_ctx", "y", "y_busy", "part", "mu", "sd")} | {"classes": CLASSES, **meta}


def standardise_flow(raw: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.log1p(np.maximum(raw, 0)) - mu) / sd).astype(np.float32)
