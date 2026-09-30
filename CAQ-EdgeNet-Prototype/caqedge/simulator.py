"""Device / network simulator and the cost model of the three execution paths (Sections 5.2.4, 5.3.5, 5.4.3).

  S_t = (battery_t, CPU_t, latency_t)                                  Eq. (5.5)
  a_t in {local, partial, edge}                                         Eq. (5.6)
  r_t = lambda1 * Accuracy_t - lambda2 * Latency_t - lambda3 * Energy_t Eq. (5.7)  (latency / energy normalised by reference values)

Path costs (Table 5.12): what runs on the device, what crosses the link, what runs at the edge.
  local    collect + queueing behind the device's own load + INT8 inference on the device;   nothing transmitted
  partial  collect + (smaller) queueing + INT8 encoder on the device; z_hat (d_z x b bits) up; FP32 heads at the edge
  edge     collect; the 53 feature values (FP32) up; the full FP32 model at the edge (service time grows with edge load)
Device-side inference times are the FP32 latency measured on this machine scaled to the device profile
(anchored by default to the 12.6 ms of Table 5.7) and divided by the precision speed-up of a mobile
integer runtime; the numpy runtimes measure the *sizes* and the *accuracy* of each precision here.
Every other quantity (collection, contention, service, power) is a documented simulator parameter of
config.yaml that a field prototype replaces with measurements (Section 5.4.5, guideline 5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import DeviceContextSim

ACTIONS = ["local", "partial", "edge"]
LOCAL, PARTIAL, EDGE = 0, 1, 2
POLICY_FEATURES = ["battery", "cpu", "latency_norm", "bandwidth_norm", "link_up", "busy_forecast", "charging"]


class NetworkSim:
    """Link to the edge server (RTT, bandwidth, outages, congestion) and the load of the shared edge server."""

    def __init__(self, cfg: dict[str, Any], rng: np.random.RandomState):
        self.cfg, self.rng = cfg, rng
        self.base = float(cfg.get("base_rtt_ms", 25)); self.jit = float(cfg.get("rtt_jitter_ms", 6)); self.cong = float(cfg.get("congestion_rtt_ms", 140))
        self.bw_nom = float(cfg.get("bandwidth_mbps", 4)); self.p_down = float(cfg.get("link_down_prob", 0.03)); self.outage = cfg.get("outage_steps", [3, 12])
        self.link_up = True; self.down_left = 0; self.congested = 0
        self.rtt = self.base; self.bw = self.bw_nom
        self.edge_load = float(rng.uniform(0.1, 0.5))
        self.step()

    def step(self) -> None:
        r = self.rng
        if self.link_up and r.rand() < self.p_down:
            self.link_up, self.down_left = False, int(r.randint(int(self.outage[0]), int(self.outage[1]) + 1))
        elif not self.link_up:
            self.down_left -= 1
            if self.down_left <= 0:
                self.link_up = True
        if self.congested == 0 and r.rand() < 0.05:
            self.congested = int(r.randint(2, 8))
        elif self.congested > 0:
            self.congested -= 1
        base = self.cong if self.congested else self.base
        self.rtt = float(max(2.0, base + r.normal(0, self.jit)))
        self.bw = float(max(0.3, self.bw_nom * (0.25 if self.congested else 1.0) * np.exp(r.normal(0, 0.15))))
        target = 0.85 if r.rand() < 0.08 else 0.3
        self.edge_load = float(np.clip(0.8 * self.edge_load + 0.2 * target + r.normal(0, 0.05), 0, 1))


class DeviceSim:
    """One monitored device: context (MDF-style), battery, CPU, link."""

    def __init__(self, dev_id: int, net_cfg: dict[str, Any], seed: int, battery_wh: float):
        self.id = dev_id
        self.rng = np.random.RandomState(seed)
        self.ctx = DeviceContextSim(seed=seed)
        for _ in range(int(self.rng.randint(5, 40))):
            self.ctx.step()
        self.net = NetworkSim(net_cfg, self.rng)
        self.battery_wh = battery_wh
        self.runtime_ok = True

    # S_t of Eq. (5.5): the latency component is the observed round trip to the edge service (link + edge load)
    def observed_latency_ms(self, cost: "CostModel | None" = None) -> float:
        return self.net.rtt + (cost.edge_service_ms("full", self.net.edge_load) if cost else 0.0)

    def state(self, cost: "CostModel | None" = None) -> np.ndarray:
        return np.array([self.ctx.battery, self.ctx.cpu, min(self.observed_latency_ms(cost) / 250.0, 1.0)], dtype=np.float32)

    def context(self) -> np.ndarray:
        return self.ctx.vector()

    def policy_features(self, busy_forecast: float, cost: "CostModel | None" = None) -> np.ndarray:
        return np.array([self.ctx.battery, self.ctx.cpu, min(self.observed_latency_ms(cost) / 250.0, 1.0), min(self.net.bw / 10.0, 1.0), 1.0 if self.net.link_up else 0.0, busy_forecast, 1.0 if self.ctx.charging else 0.0], dtype=np.float32)

    def advance(self) -> None:
        """Move the device and the link one interval ahead (the request is served during this interval)."""
        self.ctx.step()
        self.net.step()

    def drain(self, energy_j: float) -> None:
        self.ctx.battery = max(0.0, self.ctx.battery - energy_j / (self.battery_wh * 3600.0))

    def step(self, energy_j: float) -> None:
        self.advance()
        self.drain(energy_j)

    def snapshot(self) -> dict[str, Any]:
        return {"id": self.id, "battery": round(self.ctx.battery, 3), "cpu": round(self.ctx.cpu, 3), "memory": round(self.ctx.memory, 3), "rtt_ms": round(self.net.rtt, 1), "bandwidth_mbps": round(self.net.bw, 2),
                "link_up": self.net.link_up, "charging": self.ctx.charging, "congested": bool(self.net.congested), "edge_load": round(self.net.edge_load, 2), "runtime_ok": self.runtime_ok}


class CostModel:
    def __init__(self, dev_cfg: dict[str, Any], net_cfg: dict[str, Any], ctl_cfg: dict[str, Any], timings: dict[str, float], sizes: dict[str, int], speedups: dict[str, float]):
        self.dev, self.net, self.ctl = dev_cfg, net_cfg, ctl_cfg
        self.t = {k: float(v) for k, v in timings.items()}   # measured on this machine: fp32_full_ms, fp32_encoder_ms, fp32_heads_ms
        self.sizes = sizes                                   # z_bytes, feature_bytes
        self.speed = speedups                                # device runtime speed-up per precision
        ref = float(dev_cfg.get("reference_fp32_ms", 12.6))
        factor = dev_cfg.get("compute_factor", "auto")
        self.factor = float(factor) if isinstance(factor, (int, float)) else ref / max(self.t["fp32_full_ms"], 1e-6)
        self.lam = (float(ctl_cfg.get("lambda1", 1.0)), float(ctl_cfg.get("lambda2", 0.6)), float(ctl_cfg.get("lambda3", 0.6)))
        self.lat_ref, self.e_ref = float(ctl_cfg.get("latency_ref_ms", 150)), float(ctl_cfg.get("energy_ref_j", 0.15))

    # device-side inference times (ms) after scaling
    def device_ms(self, part: str, precision: str) -> float:
        base = self.t["fp32_full_ms"] if part == "full" else self.t["fp32_encoder_ms"]
        return base * self.factor / float(self.speed.get(precision, 1.0))

    def edge_service_ms(self, part: str, load: float) -> float:
        """Service time at the shared edge server (deserialise, schedule, infer): grows with the server's load."""
        base = float(self.net.get("edge_service_full_ms", 18)) if part == "full" else float(self.net.get("edge_service_heads_ms", 4))
        return base * (1.0 + 3.0 * load)

    def tx_ms(self, n_bytes: int, bw_mbps: float) -> float:
        return n_bytes * 8 / (bw_mbps * 1e6) * 1000 + float(self.net.get("message_overhead_ms", 2.0))

    def feasible(self, dev: DeviceSim) -> list[bool]:
        return [dev.runtime_ok, dev.net.link_up and dev.runtime_ok, dev.net.link_up]

    def path(self, action: int, dev: DeviceSim, expected: bool = False) -> dict[str, Any]:
        """Latency / energy breakdown of one path in the device's current state.  expected=True uses the nominal link."""
        d = self.dev
        cpu = float(dev.ctx.cpu)
        rtt = dev.net.base if expected else dev.net.rtt
        bw = dev.net.bw_nom if expected else dev.net.bw
        load = 0.3 if expected else dev.net.edge_load
        collect = float(d.get("collect_ms", 35))
        ser = float(d.get("serialize_ms", 1.5))
        slow = 1.0 + float(d.get("cpu_slowdown", 25)) * cpu * cpu             # time-sharing with the device's own load stretches every on-device job
        P_act, P_inf, P_idle, P_radio = float(d.get("power_active_w", 2.4)), float(d.get("power_inference_w", 3.5)), float(d.get("power_idle_w", 0.3)), float(d.get("power_radio_w", 1.6))
        if action == LOCAL:
            work = self.device_ms("full", "int8")
            inf = work * slow
            lat = collect + inf
            energy = P_act * collect / 1000 + P_inf * work / 1000 + P_idle * (inf - work) / 1000
            return {"latency_ms": float(lat), "energy_j": float(energy), "bytes": 0, "breakdown": {"collect": collect, "inference": float(work), "contention": float(inf - work)}}
        if action == PARTIAL:
            work = self.device_ms("encoder", "int8")
            enc = work * slow
            tx = self.tx_ms(self.sizes["z_bytes"], bw)
            edge = self.edge_service_ms("heads", load)
            lat = collect + enc + ser + tx + rtt + edge
            energy = P_act * (collect + ser) / 1000 + P_inf * work / 1000 + P_idle * (enc - work) / 1000 + P_radio * tx / 1000 + P_idle * (rtt + edge) / 1000
            return {"latency_ms": float(lat), "energy_j": float(energy), "bytes": self.sizes["z_bytes"], "breakdown": {"collect": collect, "encoder": float(work), "contention": float(enc - work), "serialise": ser, "transmit": float(tx), "rtt": float(rtt), "edge": float(edge)}}
        tx = self.tx_ms(self.sizes["feature_bytes"], bw)
        edge = self.edge_service_ms("full", load)
        lat = collect + ser + tx + rtt + edge
        energy = P_act * (collect + ser) / 1000 + P_radio * tx / 1000 + P_idle * (rtt + edge) / 1000
        return {"latency_ms": float(lat), "energy_j": float(energy), "bytes": self.sizes["feature_bytes"], "breakdown": {"collect": collect, "serialise": ser, "transmit": float(tx), "rtt": float(rtt), "edge": float(edge)}}

    def reward(self, correct: float, lat: float, energy: float) -> float:
        l1, l2, l3 = self.lam
        return float(l1 * correct - l2 * lat / self.lat_ref - l3 * energy / self.e_ref)

    def cost(self, lat: float, energy: float) -> float:
        _, l2, l3 = self.lam
        return float(l2 * lat / self.lat_ref + l3 * energy / self.e_ref)

    def describe(self) -> dict[str, Any]:
        return {"compute_factor": round(self.factor, 1), "device_full_int8_ms": round(self.device_ms("full", "int8"), 2), "device_full_fp32_ms": round(self.device_ms("full", "fp32"), 2),
                "device_encoder_int8_ms": round(self.device_ms("encoder", "int8"), 2), "edge_full_ms_nominal": round(self.edge_service_ms("full", 0.3), 2), "edge_heads_ms_nominal": round(self.edge_service_ms("heads", 0.3), 2),
                "z_bytes": self.sizes["z_bytes"], "feature_bytes": self.sizes["feature_bytes"], "lambdas": list(self.lam), "latency_ref_ms": self.lat_ref, "energy_ref_j": self.e_ref, "measured_ms": self.t, "speedups": self.speed,
                "device": self.dev, "network": self.net}


class Env:
    """One device, one episode: at every step a flow window arrives, the policy chooses a path, the cost model bills it.
    `predictors` maps a path to a function (x_flow, x_ctx, state, label) -> correct (0/1) using the precision that path executes."""

    def __init__(self, cost: CostModel, predictors: dict[int, Any], flows: np.ndarray, ctxs_unused: np.ndarray, labels: np.ndarray, net_cfg: dict[str, Any], battery_wh: float,
                 busy_forecaster=None, seed: int = 0, dev_id: int = 0, steps: int = 120):
        self.cost, self.pred, self.flows, self.labels, self.net_cfg, self.battery_wh = cost, predictors, flows, labels, net_cfg, battery_wh
        self.busy = busy_forecaster
        self.seed, self.dev_id, self.steps = seed, dev_id, steps
        self.dev: DeviceSim | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed = self.seed if seed is None else seed
        self.rng = np.random.RandomState(self.seed)
        self.dev = DeviceSim(self.dev_id, self.net_cfg, self.seed, self.battery_wh)
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        self._i = int(self.rng.randint(0, len(self.flows)))
        self._xf, self._xc, self._st = self.flows[self._i:self._i + 1], self.dev.context()[None], self.dev.state(self.cost)[None]
        fc = float(self.busy(self._xf, self._xc, self._st)) if self.busy else float(self.dev.ctx.cpu)
        self._fc = fc
        return self.dev.policy_features(fc, self.cost)

    def mask(self) -> list[bool]:
        return self.cost.feasible(self.dev)

    def step(self, action: int) -> tuple[np.ndarray, float, dict[str, Any]]:
        m = self.mask()
        label = int(self.labels[self._i])
        # the decision is taken now, but the request is served during the coming interval: the device and the link
        # move first and the path is billed at the load then in force (why the busy-state forecast matters, Table 5.3)
        self.dev.advance()
        outcomes: list[dict[str, Any] | None] = []
        for a in range(3):
            if not m[a]:
                outcomes.append(None)
                continue
            p = self.cost.path(a, self.dev)
            correct = float(self.pred[a](self._xf, self._xc, self._st, label))
            outcomes.append({**p, "correct": correct, "reward": self.cost.reward(correct, p["latency_ms"], p["energy_j"])})
        if outcomes[action] is None:                       # infeasible choice -> deterministic fallback (Section 5.4.4 rule 2)
            action = next((a for a in (LOCAL, EDGE, PARTIAL) if outcomes[a] is not None), LOCAL)
            if outcomes[action] is None:
                p = self.cost.path(LOCAL, self.dev)
                outcomes[LOCAL] = {**p, "correct": 0.0, "reward": self.cost.reward(0.0, p["latency_ms"], p["energy_j"])}
                action = LOCAL
        o = outcomes[action]
        oracle = int(np.argmax([(-1e9 if x is None else x["reward"]) for x in outcomes]))
        self.dev.drain(o["energy_j"])
        self.t += 1
        info = {"action": action, "oracle": oracle, "latency_ms": o["latency_ms"], "energy_j": o["energy_j"], "correct": o["correct"], "bytes": o["bytes"], "reward": o["reward"],
                "mask": m, "busy_forecast": self._fc, "busy_actual_next": float(self.dev.ctx.cpu), "state": self.dev.snapshot(), "label": int(self.labels[self._i]), "flow_index": self._i,
                "breakdown": o["breakdown"], "all": [None if x is None else {"latency_ms": x["latency_ms"], "energy_j": x["energy_j"], "reward": x["reward"], "correct": x["correct"]} for x in outcomes]}
        done = self.t >= self.steps
        return self._obs(), o["reward"], {**info, "done": done}
