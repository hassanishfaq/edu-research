"""Offloading controllers (Section 5.2.4) and the baselines of Table 5.4.

  proposed   DQN over the device state plus the busy-state forecast of the detector (the joint head of
             Table 5.3): Q(s, a) learned from the reward of Eq. (5.7), objective J(pi) of Eq. (5.8)
  baselines  random, rule-based thresholds, cost threshold, greedy (immediate estimated reward, no forecast),
             tabular Q-learning
Every policy is applied through the action mask (infeasible actions removed) and, in the live fleet,
through the hysteresis rule of Section 5.4.4.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .simulator import ACTIONS, EDGE, LOCAL, PARTIAL, Env

N_FEAT = 7


def masked_argmax(scores: np.ndarray, mask: list[bool]) -> int:
    s = np.array(scores, dtype=np.float64)
    s[~np.array(mask)] = -1e9
    return int(s.argmax()) if any(mask) else LOCAL


class Policy:
    name = "policy"
    kind = "baseline"

    def act(self, f: np.ndarray, mask: list[bool], env: Env | None = None) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomPolicy(Policy):
    name = "Random policy"

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def act(self, f, mask, env=None):
        ok = [a for a in range(3) if mask[a]]
        return int(self.rng.choice(ok)) if ok else LOCAL


class RulePolicy(Policy):
    """Static thresholds (the deterministic fallback of Section 5.4.4): battery low and link up -> edge;
    CPU busy and link up -> partial; otherwise local."""
    name = "Rule-based"

    def __init__(self, battery_low: float = 0.30, cpu_busy: float = 0.60):
        self.bl, self.cb = battery_low, cpu_busy

    def act(self, f, mask, env=None):
        battery, cpu, link = f[0], f[1], f[4] > 0.5
        if link and battery < self.bl and mask[EDGE]:
            return EDGE
        if link and cpu > self.cb and mask[PARTIAL]:
            return PARTIAL
        return LOCAL if mask[LOCAL] else masked_argmax([0, 1, 2], mask)


class CostThresholdPolicy(Policy):
    """Offload only when the expected (nominal-link) cost of the cheapest offloading path undercuts the local cost by a margin."""
    name = "Cost threshold"

    def __init__(self, margin: float = 0.85):
        self.margin = margin

    def act(self, f, mask, env=None):
        c = env.cost
        costs = []
        for a in range(3):
            if not mask[a]:
                costs.append(1e9); continue
            p = c.path(a, env.dev, expected=True)
            costs.append(c.cost(p["latency_ms"], p["energy_j"]))
        best_off = int(np.argmin(costs[1:])) + 1
        if mask[LOCAL] and costs[best_off] > self.margin * costs[LOCAL]:
            return LOCAL
        return best_off if mask[best_off] else masked_argmax([0, 0, 0], mask)


class GreedyPolicy(Policy):
    """Immediate expected reward with the current state and the nominal network; no forecast, no look-ahead."""
    name = "Greedy offloading"

    def __init__(self, acc: dict[int, float]):
        self.acc = acc

    def act(self, f, mask, env=None):
        c = env.cost
        r = []
        for a in range(3):
            if not mask[a]:
                r.append(-1e9); continue
            p = c.path(a, env.dev, expected=True)
            r.append(c.reward(self.acc.get(a, 0.9), p["latency_ms"], p["energy_j"]))
        return int(np.argmax(r))


class QLearningPolicy(Policy):
    name = "Q-learning (tabular)"
    kind = "learned"

    def __init__(self, alpha: float = 0.15, gamma: float = 0.95, seed: int = 0):
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.RandomState(seed)
        self.Q = np.zeros((3, 4, 3, 2, 3, 3), dtype=np.float32)
        self.eps = 1.0

    @staticmethod
    def idx(f):
        b = min(int(f[0] * 3), 2); c = min(int(f[1] * 4), 3); r = min(int(f[2] * 3), 2); l = int(f[4] > 0.5); fc = min(int(f[5] * 3), 2)
        return (b, c, r, l, fc)

    def act(self, f, mask, env=None):
        if self.rng.rand() < self.eps:
            ok = [a for a in range(3) if mask[a]]
            return int(self.rng.choice(ok)) if ok else LOCAL
        return masked_argmax(self.Q[self.idx(f)], mask)

    def update(self, f, a, r, f2, mask2):
        q = self.Q[self.idx(f)]
        q2 = self.Q[self.idx(f2)]
        best = max(q2[k] for k in range(3) if mask2[k]) if any(mask2) else 0.0
        q[a] += self.alpha * (r + self.gamma * best - q[a])


class DQNPolicy(Policy):
    """The proposed controller: a small Q-network over the device state and the busy-state forecast."""
    name = "CAQ-EdgeNet controller (DQN + busy forecast)"
    kind = "proposed"

    def __init__(self, hidden: int = 64, seed: int = 0, use_forecast: bool = True):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.target = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.target.load_state_dict(self.net.state_dict())
        self.rng = np.random.RandomState(seed)
        self.eps = 1.0
        self.use_forecast = use_forecast
        self.version = "untrained"

    def _f(self, f: np.ndarray) -> np.ndarray:
        f = np.array(f, dtype=np.float32)
        if not self.use_forecast:
            f[5] = f[1]
        return f

    def q(self, f: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.from_numpy(self._f(f))[None])[0].numpy()

    def act(self, f, mask, env=None):
        if self.rng.rand() < self.eps:
            ok = [a for a in range(3) if mask[a]]
            return int(self.rng.choice(ok)) if ok else LOCAL
        return masked_argmax(self.q(f), mask)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def save(self, path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"state_dict": self.net.state_dict(), "version": self.version, "use_forecast": self.use_forecast, **(extra or {})}, path)

    def load(self, path) -> dict[str, Any]:
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(d["state_dict"])
        self.target.load_state_dict(d["state_dict"])
        self.version = d.get("version", "loaded")
        self.use_forecast = d.get("use_forecast", True)
        self.eps = 0.0
        return d


# =============================================================================
# Training
# =============================================================================
def train_dqn(policy: DQNPolicy, make_env, episodes: int, steps: int, gamma: float, eps_start: float, eps_end: float, progress=None, lr: float = 1e-3, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(policy.net.parameters(), lr=lr)
    buf: deque = deque(maxlen=30000)
    hist = []
    step_count = 0
    t_all = time.perf_counter()
    for ep in range(1, episodes + 1):
        env = make_env(seed + ep)
        f = env.reset()
        policy.eps = eps_start + (eps_end - eps_start) * (ep - 1) / max(1, episodes - 1)
        tot = 0.0
        losses = []
        acts = [0, 0, 0]
        oracle_hits = 0
        for _ in range(steps):
            mask = env.mask()
            a = policy.act(f, mask, env)
            f2, r, info = env.step(a)
            mask2 = env.mask()
            buf.append((policy._f(f), info["action"], r, policy._f(f2), np.array(mask2, dtype=np.float32)))
            tot += r
            acts[info["action"]] += 1
            oracle_hits += int(info["action"] == info["oracle"])
            f = f2
            step_count += 1
            if len(buf) >= 512:
                batch = [buf[i] for i in rng.randint(0, len(buf), 64)]
                s = torch.from_numpy(np.stack([b[0] for b in batch])); a_t = torch.tensor([b[1] for b in batch]); r_t = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                s2 = torch.from_numpy(np.stack([b[3] for b in batch])); m2 = torch.from_numpy(np.stack([b[4] for b in batch]))
                q = policy.net(s).gather(1, a_t[:, None])[:, 0]
                with torch.no_grad():
                    q2 = policy.target(s2)
                    q2 = torch.where(m2 > 0, q2, torch.full_like(q2, -1e9)).max(1).values
                    q2 = torch.where(q2 < -1e8, torch.zeros_like(q2), q2)
                    y = r_t + gamma * q2
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
            if step_count % 250 == 0:
                policy.target.load_state_dict(policy.net.state_dict())
        row = {"episode": ep, "reward": tot / steps, "epsilon": round(policy.eps, 3), "loss": float(np.mean(losses)) if losses else None, "actions": acts, "decision_accuracy": oracle_hits / steps,
               "battery_end": env.dev.ctx.battery, "ms": round((time.perf_counter() - t_all) * 1000)}
        hist.append(row)
        if progress:
            progress(row)
    policy.eps = 0.0
    policy.version = time.strftime("p%Y%m%d-%H%M%S")
    return hist


def train_qlearning(policy: QLearningPolicy, make_env, episodes: int, steps: int, eps_start: float, eps_end: float, seed: int = 0) -> None:
    for ep in range(1, episodes + 1):
        env = make_env(seed + ep)
        f = env.reset()
        policy.eps = eps_start + (eps_end - eps_start) * (ep - 1) / max(1, episodes - 1)
        for _ in range(steps):
            mask = env.mask()
            a = policy.act(f, mask, env)
            f2, r, info = env.step(a)
            policy.update(f, info["action"], r, f2, env.mask())
            f = f2
    policy.eps = 0.0


# =============================================================================
# Evaluation (Tables 5.4 and 5.5)
# =============================================================================
def evaluate_policies(policies: list[Policy], make_env, episodes: int, steps: int, seed: int = 1000) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support
    rows = []
    oracle_reward = 0.0
    n_total = 0
    for pol in policies:
        acts, oracles, lats, ens, rews, corr, byts = [], [], [], [], [], [], []
        for ep in range(episodes):
            env = make_env(seed + ep)
            f = env.reset()
            pol.reset()
            for _ in range(steps):
                a = pol.act(f, env.mask(), env)
                f, r, info = env.step(a)
                acts.append(info["action"]); oracles.append(info["oracle"]); lats.append(info["latency_ms"]); ens.append(info["energy_j"]); rews.append(r); corr.append(info["correct"]); byts.append(info["bytes"])
                if pol is policies[0]:
                    oracle_reward += max(x["reward"] for x in info["all"] if x is not None)
                    n_total += 1
        acts, oracles = np.array(acts), np.array(oracles)
        p, r, f1, _ = precision_recall_fscore_support(oracles, acts, labels=[0, 1, 2], average="macro", zero_division=0)
        rows.append({"policy": pol.name, "kind": pol.kind, "decision_accuracy": float((acts == oracles).mean()), "precision": float(p), "recall": float(r), "f1": float(f1),
                     "latency_ms": float(np.mean(lats)), "energy_j": float(np.mean(ens)), "cost": None, "throughput": 1000.0 / float(np.mean(lats)), "reward": float(np.mean(rews)),
                     "detection_accuracy": float(np.mean(corr)), "bytes_per_request": float(np.mean(byts)), "actions": [float((acts == k).mean()) for k in range(3)], "n": int(len(acts))})
    oracle_mean = oracle_reward / max(n_total, 1)
    for row in rows:
        row["efficiency"] = float(row["reward"] / oracle_mean) if oracle_mean > 0 else None
    return {"rows": rows, "oracle_reward": oracle_mean, "episodes": episodes, "steps": steps, "actions": ACTIONS}
