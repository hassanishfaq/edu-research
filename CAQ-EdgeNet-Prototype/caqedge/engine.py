"""CAQ-EdgeNet engine -- dataset, detector training, quantisation study, cost model, offloading
controllers, live fleet simulation, single-flow inspection, telemetry and the management plane
(versions of detector / quantised export / policy), Sections 5.2 - 5.4.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, deque
from typing import Any

import numpy as np
import torch

from . import controller as C
from . import data as D
from . import model as M
from . import simulator as S
from .config import ROOT, Config

try:
    import psutil
except ImportError:  # optional
    psutil = None

PRECISIONS = ["fp32", "fp16", "int8", "int8_hybrid"]


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._event_id = 0
        self.started_at = time.time()
        self.data: dict[str, Any] | None = None
        self.model: M.CAQEdgeNet | None = None
        self.trained = False
        self.detector_version = "untrained"
        self.export_version = "-"
        self.history: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.deploy: dict[str, Any] = {}
        self.runtimes: dict[str, M.NumpyRuntime] = {}
        self.cost: S.CostModel | None = None
        self.predictors: dict[int, Any] = {}
        self.busy_fc = None
        self.training = False
        self.progress: dict[str, Any] = {}
        self.ablation: dict[str, Any] | None = None
        self.ablation_running = False
        # controller
        self.policy: C.DQNPolicy | None = None
        self.policy_version = "untrained"
        self.ctl_history: list[dict[str, Any]] = []
        self.ctl_eval: dict[str, Any] | None = None
        self.ctl_training = False
        self.ctl_progress: dict[str, Any] = {}
        # fleet
        self.fleet_running = False
        self._fleet_thread: threading.Thread | None = None
        self.fleet: dict[str, Any] = {"devices": [], "recent": [], "stats": {}}
        self.telemetry: deque[dict[str, Any]] = deque(maxlen=3000)
        self.telemetry_file = cfg.resolve(cfg.get("simulation.telemetry_file", "data/telemetry.jsonl"))
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if self._proc:
            self._proc.cpu_percent(None)
        self._boot()

    # ------------------------------------------------------------------ events
    def log(self, msg: str, level: str = "info", **data: Any) -> None:
        with self.lock:
            self._event_id += 1
            self.events.append({"id": self._event_id, "t": time.time(), "level": level, "msg": msg, **({"data": data} if data else {})})

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if e["id"] > since]

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        self.log("CAQ-EdgeNet engine starting", "sys")
        self.data = D.load_dataset(self.cfg.cache_path)
        if self.data is None:
            self.data = D.build_dataset(self.cfg, log=lambda m: self.log(m, "sys"))
            D.save_dataset(self.data, self.cfg.cache_path)
        else:
            self.log(f"dataset cache loaded: {len(self.data['y'])} flows ({self.data['source']}), split {[int((self.data['part'] == k).sum()) for k in range(3)]}", "sys")
        self.model = self._make_model()
        ck = self.cfg.checkpoint
        if ck.exists():
            try:
                d = torch.load(ck, map_location="cpu", weights_only=False)
                self.model = M.CAQEdgeNet(**d["cfg"])
                self.model.load_state_dict(d["state_dict"])
                self.trained = True
                self.detector_version = d.get("version", "loaded")
                self.history = d.get("history", [])
                self.log(f"detector checkpoint loaded: {ck.name} ({self.model.n_params():,} params, version {self.detector_version})", "sys")
                self._after_train()
            except Exception as e:
                self.log(f"checkpoint could not be loaded ({e}); train the detector from the dashboard", "warn")
        else:
            self.log("no detector checkpoint (models/caq_edgenet.pt); train it from the dashboard", "warn")
        self.policy = C.DQNPolicy(seed=int(self.cfg.get("data.seed", 42)))
        pp = self.cfg.policy_path
        if pp.exists():
            try:
                d = self.policy.load(pp)
                self.policy_version = self.policy.version
                self.ctl_history = d.get("history", [])
                self.ctl_eval = d.get("evaluation")
                self.log(f"controller policy loaded: {pp.name} (version {self.policy_version}{', with its evaluation' if self.ctl_eval else ''})", "sys")
            except Exception as e:
                self.log(f"policy could not be loaded ({e})", "warn")
        self.log(f"engine ready ({len(D.CLASSES)} classes, {len(D.FLOW_FEATURES)} flow + {len(D.CONTEXT_FEATURES)} context features)", "sys")

    def _make_model(self, **over: Any) -> M.CAQEdgeNet:
        mc = self.cfg.get("model", {}) or {}
        kw = dict(d_flow=len(D.FLOW_FEATURES), d_ctx=len(D.CONTEXT_FEATURES), d_state=3, h_flow=int(mc.get("h_flow", 64)), h_ctx=int(mc.get("h_ctx", 16)), d_z=int(mc.get("d_z", 64)),
                  h_cls=int(mc.get("h_cls", 32)), n_classes=len(D.CLASSES), attention=bool(mc.get("conditional_attention", True)), qae=bool(mc.get("qae", True)), bits=int(mc.get("qae_bits", 8)),
                  busy_head=bool(mc.get("busy_head", True)), use_ctx=True)
        kw.update(over)
        return M.CAQEdgeNet(**kw)

    # ------------------------------------------------------------------ detector training
    def train_detector(self, epochs: int | None = None) -> None:
        if self.training:
            return
        threading.Thread(target=self._train_worker, args=(epochs,), daemon=True).start()

    def _train_worker(self, epochs: int | None) -> None:
        self.training = True
        try:
            tc = self.cfg.get("training", {}) or {}
            E = int(epochs or tc.get("epochs", 12))
            self.model = self._make_model()
            self.log(f"training the detector: {E} epochs, batch {tc.get('batch_size', 256)}, lr {tc.get('lr', 0.002)}, lambda_busy {self.cfg.get('model.lambda_busy', 0.5)} on {int((self.data['part'] == 0).sum())} flows", "fed")
            self.history = []

            def prog(row):
                self.history.append(row)
                self.progress = {"epoch": row["epoch"], "epochs": E, "phase": "training"}
                self.log(f"epoch {row['epoch']}/{E}: loss {row['loss']:.3f} · val acc {row['val_accuracy'] * 100:.2f}% · val F1 {row['val_f1'] * 100:.2f}% · busy MAE {row['val_busy_mae']:.3f} · {row['ms']} ms", "fed")

            M.train_model(self.model, self.data, E, int(tc.get("batch_size", 256)), float(tc.get("lr", 0.002)), float(tc.get("weight_decay", 1e-4)), float(self.cfg.get("model.lambda_busy", 0.5)),
                          seed=int(self.cfg.get("data.seed", 42)), progress=prog)
            self.trained = True
            self.detector_version = time.strftime("d%Y%m%d-%H%M%S")
            self.progress = {"phase": "evaluation, quantisation study and deployment measurements"}
            self._after_train()
            self.cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.model.state_dict(), "cfg": self.model.cfg, "version": self.detector_version, "history": self.history, "mu": self.data["mu"], "sd": self.data["sd"]}, self.cfg.checkpoint)
            m = self.metrics["test"]
            self.log(f"detector ready: test accuracy {m['accuracy'] * 100:.2f}%, macro-F1 {m['f1_macro'] * 100:.2f}%, AUC {m['auc']:.3f}; busy-state MAE {m['busy_mae']:.3f}; version {self.detector_version}", "fed")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"training failed: {e}", "error")
        finally:
            self.training = False
            self.progress = {}

    # ------------------------------------------------------------------ after training: metrics, quantisation study, cost model
    def _test_idx(self) -> np.ndarray:
        return np.where(self.data["part"] == 2)[0]

    def _states_for(self, idx: np.ndarray) -> np.ndarray:
        return M.sample_states(len(idx), np.random.RandomState(0), self.data["X_ctx"][idx])

    def _after_train(self) -> None:
        d = self.data
        te = self._test_idx()
        Xf, Xc, y = d["X_flow"][te], d["X_ctx"][te], d["y"][te]
        st = self._states_for(te)
        rng = np.random.RandomState(0)
        self.metrics = {"test": M.evaluate(self.model, d, te, rng), "classes": D.CLASSES, "n_params": self.model.n_params(), "layers": M.layer_table(self.model)}
        self.metrics["busy_baselines"] = self._busy_baselines()
        # ---- quantisation study (Table 5.7) with the sensitivity-aware hybrid scheme (Section 5.2.3)
        qc = self.cfg.get("quantization", {}) or {}
        reps = int(qc.get("latency_repeats", 400))
        base_rt = M.NumpyRuntime(self.model, {})
        base_acc = base_rt.accuracy(Xf, Xc, st, y)
        sens = []
        for L in M.LAYER_ORDER:
            a8 = M.NumpyRuntime(self.model, {L: 8}).accuracy(Xf, Xc, st, y)
            a4 = M.NumpyRuntime(self.model, {L: 4}).accuracy(Xf, Xc, st, y)
            sens.append({"layer": L, "role": M.LAYER_ROLE[L], "params": int(getattr(self.model, L).weight.numel() + getattr(self.model, L).bias.numel()), "drop8_pts": (base_acc - a8) * 100, "drop4_pts": (base_acc - a4) * 100})
        thr = float(qc.get("sensitivity_threshold_pts", 0.3))
        protected = set(qc.get("protected_layers", ["attn", "cls_out"]))
        hybrid = {}
        for s in sens:
            L = s["layer"]
            hybrid[L] = 16 if (L in protected or s["drop8_pts"] > thr) else (4 if s["drop4_pts"] <= thr / 3 and L.startswith("busy") else 8)
            s["hybrid_bits"] = hybrid[L]
            s["reason"] = "protected (thesis: attention / final classifier)" if L in protected else ("sensitive at 8 bits" if s["drop8_pts"] > thr else ("tolerant: 4 bits" if hybrid[L] == 4 else "tolerant: 8 bits"))
        maps = {"fp32": {}, "fp16": {k: 16 for k in M.LAYER_ORDER}, "int8": {k: 8 for k in M.LAYER_ORDER}, "int8_hybrid": hybrid}
        self.runtimes = {k: M.NumpyRuntime(self.model, v) for k, v in maps.items()}
        rows = []
        size32 = self.runtimes["fp32"].size_bytes()
        lat32 = self.runtimes["fp32"].latency_ms(Xf, Xc, st, reps)
        speed = {"fp32": 1.0, "fp16": 1.35, "int8": 2.33, "int8_hybrid": 2.2}     # integer-kernel speed-ups of a mobile runtime (Table 5.7); numpy has no integer kernels
        dev_ref = float(self.cfg.get("device_profile.reference_fp32_ms", 12.6))
        for k in PRECISIONS:
            rt = self.runtimes[k]
            lat = rt.latency_ms(Xf, Xc, st, reps)
            acc = rt.accuracy(Xf, Xc, st, y)
            rows.append({"precision": k, "label": {"fp32": "FP32", "fp16": "FP16", "int8": "INT8 uniform", "int8_hybrid": "INT8 hybrid (QAE, sensitivity-aware)"}[k], "size_bytes": rt.size_bytes(), "device_bytes": rt.size_bytes("device"),
                         "size_reduction": 1 - rt.size_bytes() / size32, "measured_ms": lat, "measured_encoder_ms": rt.latency_ms(Xf, Xc, st, reps, "device"), "measured_heads_ms": rt.latency_ms(Xf, Xc, st, reps, "edge_heads"),
                         "device_ms": dev_ref / speed[k], "latency_reduction": 1 - 1 / speed[k], "accuracy": acc, "bits": rt.describe(), "speedup_assumed": speed[k]})
        self.deploy = {"rows": rows, "sensitivity": sens, "base_accuracy": base_acc, "threshold_pts": thr, "protected": sorted(protected), "torch_bytes_fp32": M.state_dict_bytes(self.model),
                       "torch_bytes_fp16": M.state_dict_bytes(self.model, torch.float16), "measured_fp32_ms": lat32, "note": "sizes and accuracies are executed at the stated precision (Eq. 5.4 per tensor, integer accumulation); "
                       "device latencies anchor the measured FP32 time to device_profile.reference_fp32_ms and apply the integer-kernel speed-ups of a mobile runtime, which numpy does not have"}
        self.export_version = self.detector_version.replace("d", "q", 1)
        # ---- cost model and per-path predictors (local: INT8 hybrid on device; partial: INT8 encoder + FP32 heads at the edge; edge: FP32)
        timings = {"fp32_full_ms": lat32, "fp32_encoder_ms": rows[0]["measured_encoder_ms"], "fp32_heads_ms": rows[0]["measured_heads_ms"]}
        d_z, bits = self.model.cfg["d_z"], self.model.bits if self.model.qae else 32
        dev_cfg = dict(self.cfg.get("device_profile", {}) or {})
        self.cost = S.CostModel(dev_cfg, self.cfg.get("network", {}) or {}, self.cfg.get("controller", {}) or {}, timings, {"z_bytes": int(np.ceil(d_z * bits / 8)) + 4, "feature_bytes": (len(D.FLOW_FEATURES) + len(D.CONTEXT_FEATURES)) * 4}, speed)
        hyb, f32 = self.runtimes["int8_hybrid"], self.runtimes["fp32"]
        self.predictors = {S.LOCAL: lambda xf, xc, s: int(hyb.forward(xf, xc, s)[0].argmax(1)[0]),
                           S.PARTIAL: lambda xf, xc, s: int(f32.heads(hyb.encode(xf, xc, s)[0])[0].argmax(1)[0]),
                           S.EDGE: lambda xf, xc, s: int(f32.forward(xf, xc, s)[0].argmax(1)[0])}
        self.busy_fc = lambda xf, xc, s: float(hyb.forward(xf, xc, s)[1][0])
        self.path_accuracy = {S.LOCAL: rows[3]["accuracy"], S.PARTIAL: float((np.concatenate([f32.heads(hyb.encode(Xf[i:i + 4096], Xc[i:i + 4096], st[i:i + 4096])[0])[0].argmax(1) for i in range(0, len(Xf), 4096)]) == y).mean()), S.EDGE: rows[0]["accuracy"]}
        self.log(f"quantisation study: FP32 {size32 / 1e3:.1f} KB / {lat32:.3f} ms here -> INT8 hybrid {rows[3]['size_bytes'] / 1e3:.1f} KB ({rows[3]['size_reduction'] * 100:.0f}% smaller), accuracy {base_acc * 100:.2f}% -> {rows[3]['accuracy'] * 100:.2f}%; "
                 f"device estimate {self.cost.device_ms('full', 'fp32'):.1f} ms FP32 / {self.cost.device_ms('full', 'int8'):.1f} ms INT8", "fed")

    def _busy_baselines(self) -> list[dict[str, Any]]:
        """Table 5.3-style comparison of the busy-state head with stand-alone forecasters on the same context rows."""
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        d = self.data
        tr, te = np.where(d["part"] == 0)[0], self._test_idx()
        Xtr, ytr, Xte, yte = d["X_ctx"][tr], d["y_busy"][tr], d["X_ctx"][te], d["y_busy"][te]

        def met(pred):
            err = pred - yte
            return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())), "mape": float((np.abs(err) / np.maximum(np.abs(yte), 0.05)).mean() * 100),
                    "r2": float(1 - (err ** 2).sum() / max(((yte - yte.mean()) ** 2).sum(), 1e-9)), "evs": float(1 - err.var() / max(yte.var(), 1e-9))}
        rows = [{"method": "Last value (current CPU)", **met(Xte[:, 1])}, {"method": "Linear regression", **met(LinearRegression().fit(Xtr, ytr).predict(Xte))}]
        mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=0).fit(Xtr[:20000], ytr[:20000])
        rows.append({"method": "MLP forecasting (stand-alone)", **met(mlp.predict(Xte))})
        t = self.metrics["test"]
        rows.append({"method": "CAQ-EdgeNet busy-state head (shared)", "mae": t["busy_mae"], "rmse": t["busy_rmse"], "mape": t["busy_mape"], "r2": t["busy_r2"], "evs": t["busy_evs"], "proposed": True})
        return rows

    # ------------------------------------------------------------------ ablation (Table 5.9)
    def run_ablation(self) -> None:
        if self.ablation_running or self.training:
            return
        threading.Thread(target=self._ablation_worker, daemon=True).start()

    def _ablation_worker(self) -> None:
        self.ablation_running = True
        try:
            tc = self.cfg.get("training", {}) or {}
            E = int(tc.get("ablation_epochs", 5))
            variants = [("Full CAQ-EdgeNet", {}), ("Without FusionCAB (no conditional attention)", {"attention": False}), ("Without QAE (no quantisation-aware compression)", {"qae": False}),
                        ("Without busy head", {"busy_head": False}), ("Without shared learning (detector only, forecast = current CPU)", {"busy_head": False, "_noshare": True}), ("Security only (no context modality)", {"use_ctx": False, "busy_head": False})]
            rows = []
            te = self._test_idx()
            Xf, Xc, y = self.data["X_flow"][te], self.data["X_ctx"][te], self.data["y"][te]
            st = self._states_for(te)
            ctl = self.cfg.get("controller", {}) or {}
            self.log(f"ablation study: {len(variants)} variants x {E} epochs", "fed")
            for name, over in variants:
                self.ablation = {"status": "running", "current": name, "rows": rows}
                kw = {k: v for k, v in over.items() if not k.startswith("_")}
                mdl = self._make_model(**kw)
                M.train_model(mdl, self.data, E, int(tc.get("batch_size", 256)), float(tc.get("lr", 0.002)), float(tc.get("weight_decay", 1e-4)), float(self.cfg.get("model.lambda_busy", 0.5)), seed=int(self.cfg.get("data.seed", 42)))
                ev = M.evaluate(mdl, self.data, te)
                hyb_map = self.runtimes["int8_hybrid"].describe() if self.runtimes else {k: 8 for k in M.LAYER_ORDER}
                hyb, f32 = M.NumpyRuntime(mdl, hyb_map), M.NumpyRuntime(mdl, {})
                acc8 = hyb.accuracy(Xf, Xc, st, y)
                # offloading accuracy: the released controller re-evaluated with this variant's forecast and detection outcomes
                off = None
                if self.policy is not None and self.policy_version != "untrained" and self.cost is not None:
                    preds = {S.LOCAL: lambda xf, xc, s: int(hyb.forward(xf, xc, s)[0].argmax(1)[0]), S.PARTIAL: lambda xf, xc, s: int(f32.heads(hyb.encode(xf, xc, s)[0])[0].argmax(1)[0]), S.EDGE: lambda xf, xc, s: int(f32.forward(xf, xc, s)[0].argmax(1)[0])}
                    fc = None if (over.get("busy_head") is False or over.get("_noshare")) else (lambda xf, xc, s: float(hyb.forward(xf, xc, s)[1][0]))
                    mk = self._env_factory(preds, fc)
                    evp = C.evaluate_policies([self.policy], mk, int(ctl.get("eval_episodes", 12)) // 2 or 1, int(ctl.get("steps_per_episode", 120)), seed=7000)
                    off = evp["rows"][0]
                rows.append({"variant": name, "f1": ev["f1_macro"], "accuracy": ev["accuracy"], "int8_accuracy": acc8, "busy_mae": ev["busy_mae"] if mdl.busy_head else None, "params": mdl.n_params(),
                             "offload_accuracy": off["decision_accuracy"] if off else None, "cost": (self.cost.cost(off["latency_ms"], off["energy_j"]) if off else None), "latency_ms": off["latency_ms"] if off else None, "energy_j": off["energy_j"] if off else None})
                self.log(f"ablation {name}: F1 {ev['f1_macro'] * 100:.1f}%" + (f", offloading accuracy {off['decision_accuracy'] * 100:.1f}%, cost {rows[-1]['cost']:.3f}" if off else ""), "fed")
            self.ablation = {"status": "ok", "rows": rows, "epochs": E}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"ablation failed: {e}", "error")
            self.ablation = {"status": "error", "error": str(e)}
        finally:
            self.ablation_running = False

    # ------------------------------------------------------------------ environments
    def _env_factory(self, predictors: dict[int, Any] | None = None, busy_fc=None, steps: int | None = None):
        d = self.data
        te = self._test_idx()
        Xf, Xc, y = d["X_flow"][te], d["X_ctx"][te], d["y"][te]
        preds = predictors or self.predictors
        net = self.cfg.get("network", {}) or {}
        bw = float(self.cfg.get("device_profile.battery_wh", 7.4))
        n_steps = int(steps or self.cfg.get("controller.steps_per_episode", 120))
        fc = busy_fc if busy_fc is not None else self.busy_fc
        wrapped = {k: (lambda fn: (lambda xf, xc, s, label: float(fn(xf, xc, s) == label)))(v) for k, v in preds.items()}
        return lambda seed, dev_id=0: S.Env(self.cost, wrapped, Xf, Xc, y, net, bw, busy_forecaster=fc, seed=seed, dev_id=dev_id, steps=n_steps)

    # ------------------------------------------------------------------ controller training + policy comparison
    def train_controller(self, episodes: int | None = None) -> None:
        if self.ctl_training or not self.trained:
            return
        threading.Thread(target=self._ctl_worker, args=(episodes,), daemon=True).start()

    def _ctl_worker(self, episodes: int | None) -> None:
        self.ctl_training = True
        try:
            ctl = self.cfg.get("controller", {}) or {}
            E = int(episodes or ctl.get("episodes", 160))
            steps = int(ctl.get("steps_per_episode", 120))
            seed = int(self.cfg.get("data.seed", 42))
            mk = self._env_factory()
            self.policy = C.DQNPolicy(seed=seed)
            self.ctl_history = []
            self.log(f"training the offloading controller: DQN, {E} episodes x {steps} steps, gamma {ctl.get('gamma', 0.95)}, reward weights {self.cost.lam}", "fed")

            def prog(row):
                self.ctl_history.append(row)
                self.ctl_progress = {"episode": row["episode"], "episodes": E, "phase": "DQN training"}
                if row["episode"] % 10 == 0 or row["episode"] == E:
                    self.log(f"episode {row['episode']}/{E}: mean reward {row['reward']:.3f} · decisions = oracle {row['decision_accuracy'] * 100:.0f}% · actions {row['actions']} · eps {row['epsilon']}", "fed")

            C.train_dqn(self.policy, mk, E, steps, float(ctl.get("gamma", 0.95)), float(ctl.get("epsilon_start", 1.0)), float(ctl.get("epsilon_end", 0.05)), progress=prog, seed=seed)
            self.policy_version = self.policy.version
            self.cfg.policy_path.parent.mkdir(parents=True, exist_ok=True)
            self.policy.save(self.cfg.policy_path)
            self.ctl_progress = {"phase": "training the tabular Q-learning and no-forecast baselines"}
            ql = C.QLearningPolicy(seed=seed)
            C.train_qlearning(ql, mk, E, steps, float(ctl.get("epsilon_start", 1.0)), float(ctl.get("epsilon_end", 0.05)), seed=seed)
            dqn_nf = C.DQNPolicy(seed=seed, use_forecast=False)
            dqn_nf.name = "DQN without busy forecast"
            dqn_nf.kind = "baseline"
            C.train_dqn(dqn_nf, mk, E, steps, float(ctl.get("gamma", 0.95)), float(ctl.get("epsilon_start", 1.0)), float(ctl.get("epsilon_end", 0.05)), seed=seed)
            self.ctl_progress = {"phase": "policy comparison on held-out episodes"}
            pols = [C.RandomPolicy(seed), C.RulePolicy(), C.CostThresholdPolicy(), C.GreedyPolicy(self.path_accuracy), ql, dqn_nf, self.policy]
            self.ctl_eval = C.evaluate_policies(pols, mk, int(ctl.get("eval_episodes", 12)), steps, seed=5000)
            self.ctl_eval["policy_version"] = self.policy_version
            self.policy.save(self.cfg.policy_path, {"history": self.ctl_history, "evaluation": self.ctl_eval})
            best = self.ctl_eval["rows"][-1]
            self.log(f"controller ready: decision accuracy {best['decision_accuracy'] * 100:.1f}% (rule-based {self.ctl_eval['rows'][1]['decision_accuracy'] * 100:.1f}%), "
                     f"latency {best['latency_ms']:.0f} ms, energy {best['energy_j']:.2f} J, efficiency {best['efficiency']:.2f}; version {self.policy_version}", "fed")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"controller training failed: {e}", "error")
        finally:
            self.ctl_training = False
            self.ctl_progress = {}

    # ------------------------------------------------------------------ live fleet
    def set_fleet(self, enabled: bool) -> None:
        if enabled and not self.trained:
            self.log("train the detector before starting the fleet", "warn")
            return
        self.fleet_running = enabled
        if enabled and (self._fleet_thread is None or not self._fleet_thread.is_alive()):
            self._fleet_thread = threading.Thread(target=self._fleet_loop, daemon=True)
            self._fleet_thread.start()
            self.log(f"fleet simulation started: {int(self.cfg.get('simulation.n_devices', 6))} devices, proposed controller acting, rule-based policy in shadow mode", "sys")
        elif not enabled:
            self.log("fleet simulation stopped", "sys")

    def _fleet_loop(self) -> None:
        n = int(self.cfg.get("simulation.n_devices", 6))
        tick = float(self.cfg.get("simulation.tick_seconds", 0.7))
        hyst = int(self.cfg.get("controller.hysteresis", 2))
        mk = self._env_factory(steps=10 ** 9)
        envs = [mk(1000 + i, dev_id=i) for i in range(n)]
        obs = [e.reset() for e in envs]
        shadow = C.RulePolicy()
        last_action = [S.LOCAL] * n
        pending: list[tuple[int, int]] = [(S.LOCAL, 0)] * n
        win = {"actions": deque(maxlen=300), "fc_err": deque(maxlen=300), "benign_conf": deque(maxlen=300), "reward": deque(maxlen=300), "shadow_reward": deque(maxlen=300), "lat": deque(maxlen=300), "energy": deque(maxlen=300), "shadow_lat": deque(maxlen=300), "shadow_energy": deque(maxlen=300)}
        totals = {"requests": 0, "bytes": 0, "correct": 0, "fallbacks": 0}
        recent: deque[dict[str, Any]] = deque(maxlen=40)
        hyb = self.runtimes["int8_hybrid"]
        while self.fleet_running:
            t0 = time.perf_counter()
            devs = []
            for i, env in enumerate(envs):
                mask = env.mask()
                use_policy = self.policy is not None and self.policy_version != "untrained"
                raw = self.policy.act(obs[i], mask, env) if use_policy else shadow.act(obs[i], mask, env)
                # hysteresis: a switch needs the preference to persist `hyst` consecutive decisions
                if raw != last_action[i]:
                    cand, count = pending[i]
                    count = count + 1 if cand == raw else 1
                    pending[i] = (raw, count)
                    action = raw if count >= hyst else last_action[i]
                else:
                    pending[i] = (raw, 0)
                    action = raw
                if not mask[action]:
                    action = next((a for a in (S.LOCAL, S.EDGE, S.PARTIAL) if mask[a]), S.LOCAL)
                    totals["fallbacks"] += 1
                sh = shadow.act(obs[i], mask, env)
                P = hyb.forward(env._xf, env._xc, env._st)[0][0]
                f, r, info = env.step(action)
                last_action[i] = info["action"]
                obs[i] = f
                sh_out = info["all"][sh] if info["all"][sh] is not None else info["all"][info["action"]]
                win["actions"].append(info["action"]); win["fc_err"].append(abs(info["busy_forecast"] - info["busy_actual_next"])); win["reward"].append(r); win["shadow_reward"].append(sh_out["reward"])
                win["lat"].append(info["latency_ms"]); win["energy"].append(info["energy_j"]); win["shadow_lat"].append(sh_out["latency_ms"]); win["shadow_energy"].append(sh_out["energy_j"])
                if info["label"] == 0:
                    win["benign_conf"].append(float(P.max()))
                totals["requests"] += 1; totals["bytes"] += info["bytes"]; totals["correct"] += int(info["correct"])
                rec = {"t": time.time(), "device": i, "action": S.ACTIONS[info["action"]], "shadow": S.ACTIONS[sh], "oracle": S.ACTIONS[info["oracle"]], "latency_ms": round(info["latency_ms"], 1), "energy_j": round(info["energy_j"], 4),
                       "bytes": info["bytes"], "reward": round(r, 4), "correct": bool(info["correct"]), "label": D.CLASSES[info["label"]], "pred_conf": round(float(P.max()), 3), "busy_forecast": round(info["busy_forecast"], 3),
                       "state": {k: info["state"][k] for k in ("battery", "cpu", "rtt_ms", "bandwidth_mbps", "link_up")}, "versions": {"detector": self.detector_version, "export": self.export_version, "policy": self.policy_version}}
                self.telemetry.appendleft(rec)
                recent.appendleft(rec)
                try:
                    with open(self.telemetry_file, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                devs.append({**env.dev.snapshot(), "last_action": S.ACTIONS[info["action"]], "shadow_action": S.ACTIONS[sh], "last_latency_ms": round(info["latency_ms"], 1), "last_energy_j": round(info["energy_j"], 4), "busy_forecast": round(info["busy_forecast"], 3),
                             "label": D.CLASSES[info["label"]], "correct": bool(info["correct"]), "bytes": info["bytes"], "q": [float(x) for x in self.policy.q(obs[i])] if use_policy else None})
            acts = np.array(win["actions"]) if win["actions"] else np.zeros(0)
            self.fleet = {"devices": devs, "recent": list(recent)[:16], "stats": {
                "requests": totals["requests"], "bytes": totals["bytes"], "detection_accuracy": totals["correct"] / max(totals["requests"], 1), "fallbacks": totals["fallbacks"],
                "action_share": [float((acts == k).mean()) if len(acts) else 0.0 for k in range(3)], "busy_mae": float(np.mean(win["fc_err"])) if win["fc_err"] else None,
                "reward": float(np.mean(win["reward"])) if win["reward"] else None, "shadow_reward": float(np.mean(win["shadow_reward"])) if win["shadow_reward"] else None,
                "latency_ms": float(np.mean(win["lat"])) if win["lat"] else None, "shadow_latency_ms": float(np.mean(win["shadow_lat"])) if win["shadow_lat"] else None,
                "energy_j": float(np.mean(win["energy"])) if win["energy"] else None, "shadow_energy_j": float(np.mean(win["shadow_energy"])) if win["shadow_energy"] else None,
                "benign_conf_hist": np.histogram(list(win["benign_conf"]), np.linspace(0, 1, 11))[0].tolist() if win["benign_conf"] else None, "window": len(acts), "policy_active": use_policy}}
            time.sleep(max(0.05, tick - (time.perf_counter() - t0)))

    # ------------------------------------------------------------------ inspect one flow under a chosen device state
    def samples(self) -> dict[str, Any]:
        te = self._test_idx()
        y = self.data["y"][te]
        rng = np.random.RandomState(3)
        out = []
        for ci, c in enumerate(D.CLASSES):
            idx = np.where(y == ci)[0]
            for j in rng.choice(idx, size=min(4, len(idx)), replace=False):
                out.append({"index": int(j), "class": c})
        return {"samples": out, "classes": D.CLASSES, "n_test": int(len(te))}

    def inspect(self, index: int | None, values: list[float] | None, state: dict[str, Any]) -> dict[str, Any]:
        if not self.trained:
            return {"status": "not_ready", "message": "Train the detector first."}
        te = self._test_idx()
        if values is not None:
            raw = np.array(values, dtype=np.float32)[None]
            xf = D.standardise_flow(raw, self.data["mu"], self.data["sd"])
            label = None
            idx = None
        else:
            idx = int(index or 0) % len(te)
            xf = self.data["X_flow"][te[idx]][None]
            raw = self.data["X_flow_raw"][te[idx]][None]
            label = D.CLASSES[int(self.data["y"][te[idx]])]
        # a device in the requested state
        dev = S.DeviceSim(99, self.cfg.get("network", {}) or {}, 99, float(self.cfg.get("device_profile.battery_wh", 7.4)))
        dev.ctx.battery = float(state.get("battery", dev.ctx.battery)); dev.ctx.cpu = float(state.get("cpu", dev.ctx.cpu)); dev.net.rtt = float(state.get("rtt_ms", dev.net.rtt))
        dev.net.bw = float(state.get("bandwidth_mbps", dev.net.bw)); dev.net.link_up = bool(state.get("link_up", True)); dev.ctx.charging = bool(state.get("charging", False))
        if "edge_load" in state:
            dev.net.edge_load = float(state["edge_load"])
        xc = dev.context()[None]
        st = dev.state(self.cost)[None]
        t0 = time.perf_counter()
        hyb, f32 = self.runtimes["int8_hybrid"], self.runtimes["fp32"]
        zq, gate, z = hyb.encode(xf, xc, st)
        P_dev, busy_dev = hyb.heads(zq)
        t1 = time.perf_counter()
        P_edge, busy_edge = f32.forward(xf, xc, st)[:2]
        P_partial = f32.heads(zq)[0]
        t2 = time.perf_counter()
        feats = dev.policy_features(float(busy_dev[0]), self.cost)
        mask = self.cost.feasible(dev)
        paths = []
        for a in range(3):
            p = self.cost.path(a, dev)
            Pa = [P_dev, P_partial, P_edge][a][0]
            pred = D.CLASSES[int(Pa.argmax())]
            corr = None if label is None else float(pred == label)
            paths.append({"action": S.ACTIONS[a], "feasible": mask[a], "latency_ms": p["latency_ms"], "energy_j": p["energy_j"], "bytes": p["bytes"], "breakdown": p["breakdown"], "prediction": pred, "confidence": float(Pa.max()),
                          "reward": (self.cost.reward(corr if corr is not None else float(Pa.max()), p["latency_ms"], p["energy_j"])), "cost": self.cost.cost(p["latency_ms"], p["energy_j"]),
                          "precision": ["INT8 hybrid on device", "INT8 encoder on device + FP32 heads at the edge", "FP32 at the edge"][a]})
        q = [float(x) for x in self.policy.q(feats)] if (self.policy is not None and self.policy_version != "untrained") else None
        chosen = C.masked_argmax(q, mask) if q else C.RulePolicy().act(feats, mask)
        oracle = int(np.argmax([p["reward"] if p["feasible"] else -1e9 for p in paths]))
        bits = self.model.bits if self.model.qae else 32
        s_q = float((z.max() - z.min()) / (2 ** bits - 1)) if self.model.qae else None
        return {"status": "ok", "index": idx, "label": label, "raw_features": {f: float(v) for f, v in zip(D.FLOW_FEATURES, raw[0])}, "context": {f: float(v) for f, v in zip(D.CONTEXT_FEATURES, xc[0])},
                "state": {"battery": float(st[0][0]), "cpu": float(st[0][1]), "latency_norm": float(st[0][2]), "observed_latency_ms": dev.observed_latency_ms(self.cost), "rtt_ms": dev.net.rtt, "bandwidth_mbps": dev.net.bw, "link_up": dev.net.link_up, "edge_load": dev.net.edge_load},
                "encoder": {"h_flow": self.model.cfg["h_flow"], "h_ctx": self.model.cfg["h_ctx"], "d_z": self.model.cfg["d_z"], "z": z[0].tolist(), "gate": gate[0].tolist(), "z_hat": zq[0].tolist(), "bits": bits, "scale": s_q, "z_bytes": self.cost.sizes["z_bytes"], "feature_bytes": self.cost.sizes["feature_bytes"]},
                "verdict": {"class": D.CLASSES[int(P_dev[0].argmax())], "confidence": float(P_dev[0].max()), "probs": {c: float(P_dev[0][i]) for i, c in enumerate(D.CLASSES)}, "edge_class": D.CLASSES[int(P_edge[0].argmax())], "edge_confidence": float(P_edge[0].max()),
                            "busy_forecast": float(busy_dev[0]), "busy_forecast_edge": float(busy_edge[0]), "malicious": D.CLASSES[int(P_dev[0].argmax())] != "Benign"},
                "paths": paths, "controller": {"features": {k: float(v) for k, v in zip(S.POLICY_FEATURES, feats)}, "q": q, "chosen": S.ACTIONS[chosen], "oracle": S.ACTIONS[oracle], "mask": mask, "source": "DQN policy" if q else "rule-based fallback (policy untrained)", "policy_version": self.policy_version},
                "timings_ms": {"device_int8": round((t1 - t0) * 1000, 3), "edge_fp32": round((t2 - t1) * 1000, 3)}, "versions": {"detector": self.detector_version, "export": self.export_version, "policy": self.policy_version}}

    # ------------------------------------------------------------------ status
    def resources(self) -> dict[str, Any]:
        out: dict[str, Any] = {"uptime_s": round(time.time() - self.started_at)}
        if self._proc:
            mi = self._proc.memory_info()
            out.update({"cpu_percent": self._proc.cpu_percent(None), "rss_mb": round(mi.rss / 1e6, 1), "threads": self._proc.num_threads(), "system_cpu_percent": psutil.cpu_percent(None), "system_ram_percent": psutil.virtual_memory().percent, "cpu_count": psutil.cpu_count()})
        return out

    def dataset_summary(self) -> dict[str, Any]:
        d = self.data
        y = d["y"]
        cnt = np.bincount(y, minlength=len(D.CLASSES))
        return {"source": d["source"], "info": d.get("info", {}), "n": int(len(y)), "classes": [{"name": c, "group": D.CLASS_GROUP[c], "n": int(cnt[i]), "share": float(cnt[i] / len(y)), "ratio": float(cnt.max() / max(cnt[i], 1))} for i, c in enumerate(D.CLASSES)],
                "split": [int((d["part"] == k).sum()) for k in range(3)], "flow_features": D.FLOW_FEATURES, "context_features": D.CONTEXT_FEATURES, "imbalance": float(cnt.max() / max(cnt.min(), 1))}

    def status(self) -> dict[str, Any]:
        import sklearn
        versions = {"python": __import__("platform").python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()}
        return {
            "trained": self.trained, "training": self.training, "progress": self.progress, "ablation_running": self.ablation_running, "ctl_training": self.ctl_training, "ctl_progress": self.ctl_progress,
            "fleet_running": self.fleet_running, "policy_trained": self.policy is not None and self.policy_version != "untrained",
            "versions_model": {"detector": self.detector_version, "export": self.export_version, "policy": self.policy_version},
            "model": {"n_params": self.model.n_params() if self.model else None, "cfg": self.model.cfg if self.model else None, "layers": M.layer_table(self.model) if self.model else []},
            "dataset": {"source": self.data["source"], "n": int(len(self.data["y"])), "split": [int((self.data["part"] == k).sum()) for k in range(3)]} if self.data else None,
            "metrics_summary": {"accuracy": self.metrics["test"]["accuracy"], "f1_macro": self.metrics["test"]["f1_macro"], "busy_mae": self.metrics["test"]["busy_mae"]} if self.metrics else None,
            "deploy_summary": [{k: r[k] for k in ("precision", "label", "size_bytes", "device_ms", "accuracy")} for r in self.deploy.get("rows", [])] if self.deploy else [],
            "cost_model": self.cost.describe() if self.cost else None, "policy_params": self.policy.n_params() if self.policy else None,
            "fleet": self.fleet, "telemetry_count": len(self.telemetry), "resources": self.resources(), "versions": versions, "config": self.cfg.summary(), "classes": D.CLASSES, "actions": S.ACTIONS,
        }
