"""HTTP API + static dashboard for the CAQ-EdgeNet demo."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT, load_config
from .engine import Engine

WEB = ROOT / "web"


class TrainRequest(BaseModel):
    epochs: int | None = None


class CtlRequest(BaseModel):
    episodes: int | None = None


class ToggleRequest(BaseModel):
    enabled: bool


class InspectRequest(BaseModel):
    index: int | None = None
    values: list[float] | None = None
    state: dict[str, Any] = {}


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg = load_config(config_path) if config_path else load_config()
    engine = Engine(cfg)
    app = FastAPI(title="CAQ-EdgeNet demo", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.engine = engine

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return engine.status()

    @app.get("/api/events")
    def events(since: int = 0) -> dict[str, Any]:
        return {"events": engine.events_since(since), "now": time.time()}

    @app.get("/api/resources")
    def resources() -> dict[str, Any]:
        return engine.resources()

    @app.get("/api/dataset")
    def dataset() -> dict[str, Any]:
        return engine.dataset_summary()

    # ------------------------------------------------------------------ detector
    @app.post("/api/detector/train")
    def detector_train(req: TrainRequest) -> dict[str, Any]:
        if engine.training:
            return {"started": False, "reason": "already running"}
        engine.train_detector(req.epochs)
        return {"started": True}

    @app.get("/api/detector/state")
    def detector_state() -> dict[str, Any]:
        return {"training": engine.training, "progress": engine.progress, "history": engine.history, "metrics": engine.metrics, "deploy": engine.deploy, "trained": engine.trained,
                "version": engine.detector_version, "ablation": engine.ablation, "ablation_running": engine.ablation_running, "path_accuracy": getattr(engine, "path_accuracy", None)}

    @app.post("/api/ablation/run")
    def ablation_run() -> dict[str, Any]:
        if engine.ablation_running or engine.training:
            return {"started": False, "reason": "busy"}
        engine.run_ablation()
        return {"started": True}

    # ------------------------------------------------------------------ controller
    @app.post("/api/controller/train")
    def controller_train(req: CtlRequest) -> dict[str, Any]:
        if engine.ctl_training:
            return {"started": False, "reason": "already running"}
        if not engine.trained:
            return {"started": False, "reason": "train the detector first"}
        engine.train_controller(req.episodes)
        return {"started": True}

    @app.get("/api/controller/state")
    def controller_state() -> dict[str, Any]:
        return {"training": engine.ctl_training, "progress": engine.ctl_progress, "history": engine.ctl_history, "evaluation": engine.ctl_eval, "version": engine.policy_version,
                "cost_model": engine.cost.describe() if engine.cost else None}

    # ------------------------------------------------------------------ fleet
    @app.post("/api/fleet")
    def fleet(req: ToggleRequest) -> dict[str, Any]:
        engine.set_fleet(req.enabled)
        return {"enabled": engine.fleet_running}

    @app.get("/api/fleet/state")
    def fleet_state() -> dict[str, Any]:
        return {"running": engine.fleet_running, **engine.fleet, "telemetry": list(engine.telemetry)[:12]}

    # ------------------------------------------------------------------ inspect
    @app.get("/api/samples")
    def samples() -> dict[str, Any]:
        return engine.samples()

    @app.post("/api/inspect")
    def inspect(req: InspectRequest) -> dict[str, Any]:
        return engine.inspect(req.index, req.values, req.state)

    # ------------------------------------------------------------------ static
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.exception_handler(Exception)
    async def on_error(_, exc: Exception):
        engine.log(f"API error: {exc}", "error")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
