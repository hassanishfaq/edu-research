"""HTTP API + static dashboard for the SigStack demo."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT, load_config
from .engine import Engine

WEB = ROOT / "web"


class SamplePath(BaseModel):
    path: str


class HoldoutRequest(BaseModel):
    categories: list[str] | None = None


class WatchRequest(BaseModel):
    enabled: bool


class MapRequest(BaseModel):
    engine: str
    raw: str


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg = load_config(config_path) if config_path else load_config()
    engine = Engine(cfg)
    app = FastAPI(title="SigStack demo", version="0.1.0", docs_url="/api/docs", redoc_url=None)
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

    # ------------------------------------------------------------------ analyse
    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        if len(data) > 32 * 1024 * 1024:
            raise HTTPException(413, "report larger than 32 MB")
        return engine.analyze(data, file.filename or "report.json", source="upload")

    @app.get("/api/samples")
    def samples() -> dict[str, Any]:
        return engine.samples()

    @app.get("/api/samples/file/{path:path}")
    def sample_file(path: str):
        p = (cfg.samples_dir / path).resolve()
        if not str(p).startswith(str(cfg.samples_dir.resolve())) or not p.exists():
            raise HTTPException(404, path)
        return FileResponse(p)

    @app.post("/api/analyze_sample")
    def analyze_sample(req: SamplePath) -> dict[str, Any]:
        try:
            return engine.analyze_sample(req.path)
        except FileNotFoundError:
            raise HTTPException(404, req.path)

    # ------------------------------------------------------------------ label service
    @app.get("/api/labels")
    def labels() -> dict[str, Any]:
        return {"stats": engine.label_stats, "map_version": engine.cmap.version, "categories": engine.categories, "rules": engine.cmap.rules_table()}

    @app.post("/api/labels/map")
    def map_string(req: MapRequest) -> dict[str, Any]:
        cat, rule = engine.cmap.map(req.engine, req.raw)
        return {"engine": req.engine, "raw": req.raw, "category": cat, "rule": rule, "map_version": engine.cmap.version}

    # ------------------------------------------------------------------ training
    @app.post("/api/train/run")
    def train_run() -> dict[str, Any]:
        if engine.training:
            return {"started": False, "reason": "already running"}
        engine.train()
        return {"started": True}

    @app.get("/api/train/state")
    def train_state() -> dict[str, Any]:
        return {"training": engine.training, "progress": engine.progress, "history": engine.history, "metrics": engine.metrics, "threshold": engine.threshold,
                "model_version": engine.model_version}

    # ------------------------------------------------------------------ hold-out
    @app.post("/api/holdout/run")
    def holdout_run(req: HoldoutRequest) -> dict[str, Any]:
        if engine.holdout_running:
            return {"started": False, "reason": "already running"}
        engine.run_holdout(req.categories)
        return {"started": True}

    @app.get("/api/holdout/state")
    def holdout_state() -> dict[str, Any]:
        return {"running": engine.holdout_running, "progress": engine.holdout_progress, "result": engine.holdout_result}

    # ------------------------------------------------------------------ watch
    @app.post("/api/watch")
    def watch(req: WatchRequest) -> dict[str, Any]:
        engine.set_watch(req.enabled)
        return {"enabled": engine.watch_enabled, "dir": str(cfg.resolve(cfg.get("watch.dir", "watch_inbox")))}

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
