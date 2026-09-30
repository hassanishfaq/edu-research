"""HTTP API + static dashboard for the FedStack demo."""
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


class FedRequest(BaseModel):
    rounds: int | None = None
    epochs: int | None = None


class WatchRequest(BaseModel):
    enabled: bool


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg = load_config(config_path) if config_path else load_config()
    engine = Engine(cfg)
    app = FastAPI(title="FedStack demo", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.engine = engine

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return engine.status()

    @app.get("/api/events")
    def events(since: int = 0) -> dict[str, Any]:
        return {"events": engine.events_since(since), "now": time.time()}

    @app.get("/api/resources")
    def resources() -> dict[str, Any]:
        return engine.resources()

    # ------------------------------------------------------------------ scanning
    @app.post("/api/scan")
    async def scan(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(413, "file larger than 64 MB")
        return engine.scan_bytes(data, file.filename or "upload.bin", source="upload")

    @app.get("/api/samples")
    def samples() -> dict[str, Any]:
        return engine.samples()

    @app.get("/api/samples/file/{path:path}")
    def sample_file(path: str):
        p = (cfg.samples_dir / path).resolve()
        if not str(p).startswith(str(cfg.samples_dir.resolve())) or not p.exists():
            raise HTTPException(404, path)
        return FileResponse(p)

    @app.post("/api/scan_sample")
    def scan_sample(req: SamplePath) -> dict[str, Any]:
        try:
            return engine.scan_sample(req.path)
        except FileNotFoundError:
            raise HTTPException(404, req.path)

    # ------------------------------------------------------------------ federation
    @app.post("/api/federation/run")
    def federation_run(req: FedRequest) -> dict[str, Any]:
        if engine.federation_running:
            return {"started": False, "reason": "already running"}
        engine.run_federation(req.rounds, req.epochs)
        return {"started": True}

    @app.get("/api/federation/state")
    def federation_state() -> dict[str, Any]:
        return {"running": engine.federation_running, "progress": engine.federation_progress, "history": engine.history,
                "meta": engine.meta_metrics, "threshold": engine.threshold, "payload_now": engine.payload_round() if engine.nodes and engine.cache else None,
                "nodes": [{"name": rt.name, "label": rt.label, "val_accuracy": rt.val_accuracy} for rt in engine.nodes]}

    # ------------------------------------------------------------------ zero-day
    @app.post("/api/zeroday/run")
    def zeroday_run() -> dict[str, Any]:
        return engine.zero_day_eval()

    @app.get("/api/zeroday/state")
    def zeroday_state() -> dict[str, Any]:
        return engine.zero_day_result or {"status": "idle"}

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
