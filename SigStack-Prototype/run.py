"""Start the SigStack demo:   python run.py [--config config.yaml] [--port 8766] [--no-browser]"""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from sigstack.api import create_app
from sigstack.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="SigStack demo server")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.get("server.host", "127.0.0.1")
    port = int(args.port or cfg.get("server.port", 8766))
    url = f"http://{host}:{port}/"

    if not cfg.cache_path.exists():
        print("Report cache missing — run:  python scripts/prepare_data.py")

    app = create_app(args.config)
    print(f"\n  SigStack demo  ->  {url}\n  API docs       ->  {url}api/docs\n")
    if not args.no_browser and bool(cfg.get("server.open_browser", True)):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
