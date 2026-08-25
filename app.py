"""Standalone entrypoint: serves the dashboard frontend and the /vllm-api/all
metrics feed it polls. All the actual scraping/history/pricing logic lives in
metrics.py — this file just wires it up as a small FastAPI app.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8765

Then open http://localhost:8765/ in a browser.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import metrics

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="vLLM Dashboard")


@app.on_event("startup")
def _start_metrics() -> None:
    metrics.start_poller()


@app.get("/vllm-api/all")
def vllm_api_all() -> dict:
    """Aggregate feed for static/index.html (live stats + history + pricing)."""
    return metrics.get_all()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
