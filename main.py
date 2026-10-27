"""TypeSafe Jev-compatible API (POST /v1/systemone) served by a local Laya model.

    uv run main.py                     # http://127.0.0.1:8000/docs
    LAYA_MODEL=... LAYA_DEVICE=cuda LAYA_PORT=... uv run main.py
"""

import logging
import os
import platform
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from rich.markup import escape

# Apple Silicon -> laya-mlx (MLX). Everything else -> upstream laya (PyTorch: CUDA/XPU/CPU).
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
BACKEND = os.environ.get("LAYA_BACKEND", "mlx" if APPLE_SILICON else "torch")
if BACKEND == "mlx":
    import laya_mlx as laya

    DEFAULT_MODEL = "aac6fef/laya-mlx"
else:
    import laya

    DEFAULT_MODEL = "convaiinnovations/laya"

MODEL_ID = os.environ.get("LAYA_MODEL", DEFAULT_MODEL)
DEVICE = os.environ.get("LAYA_DEVICE") or None  # None = backend picks (torch: cuda → mps → xpu → cpu)
MODEL_NAME = "laya"

log = logging.getLogger("laya")
MARKUP = {"markup": True}

# --- Request schema (Jev /v1/systemone; box questions and image assets aren't supported) ---

Content = str | dict[str, Any] | list[Any]


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content | None = None
    criteria: dict[Literal["true", "false"], Content | None] | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content | None = None
    criteria: dict[str, Content | None] | list[str] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content | None = None
    criteria: list[Content] = Field(min_length=1, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model: str = Field(examples=["jev-latest"])
    state: Content = Field(examples=["I was charged twice. Please refund the duplicate."])
    questions: dict[str, Question] = Field(min_length=1)


# --- App ---

agent = None
# ponytail: one forward pass at a time on one model; add a worker pool if throughput matters.
predict_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global agent
    t0 = time.perf_counter()
    log.info(f"loading [magenta]{escape(MODEL_ID)}[/] on [bold]{BACKEND}[/] …", extra=MARKUP)
    agent = laya.load(MODEL_ID, device=DEVICE)
    device = getattr(agent, "device", DEVICE or "auto")
    # Pay first-call costs (kernel compile, allocator warmup) at boot, not on the first request.
    agent.predict("warmup", {"q": {"type": "noul", "instructions": "Is this a warmup?"}})
    log.info(f"model ready in {time.perf_counter() - t0:.1f}s · device [bold]{escape(str(device))}[/]", extra=MARKUP)
    yield


app = FastAPI(title="Laya System One", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    status = response.status_code
    color = "green" if status < 400 else "yellow" if status < 500 else "red"
    level = logging.DEBUG if request.url.path == "/health" else logging.INFO
    log.log(level, f"[bold]{request.method}[/] {escape(request.url.path)} [{color}]{status}[/] {ms:.1f}ms", extra=MARKUP)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "backend": BACKEND}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    # Sync def: FastAPI runs it in a worker thread, so inference doesn't block the event loop.
    # Drop unset top-level fields only; exclude_none would also strip `{"billing": None}` choices.
    questions = {
        name: {k: v for k, v in q.model_dump().items() if v is not None}
        for name, q in req.questions.items()
    }
    t0 = time.perf_counter()
    try:
        with predict_lock:
            result = agent.predict(req.state, questions)
    except ValueError as e:  # laya's question validation errors are safe to show
        raise HTTPException(status_code=422, detail=str(e))
    ms = (time.perf_counter() - t0) * 1000

    usage = result.get("usage", {})
    response = {
        "model": MODEL_NAME,
        "answers": result["answers"],
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)},
    }
    _log_answers(req, response, ms)
    return response


def _log_answers(req, response, ms):
    # Never log `state` itself: it's user content. Its size is enough to debug with.
    size = len(req.state) if isinstance(req.state, str) else len(str(req.state))
    u = response["usage"]
    log.info(
        f"inference {ms:.0f}ms · state {size} chars · {len(req.questions)} q · "
        f"tok {u['input_tokens']}/{u['output_tokens']}",
        extra=MARKUP,
    )
    for name, a in response["answers"].items():
        t = a.get("type")
        if t == "choice":
            value = escape(str(a["choice"]))
        elif t == "score":
            value = f"{a['score']:.2f}"
        else:
            value = f"{a.get('noul', 0):.2f}"
        c = a.get("confidence")
        conf = "" if c is None else f"[{'green' if c >= 0.8 else 'yellow' if c >= 0.5 else 'red'}]conf {c:.2f}[/]"
        log.info(f"   [cyan]{escape(name)}[/] [dim]{t}[/] → [bold]{value}[/] {conf}", extra=MARKUP)


def main():
    logging.basicConfig(
        level=os.environ.get("LAYA_LOG_LEVEL", "info").upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # HF Hub download chatter
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "127.0.0.1"),
        port=int(os.environ.get("LAYA_PORT", "8000")),
        log_config=None,  # uvicorn's loggers propagate to the Rich handler
        access_log=False,  # replaced by access_log middleware
    )


if __name__ == "__main__":
    main()
