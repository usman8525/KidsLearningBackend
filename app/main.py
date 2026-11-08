"""Speech server: keeps Azure/OpenAI credentials off the device and judges pronunciation.

Run locally:  uvicorn app.main:app --host 0.0.0.0 --port 4000
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from . import azure_speech
from .audio import FfmpegMissingError, is_conversion_error, to_wav_16k_mono
from .config import settings
from .evaluate import evaluate_assessment
from .openai_tips import generate_tip

log = logging.getLogger("speech-server")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# httpx logs every outgoing request at INFO; keep the log to our own lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Learning words are short: a letter, then letters, spaces, apostrophes or hyphens.
WORD_PATTERN = re.compile(r"[^\W\d_](?:[^\W\d_]|['\- ]){0,39}")
TTS_CACHE_LIMIT = 200
tts_cache: OrderedDict[str, bytes] = OrderedDict()


def parse_reference_text(value: Any) -> str:
    text = str(value or "").strip()
    if not WORD_PATTERN.fullmatch(text):
        raise HTTPException(400, "Invalid word")
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared HTTP client (connection pooling) for Azure/OpenAI calls.
    app.state.http = httpx.AsyncClient()
    if not settings.azure_configured:
        log.warning("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set - speech endpoints will return 503.")
    yield
    await app.state.http.aclose()


app = FastAPI(title="KidsPlay Speech Server", lifespan=lifespan, docs_url=None, redoc_url=None)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": "Server error" if status >= 500 else message}, status_code=status)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    # Never log request bodies/audio - only the route and an error summary.
    log.error("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail)
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    log.error("%s %s -> 400: invalid request", request.method, request.url.path)
    return _error(400, "Invalid request")


@app.exception_handler(azure_speech.NotConfiguredError)
async def not_configured(request: Request, exc: Exception):
    log.error("%s %s -> 503: %s", request.method, request.url.path, exc)
    return _error(503, str(exc))


@app.exception_handler(azure_speech.UpstreamError)
async def upstream_error(request: Request, exc: azure_speech.UpstreamError):
    log.error("%s %s -> %s: %s", request.method, request.url.path, exc.status, exc)
    return _error(exc.status, str(exc))


@app.middleware("http")
async def request_log(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    if settings.log_requests:
        ms = (time.perf_counter() - started) * 1000
        log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "azureConfigured": settings.azure_configured,
        "openaiConfigured": bool(settings.openai_key),
    }


@app.get("/api/tts")
async def tts(request: Request, text: str = Query("")):
    word = parse_reference_text(text).lower()
    audio = tts_cache.get(word)
    if audio is None:
        audio = await azure_speech.synthesize_speech(request.app.state.http, word)
        if len(tts_cache) >= TTS_CACHE_LIMIT:
            tts_cache.popitem(last=False)
        tts_cache[word] = audio
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})


async def _read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Recording too large")
    if not data:
        raise HTTPException(400, "Missing audio")
    return data


@app.post("/api/pronunciation/assess")
async def assess(
    request: Request,
    referenceText: str = Form(""),  # noqa: N803 - field names match the app's form
    audio: UploadFile | None = File(None),
):
    if audio is None:
        raise HTTPException(400, "Missing audio")
    data = await _read_limited(audio)
    reference = parse_reference_text(referenceText)

    try:
        wav = await to_wav_16k_mono(data)
    except FfmpegMissingError as e:
        raise HTTPException(500, str(e)) from e
    except Exception as e:  # noqa: BLE001
        if is_conversion_error(e):
            raise HTTPException(422, "Could not read the recording") from e
        raise

    client = request.app.state.http
    # Both calls run in parallel on the same audio; see evaluate.py for why both are needed.
    if settings.thresholds.require_free_recognition:
        assessment, free = await asyncio.gather(
            azure_speech.assess_pronunciation(client, wav, reference),
            azure_speech.recognize_speech(client, wav),
        )
    else:
        assessment, free = await azure_speech.assess_pronunciation(client, wav, reference), None
    return evaluate_assessment(assessment, reference, settings.thresholds, free)


class FeedbackRequest(BaseModel):
    referenceText: str = ""  # noqa: N815 - matches the app's JSON
    words: list[Any] = []


@app.post("/api/pronunciation/feedback")
async def feedback(request: Request, body: FeedbackRequest):
    reference = parse_reference_text(body.referenceText)
    return {"tip": await generate_tip(request.app.state.http, reference, body.words)}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
async def not_found(path: str):
    raise HTTPException(404, "Not found")
