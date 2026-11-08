"""Azure Speech REST calls: Text-to-Speech, Speech-to-Text and Pronunciation Assessment."""

from __future__ import annotations

import base64
import json
from xml.sax.saxutils import escape

import httpx

from .config import settings


class UpstreamError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class NotConfiguredError(Exception):
    pass


def _require_config() -> None:
    if not settings.azure_configured:
        raise NotConfiguredError("Azure Speech is not configured")


async def _post(client: httpx.AsyncClient, service: str, url: str, **kwargs) -> httpx.Response:
    try:
        res = await client.post(url, timeout=settings.upstream_timeout_s, **kwargs)
    except httpx.TimeoutException as e:
        raise UpstreamError(f"{service} timed out", 504) from e
    except httpx.HTTPError as e:
        raise UpstreamError(f"{service} unreachable", 502) from e
    if res.status_code >= 400:
        raise UpstreamError(f"{service} responded with {res.status_code}", 502)
    return res


async def synthesize_speech(client: httpx.AsyncClient, text: str) -> bytes:
    """Azure Text-to-Speech -> MP3 bytes. Slightly slowed down for children."""
    _require_config()
    ssml = (
        f'<speak version="1.0" xml:lang="{settings.azure_language}">'
        f'<voice name="{settings.azure_tts_voice}"><prosody rate="-15%">{escape(text)}</prosody></voice>'
        "</speak>"
    )
    res = await _post(
        client,
        "Azure TTS",
        f"https://{settings.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Ocp-Apim-Subscription-Key": settings.azure_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
            "User-Agent": "kidsplay-speech-server",
        },
        content=ssml.encode("utf-8"),
    )
    return res.content


async def _recognize(client: httpx.AsyncClient, wav: bytes, extra_headers: dict[str, str]) -> dict:
    url = (
        f"https://{settings.azure_region}.stt.speech.microsoft.com/speech/recognition/"
        f"conversation/cognitiveservices/v1"
    )
    res = await _post(
        client,
        "Azure STT",
        url,
        params={"language": settings.azure_language, "format": "detailed"},
        headers={
            "Ocp-Apim-Subscription-Key": settings.azure_key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
            **extra_headers,
        },
        content=wav,
    )
    return res.json()


async def assess_pronunciation(client: httpx.AsyncClient, wav: bytes, reference_text: str) -> dict:
    """Speech-to-Text with Pronunciation Assessment against the expected word.

    Returns the raw Azure JSON (recognised text, accuracy/completeness/fluency and
    word- and phoneme-level scores) for evaluate.py to judge.
    """
    _require_config()
    params = {
        "ReferenceText": reference_text,
        "GradingSystem": "HundredMark",
        "Granularity": "Phoneme",
        "Dimension": "Comprehensive",
        "EnableMiscue": True,
        # Return the sounds Azure actually heard at each position, so substitutions
        # like "fat" for "cat" (f instead of k) can be detected.
        "PhonemeAlphabet": "IPA",
        "NBestPhonemeCount": 5,
    }
    header = base64.b64encode(json.dumps(params).encode("utf-8")).decode("ascii")
    return await _recognize(client, wav, {"Pronunciation-Assessment": header})


async def recognize_speech(client: httpx.AsyncClient, wav: bytes) -> dict:
    """Plain Speech-to-Text with NO reference text.

    When a reference is supplied, Azure is biased towards reporting it ("fat" comes back
    as "cat"); this independent pass is not.
    """
    _require_config()
    return await _recognize(client, wav, {})
