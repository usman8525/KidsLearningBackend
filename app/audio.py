"""Converts uploads to the WAV format Azure's short-audio REST API requires."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """FFMPEG_PATH env -> bundled imageio-ffmpeg binary -> `ffmpeg` on PATH."""
    if os.getenv("FFMPEG_PATH"):
        return os.environ["FFMPEG_PATH"]
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency or download failure
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise FfmpegMissingError("ffmpeg not found - install it or set FFMPEG_PATH")
    return found


def _convert(data: bytes, timeout_s: float) -> bytes:
    tmp = Path(tempfile.gettempdir())
    token = uuid.uuid4().hex
    src, dst = tmp / f"attempt-{token}.in", tmp / f"attempt-{token}.wav"
    try:
        src.write_bytes(data)
        # M4A can't be decoded from a pipe (its index is at the end), hence temp files.
        subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", "-f", "wav", str(dst)],
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return dst.read_bytes()
    finally:
        # Recordings are never kept on disk.
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


async def to_wav_16k_mono(data: bytes, timeout_s: float = 10) -> bytes:
    """Android records AAC/M4A; Azure needs 16 kHz mono 16-bit PCM WAV."""
    return await asyncio.to_thread(_convert, data, timeout_s)


def is_conversion_error(exc: Exception) -> bool:
    return isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
