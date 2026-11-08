"""Settings read from environment variables (and server/.env when present)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load server/.env for local runs. On a host (e.g. Render) variables come from its dashboard.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail thresholds. Azure scores are 0-100, confidence 0-1."""

    accuracy: float = field(default_factory=lambda: _num("PRONUNCIATION_ACCURACY_THRESHOLD", 75))
    word_accuracy: float = field(default_factory=lambda: _num("PRONUNCIATION_WORD_THRESHOLD", 70))
    completeness: float = field(default_factory=lambda: _num("PRONUNCIATION_COMPLETENESS_THRESHOLD", 80))
    min_confidence: float = field(default_factory=lambda: _num("PRONUNCIATION_MIN_CONFIDENCE", 0.5))
    # Every individual sound must score at least this.
    phoneme_accuracy: float = field(default_factory=lambda: _num("PRONUNCIATION_PHONEME_THRESHOLD", 50))
    # A sound counts as replaced (e.g. "f" said instead of "k") when Azure's most likely
    # heard sound beats the expected one by at least this many points...
    substitution_margin: float = field(default_factory=lambda: _num("PRONUNCIATION_SUBSTITUTION_MARGIN", 10))
    # ...unless that sound still scores at least this (accent variants).
    substitution_max_accuracy: float = field(
        default_factory=lambda: _num("PRONUNCIATION_SUBSTITUTION_MAX_ACCURACY", 80)
    )
    # Also require plain recognition (no reference text) to hear the word.
    require_free_recognition: bool = field(
        default_factory=lambda: _str("PRONUNCIATION_REQUIRE_FREE_RECOGNITION", "true").lower() != "false"
    )


@dataclass(frozen=True)
class Settings:
    azure_key: str = field(default_factory=lambda: _str("AZURE_SPEECH_KEY"))
    azure_region: str = field(default_factory=lambda: _str("AZURE_SPEECH_REGION").lower())
    azure_language: str = field(default_factory=lambda: _str("AZURE_SPEECH_LANGUAGE", "en-US"))
    azure_tts_voice: str = field(default_factory=lambda: _str("AZURE_TTS_VOICE", "en-US-AnaNeural"))
    openai_key: str = field(default_factory=lambda: _str("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _str("OPENAI_MODEL", "gpt-4o-mini"))
    max_upload_bytes: int = field(default_factory=lambda: int(_num("MAX_UPLOAD_BYTES", 2 * 1024 * 1024)))
    upstream_timeout_s: float = field(default_factory=lambda: _num("UPSTREAM_TIMEOUT_MS", 15000) / 1000)
    log_requests: bool = field(default_factory=lambda: _str("LOG_REQUESTS") == "1")
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_key and self.azure_region)


settings = Settings()
