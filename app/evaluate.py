"""Decides whether an attempt passes.

Inputs are two Azure results for the same audio:
 1. Pronunciation Assessment (reference text = expected word): accuracy scores and,
    per expected sound, the sounds Azure actually heard (NBestPhonemes).
 2. Plain recognition with NO reference text.

Why both: with a reference text Azure is biased to report the expected word. Saying
"fat" for "cat" comes back as "cat" with a decent overall score. Measured with real
Azure audio: "cut"/"cap" scored 86-90 against "cat", the same as a correct "cat".
So the overall score alone can't be trusted for short words. An attempt passes only if:
 - every expected sound was the sound actually heard (no substitution) and scores well,
 - plain recognition independently heard the expected word,
 - the overall/word/completeness scores clear their thresholds.
Anything unclear fails.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .config import Thresholds

NO_SPEECH_STATUSES = {"NoMatch", "InitialSilenceTimeout", "BabbleTimeout"}


def normalize_text(text: Any) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    value = unicodedata.normalize("NFKD", str(text or "").lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^\w\s']", " ", value).replace("_", " ").replace("'", "")
    return re.sub(r"\s+", " ", value).strip()


def _pa(obj: Any) -> dict:
    """Newer API versions nest scores under `PronunciationAssessment`; older ones don't."""
    if not isinstance(obj, dict):
        return {}
    return obj.get("PronunciationAssessment") or obj


def _score(value: Any) -> int:
    return round(value) if isinstance(value, (int, float)) else 0


def _result(passed, reason, recognized="", scores=None, words=None, heard="") -> dict:
    return {
        "passed": passed,
        "reason": reason,
        "recognizedText": recognized,
        "heardText": heard,
        "scores": scores,
        "words": words or [],
    }


def _analyse_phoneme(phoneme: dict, margin: float) -> dict:
    """Sound Azure most likely heard where the expected one should be, and whether it replaced it."""
    expected = phoneme.get("Phoneme")
    candidates = _pa(phoneme).get("NBestPhonemes") or []
    top = candidates[0] if candidates else None
    if not top or top.get("Phoneme") == expected:
        return {"heard": (top or {}).get("Phoneme", expected), "substituted": False}
    expected_score = next((c.get("Score") for c in candidates if c.get("Phoneme") == expected), 0)
    return {
        "heard": top.get("Phoneme"),
        "substituted": _score(top.get("Score")) - _score(expected_score) >= margin,
    }


def _contains_all(text: str, expected_words: list[str]) -> bool:
    heard = set(normalize_text(text).split(" "))
    return all(w in heard for w in expected_words)


def evaluate_assessment(azure: dict | None, reference: str, t: Thresholds, free: dict | None = None) -> dict:
    """Return the API response for one attempt (see POST /api/pronunciation/assess)."""
    if not azure or azure.get("RecognitionStatus") in NO_SPEECH_STATUSES:
        return _result(False, "no_speech")
    nbest = azure.get("NBest") or []
    best = nbest[0] if azure.get("RecognitionStatus") == "Success" and nbest else None
    if not best:
        return _result(False, "unclear")

    recognized = normalize_text(best.get("Lexical") or best.get("Display") or azure.get("DisplayText"))
    expected_words = [w for w in normalize_text(reference).split(" ") if w]
    overall = _pa(best)

    words = [
        {
            "word": normalize_text(w.get("Word")),
            "accuracy": _score(_pa(w).get("AccuracyScore")),
            "errorType": _pa(w).get("ErrorType", "None"),
            "phonemes": [
                {
                    "phoneme": p.get("Phoneme"),
                    "accuracy": _score(_pa(p).get("AccuracyScore")),
                    **_analyse_phoneme(p, t.substitution_margin),
                }
                for p in w.get("Phonemes") or []
            ],
        }
        for w in best.get("Words") or []
    ]

    confidence = best.get("Confidence")
    scores = {
        "accuracy": _score(overall.get("AccuracyScore")),
        "pronunciation": _score(overall.get("PronScore")),
        "completeness": _score(overall.get("CompletenessScore")),
        "fluency": _score(overall.get("FluencyScore")),
        "confidence": confidence if isinstance(confidence, (int, float)) else 0,
    }

    # Independent recognition (no reference text): what was really said.
    free_alternatives = (
        [n.get("Lexical") or n.get("Display") or "" for n in free.get("NBest") or []]
        if free and free.get("RecognitionStatus") == "Success"
        else []
    )
    heard = normalize_text(free_alternatives[0]) if free_alternatives else ""

    def fail(reason: str) -> dict:
        return _result(False, reason, recognized, scores, words, heard)

    if not recognized:
        return fail("no_speech")
    if scores["confidence"] < t.min_confidence:
        return fail("unclear")

    # Every expected word must be present and not flagged omitted by miscue detection.
    reference_words = [w for w in words if w["errorType"] != "Insertion"]
    omitted = any(w["errorType"] == "Omission" for w in reference_words)
    if not _contains_all(recognized, expected_words) or omitted or not reference_words:
        return fail("wrong_word")

    if t.require_free_recognition and free is not None:
        if not free_alternatives:
            return fail("unclear")
        if not any(_contains_all(alt, expected_words) for alt in free_alternatives):
            return fail("wrong_word")

    # Sound-level check: no replaced sounds, no very weak sounds. A "different" sound that
    # still scores very high is usually an accent variant (e.g. the vowel in "ball"); real
    # word swaps there are caught by plain recognition.
    sounds_ok = all(
        p["accuracy"] >= t.phoneme_accuracy
        and not (p["substituted"] and p["accuracy"] < t.substitution_max_accuracy)
        for w in reference_words
        for p in w["phonemes"]
    )
    words_ok = all(
        w["errorType"] != "Mispronunciation" and w["accuracy"] >= t.word_accuracy for w in reference_words
    )
    passed = (
        sounds_ok
        and words_ok
        and scores["accuracy"] >= t.accuracy
        and scores["completeness"] >= t.completeness
    )
    return _result(True, "ok", recognized, scores, words, heard) if passed else fail("low_accuracy")
