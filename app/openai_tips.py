"""Optional, kid-friendly retry tip from OpenAI.

Only Azure's scores are sent, never the child's audio. Does NOT decide pass/fail.
Returns None when OpenAI is not configured or fails.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import settings

SYSTEM_PROMPT = (
    "You coach a 4-7 year old learning English words. Reply with ONE short, cheerful sentence "
    "(max 12 words) telling them how to say the word better. No negative words, no scores, no emojis."
)


def _weak_sounds(words: Any) -> list[str]:
    out: list[str] = []
    for word in words if isinstance(words, list) else []:
        for p in (word or {}).get("phonemes") or []:
            accuracy = p.get("accuracy")
            if not isinstance(accuracy, (int, float)) or not (accuracy < 60 or p.get("substituted")):
                continue
            heard = f", sounded like /{str(p.get('heard'))[:8]}/" if p.get("substituted") and p.get("heard") else ""
            out.append(f"/{str(p.get('phoneme'))[:8]}/ ({accuracy}{heard})")
    return out[:6]


async def generate_tip(client: httpx.AsyncClient, reference_text: str, words: Any) -> str | None:
    if not settings.openai_key:
        return None
    weak = ", ".join(_weak_sounds(words)) or "none specific"
    try:
        res = await client.post(
            "https://api.openai.com/v1/chat/completions",
            timeout=6,
            headers={"Authorization": f"Bearer {settings.openai_key}"},
            json={
                "model": settings.openai_model,
                "max_tokens": 40,
                "temperature": 0.4,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f'Word: "{reference_text}". Sounds that need practice '
                        f"(IPA phoneme, score 0-100): {weak}.",
                    },
                ],
            },
        )
        if res.status_code >= 400:
            return None
        tip = (res.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        return tip[:120] or None
    except (httpx.HTTPError, ValueError, KeyError, IndexError, AttributeError):
        return None
