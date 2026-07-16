from __future__ import annotations

import re


def chinese_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    chinese = [c for c in chars if "\u4e00" <= c <= "\u9fff"]
    return len(chinese) / len(chars)


def compact_text(text: str, max_len: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_len]


def extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

