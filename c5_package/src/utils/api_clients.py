from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import requests

from .io_utils import project_path
from .logger import mask_key


class MissingConfiguration(RuntimeError):
    pass


def _first_text_from_response(resp: Any) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    if hasattr(resp, "choices") and resp.choices:
        message = resp.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(item.text)
            return "\n".join(p for p in parts if p).strip()
    if hasattr(resp, "text"):
        return str(resp.text).strip()
    if isinstance(resp, dict):
        if "text" in resp:
            return str(resp["text"]).strip()
        choices = resp.get("choices") or []
        if choices:
            return str(choices[0].get("message", {}).get("content", "")).strip()
    return str(resp).strip()


def _first_audio_from_response(resp: Any) -> str | None:
    """Return base64 audio payload when the model provides one."""
    if hasattr(resp, "choices") and resp.choices:
        message = resp.choices[0].message
        audio = getattr(message, "audio", None)
        if audio is not None:
            return getattr(audio, "data", None) or (audio.get("data") if isinstance(audio, dict) else None)
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices:
            audio = choices[0].get("message", {}).get("audio")
            if isinstance(audio, dict):
                return audio.get("data")
    return None


def _stream_text_and_audio(chunks: Any) -> tuple[str, str | None]:
    text_parts: list[str] = []
    audio_parts: list[str] = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            text_parts.append(content)
        audio = getattr(delta, "audio", None)
        if audio:
            data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
            if data:
                audio_parts.append(data)
    return "".join(text_parts).strip(), ("".join(audio_parts) if audio_parts else None)


class AliOpenAIClient:
    """Thin adapter around Aliyun DashScope OpenAI-compatible endpoints."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("ALI_OPENAI_BASE_URL") or config.get("ali", {}).get("openai_base_url")
        missing = []
        if not api_key:
            missing.append("DASHSCOPE_API_KEY")
        if not base_url:
            missing.append("ALI_OPENAI_BASE_URL")
        if missing:
            raise MissingConfiguration("缺少环境变量: " + ", ".join(missing))
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.masked_key = mask_key(api_key)

    def preflight(self) -> tuple[bool, str | None]:
        """Check basic endpoint auth once so batch jobs can fail fast."""
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=20,
            )
            if resp.status_code == 401:
                return False, "API 鉴权失败: invalid_api_key，请检查 DASHSCOPE_API_KEY 与 ALI_OPENAI_BASE_URL 是否匹配"
            if resp.status_code >= 500:
                return False, f"API 服务暂时不可用: HTTP {resp.status_code}"
            return True, None
        except Exception as exc:
            return False, f"API 预检连接失败: {exc}"

    def chat_text(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> tuple[str, float]:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return _first_text_from_response(resp), time.perf_counter() - start

    def asr(self, model: str, audio_path: str | Path, data_url: str, prompt: str) -> tuple[str, float, bool]:
        """Call Qwen-ASR through the documented OpenAI-compatible chat endpoint."""
        start = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": data_url}}],
            }
        ]
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            extra_body={"asr_options": {"language": "en", "enable_itn": False}},
        )
        return _first_text_from_response(resp), time.perf_counter() - start, False

    def omni_translate(
        self,
        model: str,
        data_url: str,
        prompt: str,
        request_audio: bool,
        voice: str,
        audio_format: str,
    ) -> tuple[str, str | None, float]:
        def build_kwargs(with_audio: bool) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": data_url, "format": audio_format}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
                "modalities": ["text", "audio"] if with_audio else ["text"],
            }
            if with_audio:
                kwargs["audio"] = {"voice": voice, "format": audio_format}
            return kwargs

        start = time.perf_counter()
        try:
            chunks = self.client.chat.completions.create(**build_kwargs(request_audio))
            text, audio_b64 = _stream_text_and_audio(chunks)
        except Exception:
            if not request_audio:
                raise
            chunks = self.client.chat.completions.create(**build_kwargs(False))
            text, audio_b64 = _stream_text_and_audio(chunks)
        return text, audio_b64, time.perf_counter() - start

    def tts(self, model: str, text: str, voice: str, audio_format: str) -> tuple[bytes, float]:
        start = time.perf_counter()
        dashscope_base = (
            os.getenv("ALI_DASHSCOPE_BASE_URL")
            or self.config.get("ali", {}).get("dashscope_base_url")
            or self.base_url.replace("/compatible-mode/v1", "/api/v1")
        ).rstrip("/")
        resp = requests.post(
            f"{dashscope_base}/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": {
                    "text": text,
                    "voice": voice,
                    "language_type": "Chinese",
                },
            },
            timeout=int(self.config.get("api", {}).get("timeout_seconds", 120)),
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:500]}")
        payload = resp.json()
        audio = (payload.get("output") or {}).get("audio") or {}
        audio_b64 = audio.get("data")
        audio_url = audio.get("url")
        if audio_b64:
            return base64.b64decode(audio_b64), time.perf_counter() - start
        if audio_url:
            audio_resp = requests.get(audio_url, timeout=120)
            audio_resp.raise_for_status()
            return audio_resp.content, time.perf_counter() - start
        raise RuntimeError(f"TTS 响应中没有找到 output.audio.data 或 output.audio.url: {str(payload)[:500]}")
