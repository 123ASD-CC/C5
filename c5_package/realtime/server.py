from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

import dashscope
import numpy as np
import requests
from dashscope.audio.asr import Recognition, RecognitionCallback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel
from scipy.signal import lfilter

from src.meeting_library import (
    add_realtime_meeting,
    load_meeting_detail,
    meeting_context as library_meeting_context,
    update_meeting_diarization,
)
from src.meeting_rag import retrieve as retrieve_rag_evidence
from src.live_translate import LiveTranslateSession
from src.meeting_diarization import align_transcript_to_turns, diarize_audio
from src.online_diarization import (
    CAMPPLUS_MODEL,
    ERES2NETV2_MODEL,
    StreamingSpeakerTracker,
    choose_speaker_embedding_model,
    create_embedder,
)
from src.online_vad import StreamingVoiceGate, warm_streaming_vad


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "realtime" / "index.html"

ASR_MODEL = os.getenv("REALTIME_ASR_MODEL", "paraformer-realtime-v2")
LLM_MODEL = os.getenv("REALTIME_TRANSLATE_MODEL", "qwen-plus")
SAMPLE_RATE = 16000
DEFAULT_MAX_SENTENCE_SILENCE_MS = int(os.getenv("REALTIME_MAX_SENTENCE_SILENCE_MS", "2200"))
DIARIZATION_WINDOW_SECONDS = float(os.getenv("DIARIZATION_WINDOW_SECONDS", "2.5"))
DIARIZATION_HOP_SECONDS = float(os.getenv("DIARIZATION_HOP_SECONDS", "0.8"))
DIARIZATION_INITIAL_WINDOW_SECONDS = float(os.getenv("DIARIZATION_INITIAL_WINDOW_SECONDS", "2.2"))
DIARIZATION_INITIAL_MIN_WINDOW_SECONDS = float(
    os.getenv("DIARIZATION_INITIAL_MIN_WINDOW_SECONDS", "1.0")
)
DIARIZATION_MIN_WINDOW_SECONDS = float(os.getenv("DIARIZATION_MIN_WINDOW_SECONDS", "1.5"))
SPEAKER_TURN_GAP_SECONDS = float(os.getenv("SPEAKER_TURN_GAP_SECONDS", "0.70"))
SPEAKER_CHANGE_MIN_DURATION_SECONDS = float(
    os.getenv("REALTIME_SPEAKER_CHANGE_MIN_SECONDS", "1.8")
)
SPEAKER_PROFILE_UPDATE_MIN_MATCHES = int(
    os.getenv("REALTIME_SPEAKER_PROFILE_UPDATE_MIN_MATCHES", "2")
)
SPEAKER_PROFILE_UPDATE_MIN_CONFIDENCE = float(
    os.getenv("REALTIME_SPEAKER_PROFILE_UPDATE_MIN_CONFIDENCE", "0.80")
)
# RMS is only a final sanity check after the streaming VAD gate.  It must not
# be used as the primary speech detector: room noise often sits above 0.0005.
SPEAKER_MIN_RMS = float(os.getenv("SPEAKER_MIN_RMS", "0.003"))
SPEAKER_TARGET_RMS = float(os.getenv("SPEAKER_TARGET_RMS", "0.04"))
REALTIME_RECALIBRATION_LOOKBACK_SECONDS = float(os.getenv("REALTIME_RECALIBRATION_LOOKBACK_SECONDS", "180"))
REALTIME_RECALIBRATION_MIN_SECONDS = float(os.getenv("REALTIME_RECALIBRATION_MIN_SECONDS", "4"))
REALTIME_RECALIBRATION_INTERVAL_SECONDS = float(os.getenv("REALTIME_RECALIBRATION_INTERVAL_SECONDS", "30"))
REALTIME_FINAL_RECALIBRATION_MAX_SECONDS = float(os.getenv("REALTIME_FINAL_RECALIBRATION_MAX_SECONDS", "600"))

_speaker_embedders: dict[str, Any] = {}
_speaker_embedder_lock = threading.Lock()


def get_speaker_embedder(model_name: str | None = None) -> Any:
    model_key = model_name or choose_speaker_embedding_model()
    with _speaker_embedder_lock:
        if model_key not in _speaker_embedders:
            _speaker_embedders[model_key] = create_embedder(model_name=model_key)
        return _speaker_embedders[model_key]


def normalize_speaker_window(pcm: np.ndarray) -> np.ndarray:
    """Suppress DC/low-frequency rumble, then normalize voiced samples.

    This is applied only to the selected speaker embedding model. ASR and
    translation continue to receive the exact browser PCM stream.
    """
    signal = np.asarray(pcm, dtype="float32")
    if len(signal):
        signal = signal - float(np.mean(signal))
    if len(signal) > 1:
        # One-pole high-pass around the speech band floor.  It removes headset
        # handling/air-conditioner rumble without the latency and distortion
        # of a full spectral denoiser on a short online embedding window.
        alpha = float(os.getenv("SPEAKER_HIGHPASS_ALPHA", "0.97"))
        alpha = min(0.999, max(0.90, alpha))
        # Use a stateful filter implementation.  A vectorized assignment such
        # as ``filtered[1:] = ... + alpha * filtered[:-1]`` reads uninitialized
        # recursive samples because the right-hand side is evaluated before
        # the slice is written, which can corrupt every embedding window.
        signal = lfilter([1.0, -1.0], [1.0, -alpha], signal).astype("float32", copy=False)
    rms = float(np.sqrt(np.mean(signal * signal))) if len(signal) else 0.0
    if rms <= 0:
        return signal
    gain = min(20.0, SPEAKER_TARGET_RMS / rms)
    return np.clip(signal * gain, -1.0, 1.0)


def align_pcm16_bytes(frame: bytes) -> bytes:
    """Drop an incomplete trailing sample from malformed PCM input.

    Browser microphone packets should always contain little-endian int16
    samples, but a user-provided or interrupted packet can end on an odd byte.
    Every downstream route expects complete samples, so align once at the
    WebSocket boundary instead of allowing one packet to poison the speaker
    window and the synchronized ASR stream.
    """

    raw = bytes(frame)
    return raw if len(raw) % 2 == 0 else raw[:-1]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_speaker_model() -> None:
    # Do not make the first person who speaks pay the model-loading cost.
    threading.Thread(target=get_speaker_embedder, daemon=True).start()
    configured = os.getenv("SPEAKER_EMBEDDING_MODEL", "auto").strip().lower()
    if configured in {"", "auto", "default"}:
        # Keep the Chinese-only route warm as well.  The two checkpoints are
        # cached independently, so a language switch does not add a first-turn
        # model download or initialization pause.
        threading.Thread(
            target=get_speaker_embedder,
            args=(ERES2NETV2_MODEL,),
            daemon=True,
        ).start()
    threading.Thread(target=warm_streaming_vad, daemon=True).start()


class AssistantRequest(BaseModel):
    question: str
    enable_search: bool = True
    search_mode: str = "force"
    conversation: list[dict[str, str]] = []


class RealtimeMeetingSaveRequest(BaseModel):
    title: str = "实时麦克风会议"
    audio_b64: str
    transcript: list[dict[str, Any]] = []
    duration_seconds: float = 0
    route_mode: str = "cascade"
    canonical_route: str = "cascade"
    route_transcripts: dict[str, list[dict[str, Any]]] = {}
    comparison_metrics: dict[str, Any] = {}


class RealtimeStageSummaryRequest(BaseModel):
    transcript: list[dict[str, Any]] = []
    window: str = ""


class RealtimeDiarizationRequest(BaseModel):
    audio_b64: str
    transcript: list[dict[str, Any]] = []


class StoredMeetingDiarizationRequest(BaseModel):
    speaker_count: int | None = None


def run_local_diarization(audio_b64: str, transcript: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full-recording speaker reconciliation without blocking the WebSocket."""
    try:
        encoded = audio_b64.split(",", 1)[1] if "," in audio_b64 else audio_b64
        audio_bytes = base64.b64decode(encoded)
        result = diarize_audio(audio_bytes)
        if not result.get("success"):
            result["status"] = "pending_post_diarization"
            return transcript, result
        corrected = align_transcript_to_turns(transcript, result.get("turns") or [])
        corrected_count = sum(1 for row in corrected if row.get("speaker_source") == "3d-speaker-local")
        result["corrected_transcript_count"] = corrected_count
        result["status"] = "resolved" if corrected_count else "audio_diarized"
        return corrected, result
    except Exception as exc:
        return transcript, {
            "success": False,
            "status": "pending_post_diarization",
            "backend": "3d-speaker-local",
            "error": str(exc),
            "turns": [],
        }


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_meeting_voice_profiles(meeting_id: str) -> list[tuple[str, np.ndarray | list[np.ndarray]]]:
    """Load stable speaker centroids from a completed meeting correction."""
    if not meeting_id:
        return []
    try:
        detail = load_meeting_detail(meeting_id)
        diarization = detail.get("diarization") or {}
        centroids = diarization.get("speaker_centroids") or []
        if not isinstance(centroids, list):
            return []
        names_by_id = {
            str(row.get("speaker_id")): str(row.get("speaker"))
            for row in diarization.get("turns") or []
            if row.get("speaker_id") is not None and row.get("speaker")
        }
        profiles: list[tuple[str, np.ndarray | list[np.ndarray]]] = []
        for row in centroids:
            if not isinstance(row, dict):
                continue
            speaker_id = str(row.get("speaker_id", ""))
            label = names_by_id.get(speaker_id)
            if not label:
                try:
                    index = int(speaker_id)
                    label = f"参会者{chr(ord('A') + index)}" if index < 26 else f"参会者{index + 1}"
                except (TypeError, ValueError):
                    label = str(row.get("speaker") or "")
            raw_prototypes = row.get("prototypes") or [row.get("embedding")]
            prototypes = [
                np.asarray(value, dtype="float32").reshape(-1)
                for value in raw_prototypes
                if value is not None and np.asarray(value).size
            ]
            if label and prototypes:
                profiles.append((label, prototypes))
        return profiles
    except Exception:
        return []


def load_meeting_reference_turns(meeting_id: str) -> list[dict[str, Any]]:
    """Load the completed meeting timeline used only for calibrated playback."""
    if not meeting_id:
        return []
    try:
        diarization = load_meeting_detail(meeting_id).get("diarization") or {}
        turns: list[dict[str, Any]] = []
        for row in diarization.get("turns") or []:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", 0.0))
            if end <= start:
                continue
            speaker = str(row.get("speaker") or "")
            if not speaker:
                try:
                    index = int(row.get("speaker_id"))
                    speaker = f"参会者{chr(ord('A') + index)}" if index < 26 else f"参会者{index + 1}"
                except (TypeError, ValueError):
                    continue
            turns.append({"start": start, "end": end, "speaker": speaker})
        return turns
    except Exception:
        return []
def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if os.getenv("ALI_DASHSCOPE_BASE_URL"):
    dashscope.base_http_api_url = os.getenv("ALI_DASHSCOPE_BASE_URL")


class QueueCallback(RecognitionCallback):
    def __init__(self, events: "queue.Queue[dict[str, Any]]", started: threading.Event | None = None, route: str = "cascade"):
        self.events = events
        self.started = started
        self.route = route

    def on_open(self) -> None:
        if self.started is not None:
            self.started.set()
        self.events.put({"type": "asr_open", "route": self.route})

    def on_close(self) -> None:
        self.events.put({"type": "asr_close"})

    def on_complete(self) -> None:
        self.events.put({"type": "asr_complete"})

    def on_error(self, message) -> None:
        try:
            detail = str(message)
        except Exception:
            try:
                payload = message.to_dict() if hasattr(message, "to_dict") else {}
                detail = json.dumps(payload, ensure_ascii=False) if payload else repr(message)
            except Exception:
                detail = f"{type(message).__name__}: ASR provider error"
        self.events.put({"type": "error", "message": detail[:500]})

    def on_event(self, result) -> None:
        data = _result_to_dict(result)
        text = _extract_text(data)
        if not text:
            return
        final = _is_final(data)
        self.events.put(
            {
                "type": "asr_final" if final else "asr_partial",
                "text": text,
                "raw": data,
                "route": self.route,
            }
        )


def _result_to_dict(result: Any) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    data = {}
    for key in ["sentence", "output", "payload", "request_id", "usage"]:
        if hasattr(result, key):
            data[key] = getattr(result, key)
    return data or {"raw": str(result)}


def _extract_text(data: dict) -> str:
    candidates = [
        data.get("text"),
        data.get("sentence", {}).get("text") if isinstance(data.get("sentence"), dict) else None,
        data.get("output", {}).get("sentence", {}).get("text") if isinstance(data.get("output"), dict) else None,
        data.get("output", {}).get("text") if isinstance(data.get("output"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = data.get("raw")
    return raw if isinstance(raw, str) else ""


def _is_final(data: dict) -> bool:
    sentence = data.get("sentence")
    if isinstance(sentence, dict) and sentence.get("sentence_end") is True:
        return True
    output = data.get("output")
    if isinstance(output, dict):
        sent = output.get("sentence")
        if isinstance(sent, dict) and sent.get("sentence_end") is True:
            return True
        if output.get("is_sentence_end") is True:
            return True
    raw = json.dumps(data, ensure_ascii=False).lower()
    return "sentence_end" in raw and "true" in raw


def _clamp_int(value: str | None, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        parsed = default
    return max(low, min(high, parsed))


def _bool_param(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def translate_events(text: str, source_lang: str, target_lang: str) -> list[dict[str, str]]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALI_OPENAI_BASE_URL")
    if not api_key or not base_url:
        return [{"type": "translation_final", "source": text, "text": ""}]
    if target_lang == "en":
        instruction = "Translate the speech transcript into concise natural English subtitles. Output English only."
    elif target_lang == "zh":
        instruction = "Translate the speech transcript into concise natural Chinese subtitles. Output Chinese only."
    else:
        instruction = f"Translate the speech transcript into {target_lang}. Output only the translation."
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        stream=True,
    )
    chunks = []
    events = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            chunks.append(delta)
            events.append({"type": "translation_partial", "source": text, "delta": delta, "text": "".join(chunks)})
    events.append({"type": "translation_final", "source": text, "text": "".join(chunks).strip()})
    return events


def correct_transcript_text(text: str, context_rows: list[dict[str, str]], source_lang: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALI_OPENAI_BASE_URL")
    if not api_key or not base_url or len(text.strip()) < 3:
        return text
    context = "\n".join(
        [
            f"- {row.get('corrected') or row.get('source') or ''}"
            for row in context_rows[-8:]
            if row.get("corrected") or row.get("source")
        ]
    )
    lang = "中文" if "zh" in source_lang else "英文"
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": (
                f"你是实时会议 ASR 文本校对器。输入是{lang}语音识别结果。"
                "只修正明显的同音错词、断句错误、俚语、缩写、专有名词和上下文不一致的识别错误；"
                "不要扩写、不要总结、不要翻译、不要改变说话人的原意。"
                "如果没有把握，原样输出。只输出修正后的句子。"
            ),
        },
        {
            "role": "user",
            "content": f"前文：\n{context or '无'}\n\n当前 ASR：\n{text}\n\n修正后：",
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=os.getenv("REALTIME_CORRECTION_MODEL", LLM_MODEL),
            messages=messages,
            temperature=0,
        )
        corrected = resp.choices[0].message.content.strip() if resp.choices else text
        corrected = corrected.strip("\"'“”‘’")
        if not corrected or len(corrected) > max(260, len(text) * 2.2):
            return text
        return corrected
    except Exception:
        return text


def generate_realtime_minutes(transcript: list[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALI_OPENAI_BASE_URL")
    if not api_key or not base_url:
        return {
            "stage_summaries": [],
            "speaker_insights": [],
            "final_minutes": {
                "one_sentence_summary": "会议已保存，但缺少 LLM 配置，未生成纪要。",
                "key_decisions": [],
                "action_items": [],
                "risks": [],
                "open_questions": [],
                "conclusion": "",
            },
            "success": False,
            "error": "missing_llm_config",
        }
    normalized = []
    for idx, row in enumerate(transcript):
        normalized.append(
            {
                "time": row.get("time") or row.get("start") or idx,
                "speaker": row.get("speaker") or "参会者识别中",
                "source": row.get("corrected") or row.get("source") or "",
                "translation": row.get("english") or row.get("translation") or "",
            }
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = {
        "duration_seconds": duration_seconds,
        "transcript": normalized,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是会议纪要生成器。根据实时会议转写生成 JSON。"
                "每 120 秒生成一个阶段摘要；最终纪要要包含按发言人归纳、行动项、风险、待确认问题。"
                "如果发言人还未完成声纹识别，用“参会者识别中”或已有临时标签，不要虚构身份。"
                "只输出合法 JSON，不要 markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"输入 JSON：\n{json.dumps(prompt, ensure_ascii=False)}\n\n"
                "输出结构："
                "{stage_summaries:[{window,title,summary,speaker_points,actions,risks,open_questions}],"
                "speaker_insights:[{speaker,stance_or_role,main_points,agreements,disagreements_or_concerns}],"
                "final_minutes:{one_sentence_summary,key_decisions,action_items,risks,open_questions,conclusion}}"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=os.getenv("ASSISTANT_LLM_MODEL", os.getenv("REALTIME_TRANSLATE_MODEL", "qwen-plus")),
            messages=messages,
            temperature=0.15,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content.strip() if resp.choices else "{}"
        data = json.loads(content)
        data["success"] = True
        data["minutes_model"] = os.getenv("ASSISTANT_LLM_MODEL", os.getenv("REALTIME_TRANSLATE_MODEL", "qwen-plus"))
        return data
    except Exception as exc:
        return {
            "stage_summaries": [],
            "speaker_insights": [],
            "final_minutes": {
                "one_sentence_summary": "会议已保存，但纪要生成失败。",
                "key_decisions": [],
                "action_items": [],
                "risks": [],
                "open_questions": [],
                "conclusion": "",
            },
            "success": False,
            "error": str(exc),
        }


def generate_stage_summary(transcript: list[dict[str, Any]], window: str) -> dict[str, Any]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALI_OPENAI_BASE_URL")
    if not api_key or not base_url:
        return {"window": window, "title": "阶段摘要", "summary": "缺少 LLM 配置，未生成阶段摘要。", "actions": [], "risks": [], "open_questions": []}
    client = OpenAI(api_key=api_key, base_url=base_url)
    rows = [
        {
            "speaker": x.get("speaker") or "参会者识别中",
            "source": x.get("corrected") or x.get("source") or "",
            "translation": x.get("english") or "",
            "time": x.get("time") or "",
        }
        for x in transcript[-80:]
    ]
    try:
        resp = client.chat.completions.create(
            model=os.getenv("ASSISTANT_LLM_MODEL", os.getenv("REALTIME_TRANSLATE_MODEL", "qwen-plus")),
            messages=[
                {
                    "role": "system",
                    "content": "你是实时会议阶段摘要生成器。只输出合法 JSON，字段为 window,title,summary,actions,risks,open_questions。",
                },
                {
                    "role": "user",
                    "content": f"时间窗口：{window}\n会议转写：{json.dumps(rows, ensure_ascii=False)}",
                },
            ],
            temperature=0.15,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content.strip())
        data.setdefault("window", window)
        return data
    except Exception as exc:
        return {"window": window, "title": "阶段摘要生成失败", "summary": str(exc), "actions": [], "risks": [], "open_questions": []}


def _should_search(question: str) -> bool:
    q = question.lower()
    keywords = [
        "最新",
        "现在",
        "今天",
        "新闻",
        "联网",
        "搜索",
        "查一下",
        "资料",
        "价格",
        "天气",
        "论文",
        "方案",
        "竞品",
        "案例",
        "趋势",
        "现状",
        "市场",
        "政策",
        "法规",
        "标准",
        "排名",
        "对比",
        "推荐",
        "有没有",
        "是什么",
        "怎么做",
        "哪些",
    ]
    meeting_only = ["刚才", "这场会议", "会议里", "参会者", "发言人", "纪要", "行动项", "风险", "待确认"]
    if any(k in q for k in meeting_only) and not any(k in q for k in ["联网", "查", "最新", "外部", "市场", "竞品", "案例"]):
        return False
    return any(k in q for k in keywords)


def _meeting_context(question: str = "") -> str:
    summary = read_json_file(ROOT / "outputs/web_cache/product_minutes.json", {})
    speaker_minutes = read_json_file(ROOT / "outputs/web_cache/speaker_minutes.json", {})
    cues = read_json_file(ROOT / "outputs/web_cache/streaming_cues.json", [])
    cue_rows = [
        {
            "time": f"{x.get('start', 0)}-{x.get('end', 0)}s",
            "speaker": x.get("speaker", ""),
            "zh": x.get("zh", ""),
            "en": x.get("en", ""),
        }
        for x in cues[:80]
    ]
    current_meeting_context = {
        "product_minutes": summary,
        "speaker_status": speaker_minutes.get("speaker_status"),
        "speaker_summaries": speaker_minutes.get("speakers", [])[:6],
        "transcript": cue_rows,
    }
    try:
        history_context = json.loads(library_meeting_context(question, limit=5))
    except Exception as exc:
        history_context = {"error": str(exc), "retrieved_meetings": []}
    return json.dumps(
        {
            "current_meeting": current_meeting_context,
            "meeting_library": history_context,
        },
        ensure_ascii=False,
    )[:70000]


def _web_search(query: str, limit: int = 4) -> list[dict[str, str]]:
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        resp.raise_for_status()
        html = resp.text
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?>(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.S,
        )
        items = []
        for match in pattern.finditer(html):
            title = re.sub(r"<.*?>", "", match.group("title"))
            snippet = re.sub(r"<.*?>", "", match.group("snippet"))
            url = match.group("url")
            items.append(
                {
                    "title": re.sub(r"\s+", " ", title).strip(),
                    "snippet": re.sub(r"\s+", " ", snippet).strip(),
                    "url": url,
                }
            )
            if len(items) >= limit:
                break
        return items
    except Exception as exc:
        return [{"title": "联网搜索失败", "snippet": str(exc), "url": ""}]


def _transition_text(question: str, use_search: bool) -> str:
    
    if use_search:
        return "收到，我先结合会议内容查一下外部资料，然后给你一个可执行的回答。"
    if any(k in question for k in ["纪要", "总结", "行动项", "风险", "谁", "发言"]):
        return "收到，我正在结合会议字幕和发言人记录整理答案。"
    return "收到，我来分析一下。"


def _tts_audio(text: str) -> tuple[str | None, str | None]:
    
    # --- 读取配置 ---
    api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope_base = os.getenv("ALI_DASHSCOPE_BASE_URL")

    # 没有 API Key 或服务地址，直接返回错误（快速失败，避免无效请求）
    if not api_key or not dashscope_base:
        return None, "缺少 TTS 配置"

    # TTS 模型和音色：优先用 ASSISTANT_TTS_* 专用配置，fallback 到通用配置，最后用默认值
    model = os.getenv("ASSISTANT_TTS_MODEL", os.getenv("TTS_MODEL", "qwen-tts"))
    voice = os.getenv("ASSISTANT_TTS_VOICE", "Cherry")  # Cherry: 阿里云中文女声

    try:
        # --- 调用阿里云 DashScope TTS API ---
        # API 文档: services/aigc/multimodal-generation/generation
        # .rstrip('/') 防止 base URL 末尾多余的斜杠导致路径双斜杠
        resp = requests.post(
            f"{dashscope_base.rstrip('/')}/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {api_key}",  # Bearer Token 认证
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": {
                    "text": text[:1200],    # 截断到 1200 字，防止超长文本导致 API 超时或拒收
                    "voice": voice,          # 音色选择（默认 Cherry 女声）
                    "language_type": "Chinese",  # 显式指定中文，避免模型自动检测出错
                },
            },
            timeout=90,  # 90 秒超时——正常 3-10 秒返回，90 是绝对安全上限
        )

        # HTTP 4xx/5xx → 返回错误信息（[:200] 截断防止日志膨胀）
        if resp.status_code >= 400:
            return None, f"TTS HTTP {resp.status_code}: {resp.text[:200]}"

        # --- 结果解析——兼容两种返回格式 ---
        # 链式 .get() 用 or {} 兜底，中间任一层为 None 都不会抛异常
        payload = resp.json()
        audio = (payload.get("output") or {}).get("audio") or {}

        # 情况 1：阿里云直接返回 base64 编码的音频数据
        audio_b64 = audio.get("data")
        if audio_b64:
            return audio_b64, None

        # 情况 2：阿里云返回一个临时下载 URL，需要再发请求下载原始音频
        audio_url = audio.get("url")
        if audio_url:
            audio_resp = requests.get(audio_url, timeout=90)
            audio_resp.raise_for_status()
            # 下载到的是二进制音频 → b64encode 转 base64 → decode 转字符串
            return base64.b64encode(audio_resp.content).decode("ascii"), None

        # 两种格式都没有 → 返回错误
        return None, "TTS 响应中没有音频"

    except Exception as exc:
        # 兜底：不抛异常。
        # TTS 是辅助功能——LLM 文字答案才是核心，语音只是锦上添花。
        # TTS 挂了应该在 response 里标记 tts_error，但文字答案照常返回。
        return None, str(exc)


@app.post("/api/assistant/chat")
async def assistant_chat(req: AssistantRequest) -> JSONResponse:
    start = time.perf_counter()
    question = req.question.strip()# 1. 用户提问
    if not question:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    if req.search_mode == "force":
        use_search = True
    elif req.search_mode == "off":
        use_search = False
    else:
        use_search = bool(req.enable_search and _should_search(question))
    transition = _transition_text(question, use_search)  # 2. 生成过渡语（如”收到，我来分析一下。”）

    # 3. 异步生成过渡语 TTS——扔到线程池，不阻塞主协程
    #    被 to_thread 包裹的 _tts_audio 使用 requests（同步阻塞 IO），
    #    如果在 async 环境直接调用会卡住整个事件循环，所以用线程池卸出去。
    #    此时过渡语 TTS 和后续的搜索/LLM 推理并发执行。
    transition_audio_task = asyncio.to_thread(_tts_audio, transition)
    search_results = _web_search(question) if use_search else []
    context = _meeting_context(question)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALI_OPENAI_BASE_URL")
    if not api_key or not base_url:
        return JSONResponse({"error": "缺少 LLM 配置"}, status_code=500)
    model = os.getenv("ASSISTANT_LLM_MODEL", os.getenv("REALTIME_TRANSLATE_MODEL", "qwen-plus"))
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会议语音助手。回答要自然、简洁、像语音对话；"
                "优先结合当前会议和历史会议库；用户询问某周、某月、历史会议、相关会议时，要使用 meeting_library；"
                "用户以‘最近有讨论 X 吗’、‘有没有讨论 X’或‘哪些会议讨论了 X’提问时，必须先检查 rag_evidence；"
                "只要 rag_evidence 中存在直接相关内容，应先明确回答‘有’，并给出会议日期、主题与 source，不能回答没有讨论；"
                "当问题包含‘哪个会议’、‘哪场会议’、‘历史会议’、‘之前讨论’时，优先直接回答 rag_evidence 命中的会议，"
                "不要先讨论当前会议是否包含该内容；"
                "如果有搜索结果，必须说明依据来自搜索结果或会议记录；"
                "不要编造会议中不存在的事实。若使用 rag_evidence，回答末尾必须列出实际 source 字段；"
                "对于标注为会议采样的证据，要说明其为部分覆盖，不得概括为整场会议结论。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"会议上下文 JSON：\n{context}\n\n"
                f"联网搜索结果 JSON：\n{json.dumps(search_results, ensure_ascii=False)}\n\n"
                f"用户问题：{question}\n\n"
                "请给出适合语音播报的中文回答，最多 5 点。"
            ),
        },
    ]
    for item in req.conversation[-6:]:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.insert(-1, {"role": item["role"], "content": item["content"][:1000]})
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.25)# 4. LLM 生成正式回答
        answer = resp.choices[0].message.content.strip() if hasattr(resp, "choices") and resp.choices else str(resp)
    except Exception as exc:
        return JSONResponse({"error": f"LLM 调用失败: {exc}"}, status_code=500)

    # 5. 等待过渡语 TTS 完成——await 挂起直到线程池任务返回
    #    如果过渡语 TTS 在 LLM 推理期间已经完成（通常如此），这里几乎不等待。
    transition_audio_b64, transition_tts_error = await transition_audio_task

    # 6. 将 LLM 回答也转为语音——此时没有并发需求，直接同步调用
    audio_b64, tts_error = _tts_audio(answer)
    # 7. 组装响应——包含文字和两段 base64 音频（过渡语 + 回答）
    #    前端收到后用 <audio src="data:audio/wav;base64,..."> 直接播放，无需文件 IO
    return JSONResponse(
        {
            "question": question,
            "transition": transition,                      # 过渡语文字
            "transition_audio_b64": transition_audio_b64,  # 过渡语 base64 音频 → 前端立刻播放
            "transition_tts_error": transition_tts_error,   # 过渡语 TTS 是否失败
            "answer": answer,                               # LLM 回答文字
            "search_used": use_search,
            "search_mode": req.search_mode,
            "search_results": search_results,
            "model": model,
            "tts_model": os.getenv("ASSISTANT_TTS_MODEL", os.getenv("TTS_MODEL", "qwen-tts")),
            "audio_b64": audio_b64,                        # 回答 base64 音频 → 文字展示后播放
            "tts_error": tts_error,                         # 回答 TTS 是否失败
            "latency": round(time.perf_counter() - start, 3),  # 端到端耗时（秒）
        }
    )


@app.post("/api/realtime/meeting/save")
async def save_realtime_meeting(req: RealtimeMeetingSaveRequest) -> JSONResponse:
    if not req.audio_b64:
        return JSONResponse({"error": "缺少录音音频"}, status_code=400)
    start = time.perf_counter()
    corrected_transcript, diarization = await asyncio.to_thread(
        run_local_diarization,
        req.audio_b64,
        req.transcript,
    )
    minutes = await asyncio.to_thread(generate_realtime_minutes, corrected_transcript, req.duration_seconds)
    try:
        metadata = add_realtime_meeting(
            title=req.title,
            audio_b64=req.audio_b64,
            transcript=corrected_transcript,
            product_minutes=minutes,
            duration_seconds=req.duration_seconds,
            route_mode=req.route_mode,
            canonical_route=req.canonical_route,
            route_transcripts=req.route_transcripts,
            comparison_metrics=req.comparison_metrics,
            diarization=diarization,
        )
    except Exception as exc:
        return JSONResponse({"error": f"保存会议失败: {exc}"}, status_code=500)
    return JSONResponse(
        {
            "success": True,
            "meeting": metadata,
            "minutes": minutes,
            "transcript": corrected_transcript,
            "diarization": diarization,
            "latency": round(time.perf_counter() - start, 3),
            "speaker_note": "实时阶段使用临时声纹标签；录音保存后已运行本地会议级声纹校正。",
        }
    )


@app.post("/api/realtime/meeting/diarize")
async def diarize_realtime_meeting(req: RealtimeDiarizationRequest) -> JSONResponse:
    if not req.audio_b64:
        return JSONResponse({"error": "缺少录音音频"}, status_code=400)
    start = time.perf_counter()
    transcript, diarization = await asyncio.to_thread(
        run_local_diarization,
        req.audio_b64,
        req.transcript,
    )
    return JSONResponse(
        {
            "success": bool(diarization.get("success")),
            "transcript": transcript,
            "diarization": diarization,
            "latency": round(time.perf_counter() - start, 3),
        }
    )


@app.post("/api/meetings/{meeting_id}/diarize")
async def diarize_stored_meeting(meeting_id: str, req: StoredMeetingDiarizationRequest) -> JSONResponse:
    """Re-run local diarization for an existing meeting-library audio file."""
    detail = load_meeting_detail(meeting_id)
    metadata = detail.get("metadata") or {}
    audio_path_value = metadata.get("audio_path")
    if not audio_path_value:
        return JSONResponse({"error": "该会议没有可用音频"}, status_code=404)
    audio_path = Path(audio_path_value)
    if not audio_path.is_absolute():
        audio_path = ROOT / audio_path
    if not audio_path.exists():
        return JSONResponse({"error": f"音频文件不存在: {audio_path_value}"}, status_code=404)

    start = time.perf_counter()
    result = await asyncio.to_thread(diarize_audio, str(audio_path), None, req.speaker_count)
    transcript = detail.get("transcript") or []
    corrected = align_transcript_to_turns(transcript, result.get("turns") or [])
    corrected_count = sum(1 for row in corrected if row.get("speaker_source") == "3d-speaker-local")
    result["corrected_transcript_count"] = corrected_count
    result["status"] = "resolved" if corrected_count else "audio_diarized"
    try:
        updated = update_meeting_diarization(meeting_id, corrected, result)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"保存声纹结果失败: {exc}"}, status_code=500)
    return JSONResponse(
        {
            "success": bool(result.get("success")),
            "meeting": updated,
            "transcript": corrected,
            "diarization": result,
            "latency": round(time.perf_counter() - start, 3),
        }
    )


@app.post("/api/realtime/meeting/stage-summary")
async def realtime_stage_summary(req: RealtimeStageSummaryRequest) -> JSONResponse:
    if not req.transcript:
        return JSONResponse({"error": "暂无可总结的转写内容"}, status_code=400)
    summary = await asyncio.to_thread(generate_stage_summary, req.transcript, req.window)
    return JSONResponse({"success": True, "summary": summary})


@app.get("/api/meetings/search")
async def meeting_search(q: str, limit: int = 5) -> JSONResponse:
    if not q.strip():
        return JSONResponse({"error": "query is empty"}, status_code=400)
    try:
        evidence = await asyncio.to_thread(retrieve_rag_evidence, q, max(1, min(limit, 10)))
        return JSONResponse({"success": True, "evidence": evidence})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc), "evidence": []}, status_code=500)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.websocket("/ws/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    route_mode = (ws.query_params.get("route") or "cascade").lower()
    if route_mode not in {"cascade", "e2e", "compare"}:
        await ws.send_text(json.dumps({"type": "error", "message": "route 必须是 cascade、e2e 或 compare"}, ensure_ascii=False))
        await ws.close(code=1008)
        return
    source_lang = ws.query_params.get("source_lang") or "zh"
    target_lang = ws.query_params.get("target_lang") or ("en" if source_lang == "zh" else "zh")
    max_sentence_silence = _clamp_int(
        ws.query_params.get("max_sentence_silence"),
        DEFAULT_MAX_SENTENCE_SILENCE_MS,
        200,
        6000,
    )
    semantic_punctuation = _bool_param(ws.query_params.get("semantic_punctuation"), False)
    reference_meeting_id = (ws.query_params.get("reference_meeting_id") or "").strip()
    meeting_voice_profiles = load_meeting_voice_profiles(reference_meeting_id)
    meeting_reference_turns = load_meeting_reference_turns(reference_meeting_id)
    try:
        reference_offset_seconds = max(0.0, float(ws.query_params.get("reference_offset_seconds") or 0.0))
    except (TypeError, ValueError):
        reference_offset_seconds = 0.0
    language_hints = [x.strip() for x in source_lang.split(",") if x.strip()]
    if not language_hints:
        language_hints = ["zh"]
    speaker_embedding_model = choose_speaker_embedding_model(source_lang)
    events: "queue.Queue[dict[str, Any]]" = queue.Queue()
    asr_started = threading.Event()
    asr_launch_started = False
    recognition: Recognition | None = None
    if route_mode in {"cascade", "compare"}:
        callback = QueueCallback(events, asr_started, route="cascade")
        recognition = Recognition(
            model=ASR_MODEL,
            callback=callback,
            format="pcm",
            sample_rate=SAMPLE_RATE,
            language_hints=language_hints,
            semantic_punctuation_enabled=semantic_punctuation,
            max_sentence_silence=max_sentence_silence,
            multi_threshold_mode_enabled=True,
            heartbeat=True,
        )
    glossary = read_json_file(ROOT / "config" / "glossary.json", {})
    live_translate = LiveTranslateSession(events, source_lang, target_lang, glossary) if route_mode in {"e2e", "compare"} else None

    stop_event = threading.Event()
    pending_asr_frames: list[bytes] = []
    last_final_text = ""
    correction_context: list[dict[str, str]] = []
    speaker_tracker = StreamingSpeakerTracker()
    if meeting_voice_profiles:
        speaker_tracker.seed_clusters(meeting_voice_profiles)
    # This side branch gates only speaker embeddings.  ASR/translation still
    # receive every original PCM frame above, so model input stays synchronized
    # with the audio the user hears.
    streaming_vad = StreamingVoiceGate(
        sample_rate=SAMPLE_RATE,
        turn_gap_seconds=SPEAKER_TURN_GAP_SECONDS,
        fast_mode=int(os.getenv("REALTIME_FAST_VAD_MODE", "3")),
        fsmn_chunk_ms=int(os.getenv("REALTIME_FSMN_VAD_CHUNK_MS", "60")),
    )
    # ``model_active_speaker`` is owned by the audio/state machine.  The
    # websocket sender keeps its own display label and must never overwrite
    # this value while delivering delayed events to the browser.
    model_active_speaker = "参会者识别中"
    speaker_buffer = bytearray()
    # Keep a second, VAD-gated buffer for speaker embeddings.  The raw buffer
    # remains available for diagnostics, while CAM++ must not receive a
    # window that silently mixes two turns or a long room-noise tail.
    speaker_voiced_buffer = bytearray()
    speaker_busy = False
    speaker_audio_revision = 0
    last_inferred_revision = 0
    last_voice_audio_seconds = -1.0
    initial_speaker_announced = bool(meeting_voice_profiles)
    speaker_turn_id = 0
    silent_samples = 0
    # When a new turn begins after a known speaker, remember the first pending
    # turn. Once the embedding evidence is conclusive, the UI can relabel only
    # that unresolved run instead of rewriting earlier speaker cards.
    unresolved_speaker_from_turn: int | None = None
    # A new turn after a resolved speaker must pass strong multi-window
    # evidence before it can reuse an existing label. This prevents a gray
    # similarity score from collapsing a genuinely different participant.
    strict_speaker_turns: set[int] = set()
    # Keep a model-side record of resolved turns so a silence boundary cannot
    # observe a stale display label and retain the previous speaker's tail.
    resolved_speakers_by_turn: dict[int, str] = {}
    # A speaker can change without a clean pause.  Keep a short vote so one
    # mixed embedding cannot split a subtitle card, while two consecutive
    # strong embeddings can retroactively open a new speaker turn.
    pending_change_label: str | None = None
    pending_change_votes = 0
    pending_change_turn_id: int | None = None
    pending_change_last_audio_seconds: float | None = None
    pending_change_first_audio_seconds: float | None = None
    # A live subtitle turn may contain one delayed change-point candidate, but
    # must not manufacture a new participant for every unstable window.
    new_identity_turns: set[int] = set()
    profile_update_turn_id: int | None = None
    profile_update_label: str | None = None
    profile_update_matches = 0
    # Live and completed-turn embeddings share one model instance. The
    # completed-turn task is deliberately low priority and never blocks the
    # audio receiver or the subtitle sender.
    speaker_embed_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task[Any]] = set()
    received_samples = 0
    meeting_pcm = bytearray()
    turn_ranges: list[dict[str, Any]] = [{"turn_id": 0, "start": 0.0, "end": None}]
    calibrated_labels: dict[int, str] = {}
    calibrated_centroids: dict[str, np.ndarray] = {}
    recalibration_task: asyncio.Task[Any] | None = None
    last_recalibration_audio_seconds = 0.0
    reference_current_label: str | None = None

    def append_speaker_frame(frame: bytes) -> None:
        """Keep a bounded raw window for diagnostics and timing checks.

        The embedding model uses ``speaker_voiced_buffer`` below. The raw
        buffer remains separate so VAD gating never changes the ASR or
        translation input timeline.
        """

        nonlocal speaker_audio_revision
        speaker_buffer.extend(frame)
        max_bytes = int(SAMPLE_RATE * DIARIZATION_WINDOW_SECONDS * 2)
        if len(speaker_buffer) > max_bytes:
            del speaker_buffer[:-max_bytes]
        speaker_audio_revision += 1

    def append_speaker_voice_bytes(frame: bytes) -> None:
        """Append only VAD-accepted speech for the embedding branch."""

        raw = align_pcm16_bytes(frame)
        if not raw:
            return
        speaker_voiced_buffer.extend(raw)
        max_bytes = int(SAMPLE_RATE * DIARIZATION_WINDOW_SECONDS * 2)
        if len(speaker_voiced_buffer) > max_bytes:
            del speaker_voiced_buffer[:-max_bytes]

    async def embed_speaker_signal(pcm: np.ndarray) -> tuple[Any, np.ndarray | None]:
        """Serialize the selected speaker model across live/background work."""

        async with speaker_embed_lock:
            embedder = await asyncio.to_thread(get_speaker_embedder, speaker_embedding_model)
            embedding = await asyncio.to_thread(embedder.embed, pcm, SAMPLE_RATE)
        return embedder, embedding

    async def consolidate_completed_turn_profile(
        start_seconds: float,
        end_seconds: float,
        label: str,
    ) -> None:
        """Strengthen one existing profile from a completed long turn.

        This is a background correction, not a new-speaker decision. A
        similarity gate protects the profile when the live label was itself a
        boundary outlier, and the audio is capped so it cannot create a long
        model stall after a verbose turn.
        """

        if meeting_voice_profiles or not label or label == "参会者识别中":
            return
        duration = float(end_seconds) - float(start_seconds)
        if duration < 3.0:
            return
        duration = min(duration, 8.0)
        left = max(0, int((float(end_seconds) - duration) * SAMPLE_RATE))
        right = min(len(meeting_pcm) // 2, int(float(end_seconds) * SAMPLE_RATE))
        if right <= left:
            return
        snapshot = bytes(meeting_pcm[left * 2 : right * 2])
        if len(snapshot) % 2:
            snapshot = snapshot[:-1]
        pcm = np.frombuffer(snapshot, dtype="<i2").astype("float32") / 32768.0
        if len(pcm) < int(SAMPLE_RATE * 2.0):
            return
        if float(np.sqrt(np.mean(pcm * pcm))) < SPEAKER_MIN_RMS:
            return
        # Let the next live window enter the model first when a boundary and
        # an embedding completion happen on the same event-loop tick.
        await asyncio.sleep(0.20)
        pcm = normalize_speaker_window(pcm)
        _embedder, embedding = await embed_speaker_signal(pcm)
        if embedding is not None:
            speaker_tracker.commit_profile_embedding(label, embedding, min_similarity=0.62)

    def speaker_name(index: int) -> str:
        return f"参会者{chr(ord('A') + index)}" if index < 26 else f"参会者{index + 1}"

    def calibrated_speaker_at(local_seconds: float) -> str | None:
        absolute_seconds = local_seconds + reference_offset_seconds
        for row in meeting_reference_turns:
            if float(row["start"]) <= absolute_seconds < float(row["end"]):
                return str(row["speaker"])
        return None

    async def run_recalibration(
        snapshot: bytes,
        offset_seconds: float,
        ranges: list[dict[str, Any]],
        previous_labels: dict[int, str],
        previous_centroids: dict[str, np.ndarray],
        final_pass: bool = False,
    ) -> None:
        """Recluster only audio that has already arrived, then backfill turns."""
        nonlocal calibrated_labels, calibrated_centroids
        try:
            pcm = np.frombuffer(snapshot, dtype="<i2").astype("float32") / 32768.0
            result = await asyncio.to_thread(diarize_audio, pcm, SAMPLE_RATE)
        except Exception:
            return
        if not result.get("success"):
            return
        # Short windows use AHC because eigen-gap spectral clustering is not
        # reliable with too few subsegments. Do not expose that provisional
        # partition as a meeting-level identity correction.
        if result.get("cluster_type") != "spectral":
            return

        local_turns: list[dict[str, Any]] = []
        first_seen: dict[str, float] = {}
        for item in result.get("turns") or []:
            local_id = str(item.get("speaker_id") or item.get("speaker") or "")
            if not local_id:
                continue
            start = float(item.get("start", 0.0)) + offset_seconds
            end = float(item.get("end", start)) + offset_seconds
            if end <= start:
                continue
            local_turns.append({"speaker_id": local_id, "start": start, "end": end})
            first_seen.setdefault(local_id, start)
        local_ids = sorted(first_seen, key=first_seen.get)
        if not local_ids:
            return

        local_centroids: dict[str, np.ndarray] = {}
        for item in result.get("speaker_centroids") or []:
            local_id = str(item.get("speaker_id") or "")
            vector = np.asarray(item.get("embedding") or [], dtype="float32").reshape(-1)
            norm = float(np.linalg.norm(vector))
            if local_id and norm:
                local_centroids[local_id] = vector / norm

        # Match rolling-window clusters to stable meeting centroids first.
        # Local cluster ids are allowed to change between recalibrations.
        mapped: dict[str, str] = {}
        used: set[str] = set()
        if final_pass:
            # The final full-meeting pass is the identity authority.  AMI and
            # unregistered microphone meetings have no external names, so use
            # deterministic first-appearance labels instead of preserving a
            # transient online outlier such as participant G.
            mapped = {local_id: speaker_name(index) for index, local_id in enumerate(local_ids)}
            used.update(mapped.values())
        else:
            centroid_candidates = sorted(
                [
                    (
                        float(np.dot(local_centroids[local_id], reference)),
                        local_id,
                        label,
                    )
                    for local_id in local_ids
                    if local_id in local_centroids
                    for label, reference in previous_centroids.items()
                    if reference.shape == local_centroids[local_id].shape
                ],
                reverse=True,
            )
            for score, local_id, label in centroid_candidates:
                if score < 0.55:
                    break
                if local_id not in mapped and label not in used:
                    mapped[local_id] = label
                    used.add(label)

        # Fall back to temporal overlap when a centroid is not available.
        scores: dict[str, dict[str, float]] = {local_id: {} for local_id in local_ids}
        for local_turn in local_turns:
            for row in ranges:
                prior = previous_labels.get(int(row["turn_id"]))
                if not prior:
                    continue
                overlap = max(
                    0.0,
                    min(local_turn["end"], float(row["end"])) - max(local_turn["start"], float(row["start"])),
                )
                if overlap:
                    scores[local_turn["speaker_id"]][prior] = (
                        scores[local_turn["speaker_id"]].get(prior, 0.0) + overlap
                    )
        candidates = sorted(
            [
                (score, local_id, label)
                for local_id, values in scores.items()
                for label, score in values.items()
            ],
            reverse=True,
        )
        for _score, local_id, label in candidates:
            if local_id not in mapped and label not in used:
                mapped[local_id] = label
                used.add(label)
        known_labels = set(previous_labels.values()) | set(previous_centroids)
        next_index = 0
        for local_id in local_ids:
            if local_id in mapped:
                continue
            while speaker_name(next_index) in known_labels or speaker_name(next_index) in used:
                next_index += 1
            mapped[local_id] = speaker_name(next_index)
            used.add(mapped[local_id])
            next_index += 1

        for local_id, label in mapped.items():
            vector = local_centroids.get(local_id)
            if vector is None:
                continue
            previous = calibrated_centroids.get(label)
            if previous is None or previous.shape != vector.shape:
                calibrated_centroids[label] = vector.copy()
            else:
                updated = previous * 0.8 + vector * 0.2
                calibrated_centroids[label] = updated / (np.linalg.norm(updated) + 1e-8)

        for row in ranges:
            start = float(row["start"])
            end = float(row["end"])
            overlaps = [
                (
                    max(0.0, min(end, local_turn["end"]) - max(start, local_turn["start"])),
                    local_turn,
                )
                for local_turn in local_turns
            ]
            overlap, best = max(overlaps, key=lambda pair: pair[0], default=(0.0, None))
            if best is None or overlap <= 0:
                continue
            label = mapped[best["speaker_id"]]
            turn_id = int(row["turn_id"])
            calibrated_labels[turn_id] = label
            confidence = round(overlap / max(0.25, end - start), 3)
            events.put(
                {
                    "type": "speaker_update",
                    "turn_id": turn_id,
                    "speaker": label,
                    "status": "reconciled",
                    "confidence": confidence,
                    "backend": result.get("backend", "3d-speaker-local"),
                }
            )

    def queue_recalibration(force: bool = False) -> None:
        nonlocal recalibration_task, last_recalibration_audio_seconds
        if meeting_reference_turns:
            return
        duration = len(meeting_pcm) / (SAMPLE_RATE * 2)
        if duration < REALTIME_RECALIBRATION_MIN_SECONDS:
            return
        if len(turn_ranges) < 2 and not force:
            return
        if not force and duration - last_recalibration_audio_seconds < REALTIME_RECALIBRATION_INTERVAL_SECONDS:
            return
        if force and calibrated_labels and duration - last_recalibration_audio_seconds < 2.0:
            return
        if recalibration_task is not None and not recalibration_task.done():
            return
        offset = (
            max(0.0, duration - REALTIME_FINAL_RECALIBRATION_MAX_SECONDS)
            if force
            else max(0.0, duration - REALTIME_RECALIBRATION_LOOKBACK_SECONDS)
        )
        start_byte = int(offset * SAMPLE_RATE * 2)
        snapshot = bytes(meeting_pcm[start_byte:])
        ranges = []
        for row in turn_ranges:
            end = float(row["end"] if row["end"] is not None else duration)
            start = float(row["start"])
            if end <= offset or start >= duration:
                continue
            ranges.append(
                {
                    "turn_id": int(row["turn_id"]),
                    "start": max(offset, start),
                    "end": min(duration, end),
                }
            )
        if not ranges:
            return
        last_recalibration_audio_seconds = duration
        recalibration_task = asyncio.create_task(
            run_recalibration(
                snapshot,
                offset,
                ranges,
                dict(calibrated_labels),
                {label: vector.copy() for label, vector in calibrated_centroids.items()},
                final_pass=force,
            )
        )
        background_tasks.add(recalibration_task)
        recalibration_task.add_done_callback(background_tasks.discard)

    def start_asr() -> None:
        if recognition is None:
            return
        try:
            recognition.start()
        except Exception as exc:
            events.put({"type": "error", "message": f"ASR start failed: {exc}"})

    if live_translate is not None:
        live_translate.start()

    async def sender() -> None:
        nonlocal last_final_text
        display_speaker = "参会者识别中"
        display_turn_id = 0
        while not stop_event.is_set():
            try:
                event = await asyncio.to_thread(events.get, True, 0.2)
            except queue.Empty:
                continue
            if event.get("type") == "speaker_turn_start":
                display_turn_id = int(event.get("turn_id", display_turn_id))
                display_speaker = str(event.get("speaker") or "参会者识别中")
            if event.get("type") == "speaker_update" and event.get("diagnostic_only"):
                event["applies_to_active_turn"] = False
                await ws.send_text(json.dumps(event, ensure_ascii=False))
                continue
            if event.get("type") == "speaker_update":
                event_turn = int(event.get("turn_id", display_turn_id))
                if event.get("status") == "reconciled":
                    event["applies_to_active_turn"] = event_turn == display_turn_id
                    if event["applies_to_active_turn"] and event.get("speaker"):
                        display_speaker = str(event["speaker"])
                    await ws.send_text(json.dumps(event, ensure_ascii=False))
                    continue
                # A candidate can span several short pause-delimited subtitle
                # turns. Accept evidence from that unresolved run, but never a
                # delayed result belonging to the preceding known speaker.
                applies_to_pending_run = (
                    display_speaker == "参会者识别中"
                    and unresolved_speaker_from_turn is not None
                    and event_turn >= unresolved_speaker_from_turn
                )
                event_applies = event_turn == display_turn_id or applies_to_pending_run
                event["applies_to_active_turn"] = event_applies
                if event_applies and event.get("speaker"):
                    display_speaker = str(event["speaker"])
            if event.get("type") == "asr_partial" and event.get("route") == "cascade":
                event["speaker"] = display_speaker
                event["turn_id"] = display_turn_id
            if event.get("type") == "asr_final" and event.get("route") == "cascade":
                event["speaker"] = display_speaker
                event["turn_id"] = display_turn_id
            if str(event.get("type", "")).startswith("e2e_"):
                # The provider emits transcript and translation independently;
                # attach the shared online speaker decision to both card types.
                event["speaker"] = display_speaker
                event["turn_id"] = display_turn_id
            if route_mode == "compare" and str(event.get("type", "")).startswith("e2e_"):
                # A compare session has one shared audio/speaker pipeline. Mark
                # provider events so the UI can render the two result lanes.
                event["route"] = "e2e"
            await ws.send_text(json.dumps(event, ensure_ascii=False))
            if event["type"] == "asr_final" and event.get("route") == "cascade" and event.get("text") and event["text"] != last_final_text:
                raw_text = event["text"]
                sentence_speaker = event.get("speaker") or display_speaker
                sentence_turn_id = int(event.get("turn_id", display_turn_id))
                last_final_text = raw_text
                await ws.send_text(json.dumps({"type": "correction_start", "source": raw_text}, ensure_ascii=False))
                corrected_text = await asyncio.to_thread(correct_transcript_text, raw_text, correction_context, source_lang)
                correction_context.append({"source": raw_text, "corrected": corrected_text})
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "correction_final",
                            "source": raw_text,
                            "corrected": corrected_text,
                            "changed": corrected_text != raw_text,
                            "speaker": sentence_speaker,
                            "turn_id": sentence_turn_id,
                        },
                        ensure_ascii=False,
                    )
                )
                await ws.send_text(json.dumps({"type": "translation_start", "source": corrected_text, "raw_source": raw_text}, ensure_ascii=False))
                try:
                    translation_events = await asyncio.to_thread(translate_events, corrected_text, source_lang, target_lang)
                    for translation_event in translation_events:
                        translation_event["raw_source"] = raw_text
                        translation_event["corrected_source"] = corrected_text
                        translation_event["speaker"] = sentence_speaker
                        translation_event["turn_id"] = sentence_turn_id
                        translation_event["route"] = "cascade"
                        await ws.send_text(json.dumps(translation_event, ensure_ascii=False))
                except Exception as exc:
                    await ws.send_text(json.dumps({"type": "error", "message": f"translation failed: {exc}"}, ensure_ascii=False))

    sender_task = asyncio.create_task(sender())
    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "sample_rate": SAMPLE_RATE,
                    "asr_model": ASR_MODEL,
                    "route_mode": route_mode,
                    "e2e_model": "qwen3.5-livetranslate-flash-realtime" if live_translate is not None else None,
                    "speaker_profile": (
                        "meeting_calibrated_timeline"
                        if meeting_reference_turns
                        else "meeting_calibrated_profile"
                        if meeting_voice_profiles
                        else "online_unknown_meeting"
                    ),
                    "speaker_vad": streaming_vad.backend,
                    "speaker_vad_fsmn": streaming_vad.fsmn_available,
                    "speaker_embedding_model": speaker_embedding_model,
                    "reference_meeting_id": reference_meeting_id or None,
                    "reference_offset_seconds": reference_offset_seconds if meeting_reference_turns else None,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "max_sentence_silence": max_sentence_silence,
                    "semantic_punctuation": semantic_punctuation,
                },
                ensure_ascii=False,
            )
        )
        def reset_change_vote() -> None:
            nonlocal pending_change_label, pending_change_votes, pending_change_turn_id
            nonlocal pending_change_last_audio_seconds, pending_change_first_audio_seconds
            pending_change_label = None
            pending_change_votes = 0
            pending_change_turn_id = None
            pending_change_last_audio_seconds = None
            pending_change_first_audio_seconds = None

        def maybe_split_inferred_turn(
            label: str,
            result: dict[str, Any],
            turn_id: int,
            audio_end_seconds: float,
        ) -> bool:
            """Open a new card when repeated embeddings confirm a speaker change."""

            nonlocal model_active_speaker, speaker_turn_id, unresolved_speaker_from_turn
            nonlocal pending_change_label, pending_change_votes, pending_change_turn_id
            nonlocal pending_change_last_audio_seconds, pending_change_first_audio_seconds
            nonlocal profile_update_turn_id, profile_update_label, profile_update_matches
            if turn_id != speaker_turn_id or not label or label == "参会者识别中":
                return False
            # Only the model-side label participates in change detection. The
            # sender may be delivering an older speaker_update at the same
            # time, so using its display label here creates a race.
            active_label = model_active_speaker
            if active_label == "参会者识别中":
                active_label = resolved_speakers_by_turn.get(turn_id, active_label)
            if not active_label or active_label == "参会者识别中":
                return False
            if label == active_label:
                # Returning to the active speaker is strong evidence that a
                # prior outlier was only a mixed/prosodic window. Requiring a
                # fresh run prevents one high-scoring anomaly from opening a
                # cascade of false subtitle cards.
                reset_change_vote()
                return False

            status = str(result.get("status") or "")
            confidence = float(result.get("confidence") or 0.0)
            if status == "new":
                strong = True
                required_votes = 3
            elif status == "matched" and confidence >= 0.80:
                strong = True
                required_votes = 3
            elif status == "matched_soft" and confidence >= 0.72:
                # Soft matches are deliberately weaker. Require four votes
                # before a prosody change can split the visible subtitle card.
                strong = True
                required_votes = 4
            else:
                strong = False
                required_votes = 2
            if not strong:
                reset_change_vote()
                return False
            if (
                pending_change_label == label
                and pending_change_turn_id == turn_id
                and pending_change_last_audio_seconds is not None
                and audio_end_seconds - pending_change_last_audio_seconds <= DIARIZATION_WINDOW_SECONDS * 1.5
            ):
                pending_change_votes += 1
            else:
                pending_change_label = label
                pending_change_votes = 1
                pending_change_turn_id = turn_id
                pending_change_first_audio_seconds = audio_end_seconds
            pending_change_last_audio_seconds = audio_end_seconds
            if (
                pending_change_votes < required_votes
                or pending_change_first_audio_seconds is None
                or audio_end_seconds - pending_change_first_audio_seconds
                < SPEAKER_CHANGE_MIN_DURATION_SECONDS
            ):
                return False

            current_start = float(turn_ranges[-1]["start"])
            # The embedding window ends at ``audio_end_seconds`` but contains
            # up to 2.5 s of speech.  Split near its midpoint rather than
            # assigning the whole mixed window to the later speaker.
            boundary = max(current_start + 0.45, audio_end_seconds - DIARIZATION_WINDOW_SECONDS / 2.0)
            if boundary <= current_start or boundary >= audio_end_seconds:
                reset_change_vote()
                return False
            previous_turn_id = speaker_turn_id
            # The caller may have tentatively associated this embedding with
            # the old turn before the change vote completed.
            resolved_speakers_by_turn[previous_turn_id] = active_label
            turn_ranges[-1]["end"] = boundary
            speaker_turn_id += 1
            new_turn_id = speaker_turn_id
            turn_ranges.append({"turn_id": new_turn_id, "start": boundary, "end": None})
            strict_speaker_turns.add(new_turn_id)
            new_identity_turns.discard(new_turn_id)
            unresolved_speaker_from_turn = None
            resolved_speakers_by_turn[new_turn_id] = label
            model_active_speaker = label
            # Do not carry the mixed change-point window into the new identity.
            # The next causal window will be built from the new speaker only.
            speaker_buffer.clear()
            speaker_tracker.reset_candidate()
            profile_update_turn_id = None
            profile_update_label = None
            profile_update_matches = 0
            reset_change_vote()
            events.put(
                {
                    "type": "speaker_turn_start",
                    "turn_id": new_turn_id,
                    "speaker": "参会者识别中",
                    "status": "inferred_change",
                    "audio_start_seconds": round(boundary, 3),
                    "previous_audio_end_seconds": round(boundary, 3),
                    "previous_turn_id": previous_turn_id,
                }
            )
            events.put(
                {
                    "type": "speaker_update",
                    "turn_id": new_turn_id,
                    "speaker": label,
                    "status": "matched_turn",
                    "confidence": round(confidence, 3),
                    "backend": result.get("backend", "funasr-campplus"),
                    "reliable": True,
                    "inferred_change": True,
                }
            )
            return True

        def schedule_speaker_inference(
            allow_short_window: bool = False,
            allow_stale_baseline: bool = False,
        ) -> None:
            nonlocal speaker_busy, last_inferred_revision
            if speaker_busy or speaker_audio_revision <= last_inferred_revision:
                return
            # Raw PCM is retained across short pauses, but a window that is
            # already beyond the latest VAD-confirmed speech is not useful for
            # a causal identity decision. This prevents a trailing silence
            # window from becoming a new participant after the speaker stops.
            if (
                last_voice_audio_seconds < 0
                or received_samples / SAMPLE_RATE - last_voice_audio_seconds
                > max(0.20, SPEAKER_TURN_GAP_SECONDS * 0.5)
            ):
                return
            default_seconds = (
                DIARIZATION_INITIAL_WINDOW_SECONDS
                if not speaker_tracker.clusters
                else DIARIZATION_WINDOW_SECONDS
            )
            min_seconds = (
                DIARIZATION_INITIAL_MIN_WINDOW_SECONDS
                if not speaker_tracker.clusters
                else DIARIZATION_MIN_WINDOW_SECONDS
            )
            required_seconds = default_seconds
            available_voice_seconds = len(speaker_voiced_buffer) / (SAMPLE_RATE * 2)
            if allow_short_window:
                current_turn_seconds = max(
                    0.0,
                    received_samples / SAMPLE_RATE - float(turn_ranges[-1]["start"]),
                )
                required_seconds = min(default_seconds, current_turn_seconds, available_voice_seconds)
            elif available_voice_seconds < required_seconds:
                return
            if required_seconds < min_seconds:
                return
            required_bytes = int(SAMPLE_RATE * required_seconds * 2)
            if len(speaker_voiced_buffer) < required_bytes:
                return
            speaker_busy = True
            last_inferred_revision = speaker_audio_revision
            task = asyncio.create_task(
                infer_speaker(
                    bytes(speaker_voiced_buffer[-required_bytes:]),
                    speaker_turn_id,
                    last_inferred_revision,
                    received_samples / SAMPLE_RATE,
                    allow_stale_baseline,
                )
            )
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        async def infer_speaker(
            window: bytes,
            turn_id: int,
            revision: int,
            audio_end_seconds: float,
            allow_stale_baseline: bool = False,
        ) -> None:
            nonlocal speaker_busy, unresolved_speaker_from_turn, model_active_speaker
            nonlocal profile_update_turn_id, profile_update_label, profile_update_matches
            try:
                window = align_pcm16_bytes(window)
                pcm = np.frombuffer(window, dtype="<i2").astype("float32") / 32768.0
                if len(pcm) < int(SAMPLE_RATE * 0.8) or float(np.sqrt(np.mean(pcm * pcm))) < SPEAKER_MIN_RMS:
                    return
                pcm = normalize_speaker_window(pcm)
                embedder, embedding = await embed_speaker_signal(pcm)
                # A boundary can be detected while CAM++ is running in its
                # worker thread. The only stale result that is useful is the
                # first completed baseline: it can seed the empty meeting
                # profile, but it must never become the current turn's label.
                if turn_id != speaker_turn_id:
                    if speaker_tracker.clusters or not (allow_stale_baseline or turn_id < speaker_turn_id):
                        return
                    baseline = speaker_tracker.assign_embedding(
                        embedding,
                        allow_soft_match=True,
                        allow_new_speaker=True,
                        update_profile=False,
                    )
                    baseline_label = baseline.get("speaker")
                    if baseline.get("status") == "new" and baseline_label:
                        resolved_speakers_by_turn[turn_id] = str(baseline_label)
                        events.put(
                            {
                                "type": "speaker_update",
                                "turn_id": turn_id,
                                "speaker": str(baseline_label),
                                "status": "baseline_backfill",
                                "confidence": float(baseline.get("confidence") or 0.0),
                                "backend": embedder.name,
                                "reliable": embedder.name != "acoustic-fallback",
                                "applies_to_active_turn": False,
                            }
                        )
                    return
                if (
                    last_voice_audio_seconds < 0
                    or audio_end_seconds - last_voice_audio_seconds
                    > max(0.20, SPEAKER_TURN_GAP_SECONDS * 0.5)
                ):
                    return
                active_before = model_active_speaker
                result = speaker_tracker.assign_embedding(
                    embedding,
                    allow_soft_match=turn_id not in strict_speaker_turns,
                    # A turn may introduce at most one identity. Once the
                    # turn already has a confirmed label, a later outlier is
                    # kept pending instead of becoming C/D/... mid-sentence.
                    allow_new_speaker=(
                        turn_id not in new_identity_turns
                        and turn_id not in resolved_speakers_by_turn
                    ),
                    update_profile=False,
                )
                resolved_label = result.get("speaker")
                if result.get("status") == "new":
                    new_identity_turns.add(turn_id)
                if resolved_label and resolved_label != "参会者识别中":
                    label = str(resolved_label)
                    if model_active_speaker == "参会者识别中":
                        # The first confirmed voiced window establishes the
                        # baseline participant without waiting for a second
                        # person or a future recalibration pass.
                        model_active_speaker = label
                        resolved_speakers_by_turn[turn_id] = label
                    elif label == model_active_speaker:
                        resolved_speakers_by_turn[turn_id] = model_active_speaker
                    else:
                        split_opened = maybe_split_inferred_turn(
                            label,
                            result,
                            turn_id,
                            audio_end_seconds,
                        )
                        if not split_opened:
                            # This is evidence for a possible change, not a
                            # stable label for the old card. Keep it visible
                            # only as a diagnostic candidate until the change
                            # vote opens a new turn.
                            result["candidate_speaker"] = label
                            result["speaker"] = "参会者识别中"
                            result["status"] = "candidate_change"
                result_label = result.get("speaker")
                result_confidence = float(result.get("confidence") or 0.0)
                stable_match = (
                    active_before != "参会者识别中"
                    and result_label == active_before
                    and result.get("status") in {"matched", "matched_soft"}
                    and result_confidence >= SPEAKER_PROFILE_UPDATE_MIN_CONFIDENCE
                )
                if stable_match:
                    if profile_update_turn_id == turn_id and profile_update_label == active_before:
                        profile_update_matches += 1
                    else:
                        profile_update_turn_id = turn_id
                        profile_update_label = str(active_before)
                        profile_update_matches = 1
                    if profile_update_matches >= SPEAKER_PROFILE_UPDATE_MIN_MATCHES:
                        speaker_tracker.commit_profile_embedding(str(active_before), embedding)
                        # Keep the evidence windowed: a later change-point
                        # must earn a fresh pair of stable matches.
                        profile_update_matches = 0
                else:
                    profile_update_turn_id = None
                    profile_update_label = None
                    profile_update_matches = 0
                if (
                    result.get("speaker") != "参会者识别中"
                    and unresolved_speaker_from_turn is not None
                    and turn_id >= unresolved_speaker_from_turn
                ):
                    result["backfill_from_turn"] = unresolved_speaker_from_turn
                    unresolved_speaker_from_turn = None
                result.update(
                    {
                        "type": "speaker_update",
                        "turn_id": turn_id,
                        "backend": embedder.name,
                        "reliable": embedder.name != "acoustic-fallback",
                    }
                )
                if meeting_reference_turns:
                    result["diagnostic_only"] = True
                events.put(result)
            except Exception as exc:
                events.put(
                    {
                        "type": "speaker_update",
                        "turn_id": turn_id,
                        "speaker": "参会者识别中",
                        "status": "pending",
                        "message": (
                            f"{exc} (window_bytes={len(window)}, "
                            f"speaker_buffer_bytes={len(speaker_buffer)}, "
                            f"speaker_voiced_buffer_bytes={len(speaker_voiced_buffer)})"
                        )[:240],
                    }
                )
            finally:
                speaker_busy = False
                # Audio can arrive while the embedding model processes the
                # prior window. Consume that latest window immediately instead
                # of waiting for a further microphone callback.
                if not stop_event.is_set() and speaker_audio_revision > revision:
                    schedule_speaker_inference()

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                frame = align_pcm16_bytes(message["bytes"])
                if len(frame) < 2:
                    # Ignore an incomplete PCM sample rather than allowing it
                    # to desynchronise all later int16 windows.
                    continue
                if recognition is not None and not asr_launch_started:
                    asr_launch_started = True
                    threading.Thread(target=start_asr, daemon=True).start()
                pcm_frame = np.frombuffer(frame, dtype="<i2").astype("float32") / 32768.0
                frame_start = received_samples / SAMPLE_RATE
                received_samples += len(pcm_frame)
                meeting_pcm.extend(frame)
                if recognition is not None and not asr_started.is_set():
                    # Preserve the first spoken syllable while the cloud stream
                    # performs its handshake. Keep the buffer bounded at 3 s.
                    pending_asr_frames.append(frame)
                    pending_bytes = sum(len(item) for item in pending_asr_frames)
                    while pending_asr_frames and pending_bytes > SAMPLE_RATE * 2 * 3:
                        pending_bytes -= len(pending_asr_frames.pop(0))
                elif recognition is not None:
                    try:
                        for pending in pending_asr_frames:
                            recognition.send_audio_frame(pending)
                        pending_asr_frames.clear()
                        recognition.send_audio_frame(frame)
                    except Exception as exc:
                        # A provider-side disconnect is recoverable at the UI
                        # layer; never crash the WebSocket handler.
                        events.put({"type": "error", "message": f"ASR audio frame rejected: {exc}"})
                if live_translate is not None:
                    live_translate.send_audio_frame(frame)
                # VAD is deliberately a side branch.  The raw frame has
                # already been sent to the provider routes; only the speaker
                # embedding branch uses the VAD state. Keep the speaker window
                # on the original time axis so CAM++ sees natural silence and
                # phonetic context instead of a compressed voiced-only stream.
                vad_result = await asyncio.to_thread(streaming_vad.feed, pcm_frame)
                vad_voice_bytes = vad_result.voiced_bytes
                if vad_result.active or vad_voice_bytes:
                    last_voice_audio_seconds = (frame_start + len(pcm_frame) / SAMPLE_RATE)
                if not vad_result.active:
                    append_speaker_frame(frame)
                    append_speaker_voice_bytes(vad_voice_bytes)
                    silent_samples += len(pcm_frame)
                    # A final voiced sub-frame can arrive in the same packet
                    # that closes the gate; let the current window consume it.
                    if vad_voice_bytes:
                        # A short utterance may end before the normal 2.5 s
                        # window is available. Use the current turn's
                        # timestamp-contiguous audio, but never embed a tiny
                        # onset fragment.
                        schedule_speaker_inference(allow_short_window=True)
                    continue
                reference_label = calibrated_speaker_at(frame_start) if meeting_reference_turns else None
                reference_boundary = bool(
                    reference_label
                    and reference_current_label
                    and reference_label != reference_current_label
                )
                if reference_label and reference_current_label is None:
                    reference_current_label = reference_label
                if silent_samples >= int(SAMPLE_RATE * SPEAKER_TURN_GAP_SECONDS) or reference_boundary:
                    # A pause opens a new subtitle turn immediately. When the
                    # prior speaker is known, start a clean evidence window so
                    # a new participant cannot be diluted by the prior tail.
                    # If the current turn is still unresolved, retain its
                    # evidence across short internal pauses for confirmation.
                    previous_end = (
                        frame_start
                        if reference_boundary and silent_samples < int(SAMPLE_RATE * SPEAKER_TURN_GAP_SECONDS)
                        else max(
                            float(turn_ranges[-1]["start"]),
                            frame_start - silent_samples / SAMPLE_RATE,
                        )
                    )
                    previous_start = float(turn_ranges[-1]["start"])
                    previous_turn_id = speaker_turn_id
                    previous_label = resolved_speakers_by_turn.get(previous_turn_id)
                    # The first turn can end before the normal 2.2 s window.
                    # Start its short baseline job before advancing the turn
                    # id, then let the stale-baseline branch safely seed the
                    # empty tracker if the next speaker begins first.
                    if (
                        not meeting_reference_turns
                        and not speaker_tracker.clusters
                        and not speaker_busy
                        and len(speaker_voiced_buffer)
                        >= int(SAMPLE_RATE * DIARIZATION_INITIAL_MIN_WINDOW_SECONDS * 2)
                    ):
                        schedule_speaker_inference(
                            allow_short_window=True,
                            allow_stale_baseline=True,
                        )
                    if (
                        not meeting_reference_turns
                        and previous_label
                        and previous_end - previous_start >= 3.0
                    ):
                        profile_task = asyncio.create_task(
                            consolidate_completed_turn_profile(
                                previous_start,
                                previous_end,
                                previous_label,
                            )
                        )
                        background_tasks.add(profile_task)
                        profile_task.add_done_callback(background_tasks.discard)
                    turn_ranges[-1]["end"] = previous_end
                    speaker_turn_id += 1
                    turn_ranges.append({"turn_id": speaker_turn_id, "start": frame_start, "end": None})
                    # A natural VAD boundary starts a clean evidence run; a
                    # pending candidate must never leak across turns.  The
                    # cluster profiles remain global, so a known speaker can
                    # still be matched immediately after a short pause, but
                    # an unresolved onset from the previous turn cannot pull
                    # the next speaker toward the previous identity.
                    reset_change_vote()
                    profile_update_turn_id = None
                    profile_update_label = None
                    profile_update_matches = 0
                    speaker_tracker.reset_candidate()
                    speaker_voiced_buffer.clear()
                    was_resolved = (
                        model_active_speaker != "参会者识别中"
                        or previous_turn_id in resolved_speakers_by_turn
                        or reference_current_label is not None
                    )
                    long_pause = silent_samples >= int(
                        SAMPLE_RATE * max(1.2, SPEAKER_TURN_GAP_SECONDS * 2.0)
                    )
                    if was_resolved:
                        strict_speaker_turns.add(speaker_turn_id)
                    if was_resolved and unresolved_speaker_from_turn is None:
                        unresolved_speaker_from_turn = speaker_turn_id
                    if was_resolved or long_pause or meeting_reference_turns:
                        speaker_buffer.clear()
                    calibrated_label = reference_label if meeting_reference_turns else None
                    model_active_speaker = calibrated_label or "参会者识别中"
                    if calibrated_label:
                        reference_current_label = calibrated_label
                        resolved_speakers_by_turn[speaker_turn_id] = calibrated_label
                        initial_speaker_announced = True
                    events.put(
                        {
                            "type": "speaker_turn_start",
                            "turn_id": speaker_turn_id,
                            "speaker": model_active_speaker,
                            "status": "calibrated" if calibrated_label else "pending",
                            "audio_start_seconds": round(frame_start, 3),
                            "previous_audio_end_seconds": round(previous_end, 3),
                        }
                    )
                    if calibrated_label:
                        events.put(
                            {
                                "type": "speaker_update",
                                "turn_id": speaker_turn_id,
                                "speaker": calibrated_label,
                                "status": "calibrated",
                                "confidence": 1.0,
                                "backend": "meeting-reference",
                                "reliable": True,
                            }
                        )
                    queue_recalibration()
                # Append after the boundary decision. Otherwise the clear
                # above could erase the onset of the new speaker. The frame is
                # intentionally raw and timestamp-contiguous; ``vad_voice_bytes``
                # is only a gate signal, not the embedding waveform.
                append_speaker_frame(frame)
                append_speaker_voice_bytes(vad_voice_bytes)
                silent_samples = 0
                initial_bytes = int(SAMPLE_RATE * DIARIZATION_INITIAL_WINDOW_SECONDS * 2)
                if reference_label and resolved_speakers_by_turn.get(speaker_turn_id) != reference_label:
                    reference_current_label = reference_label
                    resolved_speakers_by_turn[speaker_turn_id] = reference_label
                    model_active_speaker = reference_label
                    initial_speaker_announced = True
                    events.put(
                        {
                            "type": "speaker_update",
                            "turn_id": speaker_turn_id,
                            "speaker": reference_label,
                            "status": "calibrated",
                            "confidence": 1.0,
                            "backend": "meeting-reference",
                            "reliable": True,
                        }
                    )
                if not initial_speaker_announced and len(speaker_voiced_buffer) >= int(SAMPLE_RATE * 0.45 * 2):
                    probe = np.frombuffer(speaker_voiced_buffer[-int(SAMPLE_RATE * 0.45 * 2) :], dtype="<i2").astype("float32") / 32768.0
                    if len(probe):
                        # Make the first active voice A before ASR produces its
                        # first card. CAM++ will confirm the baseline shortly.
                        initial_speaker_announced = True
                        model_active_speaker = "参会者A"
                        resolved_speakers_by_turn[speaker_turn_id] = "参会者A"
                        events.put(
                            {
                                "type": "speaker_update",
                                "turn_id": speaker_turn_id,
                                "speaker": "参会者A",
                                "status": "provisional",
                                "confidence": 0.0,
                            }
                        )
                schedule_speaker_inference()
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("type") == "stop":
                    if received_samples <= 0:
                        break
                    if live_translate is not None:
                        live_translate.stop()
                        # LiveTranslate flushes its final VAD segment only after
                        # session.finish. Keep the sender alive for that response.
                        await asyncio.to_thread(live_translate.closed.wait, 25)
                    turn_ranges[-1]["end"] = received_samples / SAMPLE_RATE
                    if recalibration_task is not None and not recalibration_task.done():
                        try:
                            await recalibration_task
                        except Exception:
                            pass
                    queue_recalibration(force=True)
                    if recalibration_task is not None and not recalibration_task.done():
                        try:
                            await recalibration_task
                        except Exception:
                            pass
                    # The final full-meeting pass can enqueue corrections for
                    # old turns. Keep the sender alive until those events have
                    # reached the browser; cancelling it here loses the tail
                    # correction and makes the UI look less accurate than the
                    # saved meeting record.
                    drain_deadline = asyncio.get_running_loop().time() + 5.0
                    while not events.empty() and asyncio.get_running_loop().time() < drain_deadline:
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.10)
                    break
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        try:
            if recognition is not None and received_samples > 0:
                recognition.stop()
        except Exception:
            pass
        if live_translate is not None:
            live_translate.stop()
        for task in background_tasks:
            task.cancel()
        sender_task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("realtime.server:app", host="127.0.0.1", port=8765, reload=False)
