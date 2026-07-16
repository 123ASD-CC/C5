from __future__ import annotations

import json
import re
import shutil
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ROOT = ROOT / "outputs" / "meetings"
INDEX_PATH = LIBRARY_ROOT / "meeting_index.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return value[:80] or "meeting"


def _today() -> datetime:
    return datetime(2026, 7, 4, 10, 0, 0)


def _tag_pool(text: str) -> list[str]:
    tags = []
    candidates = {
        "产品设计": ["产品", "设计", "遥控器", "用户"],
        "用户体验": ["易用", "用户", "老年", "直观", "按钮"],
        "成本风险": ["成本", "利润", "风险", "价格"],
        "行动项": ["行动", "待确认", "问题", "决策"],
        "需求讨论": ["功能", "需求", "市场", "目标"],
    }
    for tag, words in candidates.items():
        if any(word in text for word in words):
            tags.append(tag)
    return tags[:4] or ["需求讨论"]


def _speaker_count(cues: list[dict[str, Any]]) -> int:
    speakers = {x.get("speaker") for x in cues if x.get("speaker")}
    return len(speakers)


def _simple_summary(cues: list[dict[str, Any]], fallback: str) -> str:
    zh = "；".join([x.get("zh", "") for x in cues[:5] if x.get("zh")])
    if zh:
        return zh[:140] + ("..." if len(zh) > 140 else "")
    return fallback


def _slice_minutes(product: dict[str, Any], slice_index: int) -> dict[str, Any]:
    stages = product.get("stage_summaries") or []
    final_minutes = product.get("final_minutes") or {}
    stage = stages[min(slice_index, max(0, len(stages) - 1))] if stages else {}
    return {
        "stage_summaries": [stage] if stage else [],
        "speaker_insights": product.get("speaker_insights", []),
        "final_minutes": {
            "one_sentence_summary": stage.get("summary") or final_minutes.get("one_sentence_summary", ""),
            "key_decisions": final_minutes.get("key_decisions", []),
            "action_items": stage.get("actions") or final_minutes.get("action_items", []),
            "risks": stage.get("risks") or final_minutes.get("risks", []),
            "open_questions": stage.get("open_questions") or final_minutes.get("open_questions", []),
            "conclusion": final_minutes.get("conclusion", ""),
        },
        "success": bool(product.get("success")),
        "minutes_model": product.get("minutes_model"),
    }


def _seed_meetings() -> list[dict[str, Any]]:
    cues = read_json(ROOT / "outputs" / "web_cache" / "streaming_cues.json", [])
    product = read_json(ROOT / "outputs" / "web_cache" / "product_minutes.json", {})
    speaker_minutes = read_json(ROOT / "outputs" / "web_cache" / "speaker_minutes.json", {})
    demo_audio = ROOT / "outputs" / "web_cache" / "demo_sequence_10_segments.wav"
    if not cues:
        return []

    chunks = [
        {
            "meeting_id": "AMI-ES2004a-20260704",
            "title": "AMI 历史会议 ES2004a：遥控器产品启动讨论",
            "offset_days": 0,
            "topic": "产品启动",
            "start": 0,
            "end": min(22, len(cues)),
        },
        {
            "meeting_id": "AMI-ES2004a-20260702",
            "title": "AMI 历史会议 ES2004a：成本与功能需求讨论",
            "offset_days": 2,
            "topic": "成本与需求",
            "start": min(22, len(cues)),
            "end": min(44, len(cues)),
        },
        {
            "meeting_id": "AMI-ES2004a-20260628",
            "title": "AMI 历史会议 ES2004a：交互设计与会议结论",
            "offset_days": 6,
            "topic": "交互设计",
            "start": min(44, len(cues)),
            "end": len(cues),
        },
    ]
    seed = []
    for idx, item in enumerate(chunks):
        part = cues[item["start"] : item["end"]] or cues
        meeting_date = (_today() - timedelta(days=item["offset_days"])).strftime("%Y-%m-%d")
        minutes = _slice_minutes(product, idx)
        text_blob = json.dumps({"cues": part, "minutes": minutes}, ensure_ascii=False)
        meeting_dir = LIBRARY_ROOT / item["meeting_id"]
        write_json(meeting_dir / "transcript.json", part)
        write_json(meeting_dir / "product_minutes.json", minutes)
        write_json(meeting_dir / "speaker_minutes.json", speaker_minutes)
        if demo_audio.exists():
            target_audio = meeting_dir / "audio.wav"
            if not target_audio.exists():
                shutil.copyfile(demo_audio, target_audio)
            audio_path = str(target_audio.relative_to(ROOT))
        else:
            audio_path = ""
        summary = (minutes.get("final_minutes") or {}).get("one_sentence_summary") or _simple_summary(part, "")
        metadata = {
            "meeting_id": item["meeting_id"],
            "title": item["title"],
            "meeting_date": meeting_date,
            "time_range": meeting_date,
            "source_type": "dataset",
            "source_label": "AMI Corpus / ES2004a dataset slice",
            "status": "completed",
            "topic": item["topic"],
            "tags": _tag_pool(text_blob),
            "summary": summary,
            "duration_seconds": round(max(x.get("end", 0) for x in part) - min(x.get("start", 0) for x in part), 1),
            "speaker_count": _speaker_count(part),
            "action_count": len((minutes.get("final_minutes") or {}).get("action_items") or []),
            "risk_count": len((minutes.get("final_minutes") or {}).get("risks") or []),
            "audio_path": audio_path,
            "transcript_path": str((meeting_dir / "transcript.json").relative_to(ROOT)),
            "minutes_path": str((meeting_dir / "product_minutes.json").relative_to(ROOT)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(meeting_dir / "meeting.json", metadata)
        seed.append(metadata)
    return seed


def ensure_meeting_library() -> list[dict[str, Any]]:
    LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    existing = read_json(INDEX_PATH, [])
    non_seed = [x for x in existing if not str(x.get("meeting_id", "")).startswith("AMI-ES2004a-")]
    seed = _seed_meetings()
    merged = sorted(seed + non_seed, key=lambda x: x.get("meeting_date", ""), reverse=True)
    write_json(INDEX_PATH, merged)
    return merged


def load_meeting_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return ensure_meeting_library()
    return read_json(INDEX_PATH, [])


def load_meeting_detail(meeting_id: str) -> dict[str, Any]:
    meeting_dir = LIBRARY_ROOT / safe_id(meeting_id)
    metadata = read_json(meeting_dir / "meeting.json", {})
    transcript = read_json(meeting_dir / "transcript.json", [])
    minutes = read_json(meeting_dir / "product_minutes.json", {})
    diarization = read_json(meeting_dir / "diarization.json", {})
    return {"metadata": metadata, "transcript": transcript, "minutes": minutes, "diarization": diarization}


def add_uploaded_meeting(file_name: str, content: bytes, title: str, topic: str) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meeting_id = safe_id(f"UPLOAD-{timestamp}-{Path(file_name).stem}")
    meeting_dir = LIBRARY_ROOT / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file_name).suffix.lower() or ".wav"
    audio_path = meeting_dir / f"audio{suffix}"
    audio_path.write_bytes(content)
    metadata = {
        "meeting_id": meeting_id,
        "title": title.strip() or f"上传会议 {timestamp}",
        "meeting_date": datetime.now().strftime("%Y-%m-%d"),
        "time_range": datetime.now().strftime("%Y-%m-%d"),
        "source_type": "upload",
        "source_label": "用户上传音频",
        "status": "pending_processing",
        "topic": topic.strip() or "未标注主题",
        "tags": [topic.strip()] if topic.strip() else ["待处理"],
        "summary": "音频已进入会议库，等待执行 ASR、翻译、声纹识别和会议纪要生成。",
        "duration_seconds": 0,
        "speaker_count": 0,
        "action_count": 0,
        "risk_count": 0,
        "audio_path": str(audio_path.relative_to(ROOT)),
        "transcript_path": "",
        "minutes_path": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(meeting_dir / "meeting.json", metadata)
    index = [x for x in load_meeting_index() if x.get("meeting_id") != meeting_id]
    index.insert(0, metadata)
    write_json(INDEX_PATH, index)
    return metadata


def _count_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def add_realtime_meeting(
    title: str,
    audio_b64: str,
    transcript: list[dict[str, Any]],
    product_minutes: dict[str, Any],
    duration_seconds: float,
    route_mode: str = "cascade",
    canonical_route: str = "cascade",
    route_transcripts: dict[str, list[dict[str, Any]]] | None = None,
    comparison_metrics: dict[str, Any] | None = None,
    diarization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meeting_id = safe_id(f"REALTIME-{timestamp}")
    meeting_dir = LIBRARY_ROOT / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]
    audio_bytes = base64.b64decode(audio_b64)
    audio_path = meeting_dir / "audio.wav"
    audio_path.write_bytes(audio_bytes)

    transcript_path = meeting_dir / "transcript.json"
    minutes_path = meeting_dir / "product_minutes.json"
    write_json(transcript_path, transcript)
    write_json(minutes_path, product_minutes)
    diarization_path = ""
    if diarization:
        diarization_file = meeting_dir / "diarization.json"
        write_json(diarization_file, diarization)
        diarization_path = str(diarization_file.relative_to(ROOT))
    if route_transcripts or comparison_metrics:
        write_json(
            meeting_dir / "route_comparison.json",
            {
                "route_mode": route_mode,
                "canonical_route": canonical_route,
                "route_transcripts": route_transcripts or {},
                "comparison_metrics": comparison_metrics or {},
            },
        )

    final_minutes = product_minutes.get("final_minutes") or {}
    transcript_text = json.dumps({"transcript": transcript, "minutes": product_minutes}, ensure_ascii=False)
    speakers = {
        x.get("speaker")
        for x in transcript
        if x.get("speaker") and x.get("speaker") != "参会者识别中"
    }
    summary = final_minutes.get("one_sentence_summary") or _simple_summary(transcript, "实时会议已保存。")
    metadata = {
        "meeting_id": meeting_id,
        "title": title.strip() or f"实时会议 {timestamp}",
        "meeting_date": datetime.now().strftime("%Y-%m-%d"),
        "time_range": datetime.now().strftime("%Y-%m-%d"),
        "source_type": "realtime_microphone",
        "source_label": "浏览器实时麦克风",
        "status": "completed",
        "speaker_status": (diarization or {}).get("status", "pending_post_diarization"),
        "speaker_backend": (diarization or {}).get("backend", "online_provisional"),
        "diarization_path": diarization_path,
        "translation_route": route_mode,
        "canonical_route": canonical_route,
        "comparison_available": bool(route_transcripts and len(route_transcripts) > 1),
        "topic": "实时会议",
        "tags": _tag_pool(transcript_text),
        "summary": summary,
        "duration_seconds": round(float(duration_seconds or 0), 1),
        "speaker_count": max(len(speakers), int((diarization or {}).get("speaker_count") or 0)),
        "action_count": _count_items(final_minutes.get("action_items")),
        "risk_count": _count_items(final_minutes.get("risks")),
        "audio_path": str(audio_path.relative_to(ROOT)),
        "transcript_path": str(transcript_path.relative_to(ROOT)),
        "minutes_path": str(minutes_path.relative_to(ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(meeting_dir / "meeting.json", metadata)
    index = [x for x in load_meeting_index() if x.get("meeting_id") != meeting_id]
    index.insert(0, metadata)
    write_json(INDEX_PATH, index)
    return metadata


def update_meeting_diarization(
    meeting_id: str,
    transcript: list[dict[str, Any]],
    diarization: dict[str, Any],
) -> dict[str, Any]:
    """Persist a re-diarization result for an existing audio meeting."""
    safe_meeting_id = safe_id(meeting_id)
    meeting_dir = LIBRARY_ROOT / safe_meeting_id
    metadata_path = meeting_dir / "meeting.json"
    metadata = read_json(metadata_path, {})
    if not metadata:
        raise FileNotFoundError(f"会议不存在: {meeting_id}")

    diarization_path = meeting_dir / "diarization.json"
    write_json(diarization_path, diarization)
    if transcript and any(
        item.get("audio_start_seconds") is not None or item.get("start") is not None
        for item in transcript
    ):
        write_json(meeting_dir / "transcript.json", transcript)

    known_speakers = {
        item.get("speaker")
        for item in transcript
        if item.get("speaker") and item.get("speaker") != "参会者识别中"
    }
    metadata.update(
        {
            "speaker_status": diarization.get("status", "pending_post_diarization"),
            "speaker_backend": diarization.get("backend", "3d-speaker-local"),
            "diarization_path": str(diarization_path.relative_to(ROOT)),
            "speaker_count": max(
                len(known_speakers),
                int(diarization.get("speaker_count") or 0),
            ),
        }
    )
    write_json(metadata_path, metadata)
    index = [
        metadata if item.get("meeting_id") == safe_meeting_id else item
        for item in load_meeting_index()
    ]
    write_json(INDEX_PATH, index)
    return metadata


def add_pending_audio_meeting(
    meeting_id: str,
    audio_path: Path,
    title: str,
    topic: str,
    source_label: str,
    meeting_date: str | None = None,
) -> dict[str, Any]:
    safe_meeting_id = safe_id(meeting_id)
    meeting_dir = LIBRARY_ROOT / safe_meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "meeting_id": safe_meeting_id,
        "title": title,
        "meeting_date": meeting_date or datetime.now().strftime("%Y-%m-%d"),
        "time_range": meeting_date or datetime.now().strftime("%Y-%m-%d"),
        "source_type": "dataset",
        "source_label": source_label,
        "status": "pending_processing",
        "topic": topic,
        "tags": [topic, "待处理"],
        "summary": "音频已入库，等待执行 ASR、翻译、声纹识别和会议纪要生成。",
        "duration_seconds": 0,
        "speaker_count": 0,
        "action_count": 0,
        "risk_count": 0,
        # Dataset files remain in data/raw. The meeting record holds a reference
        # so importing many AMI sessions does not duplicate large WAV files.
        "audio_path": str(audio_path.relative_to(ROOT)),
        "transcript_path": "",
        "minutes_path": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(meeting_dir / "meeting.json", metadata)
    index = [x for x in load_meeting_index() if x.get("meeting_id") != safe_meeting_id]
    index.insert(0, metadata)
    write_json(INDEX_PATH, index)
    return metadata


def scan_downloaded_ami_audio() -> list[dict[str, Any]]:
    base = ROOT / "data" / "raw" / "amicorpus"
    if not base.exists():
        return []
    registered = []
    start_date = _today() - timedelta(days=1)
    for offset, path in enumerate(sorted(base.glob("*/audio/*.Mix-Headset.wav"))):
        meeting_id = path.stem.replace(".Mix-Headset", "")
        if meeting_id == "ES2004a":
            continue
        registered.append(
            add_pending_audio_meeting(
                meeting_id=f"AMI-{meeting_id}",
                audio_path=path,
                title=f"AMI 历史会议 {meeting_id}",
                topic="AMI 待处理会议",
                source_label=f"AMI Corpus / {meeting_id}",
                meeting_date=(start_date - timedelta(days=offset)).strftime("%Y-%m-%d"),
            )
        )
    return registered


def retrieve_meetings(query: str, limit: int = 5) -> list[dict[str, Any]]:
    # Prefer evidence-level semantic retrieval once the persistent RAG index is ready.
    # The lightweight metadata scorer remains a deterministic bootstrap fallback.
    try:
        from .meeting_rag import retrieve

        evidence = retrieve(query, limit=limit)
        if evidence:
            by_id: dict[str, dict[str, Any]] = {}
            for row in evidence:
                meeting = load_meeting_detail(row.get("meeting_id", "")).get("metadata", {})
                if meeting:
                    by_id.setdefault(str(meeting.get("meeting_id")), meeting)
            if by_id:
                return list(by_id.values())[:limit]
    except Exception:
        pass
    index = load_meeting_index()
    q = query.lower()
    query_terms = re.findall(r"[a-z0-9]{2,}", q)
    for phrase in re.findall(r"[\u4e00-\u9fff]+", query):
        query_terms.extend(phrase[index : index + 2] for index in range(max(0, len(phrase) - 1)))
    query_terms = [term for term in dict.fromkeys(query_terms) if term not in {"最近", "讨论", "有讨", "论台", "风吗"}]
    scored = []
    for item in index:
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("summary", "")),
                str(item.get("topic", "")),
                " ".join(item.get("tags", []) or []),
                str(item.get("meeting_date", "")),
                str(item.get("source_label", "")),
            ]
        ).lower()
        score = 0
        for token in query_terms:
            if token and token in haystack:
                score += 2 if token in str(item.get("title", "")).lower() else 1
        if any(word in q for word in ["上周", "最近", "这周", "本周", "一周"]):
            score += 1
        if any(word in q for word in ["所有", "哪些", "介绍", "会议"]):
            score += 1
        if score:
            scored.append((score, item))
    if not scored:
        return index[:limit]
    return [x for _, x in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]


def meeting_context(query: str, limit: int = 5) -> str:
    try:
        from .meeting_rag import context_for_query

        evidence = context_for_query(query, limit=limit)
        if evidence and evidence != '{"rag_evidence": []}':
            return evidence[:60000]
    except Exception:
        pass
    items = retrieve_meetings(query, limit=limit)
    details = []
    for item in items:
        detail = load_meeting_detail(item.get("meeting_id", ""))
        minutes = detail.get("minutes") or {}
        transcript = detail.get("transcript") or []
        details.append(
            {
                "metadata": item,
                "final_minutes": minutes.get("final_minutes", {}),
                "stage_summaries": minutes.get("stage_summaries", [])[:3],
                "transcript_sample": transcript[:20],
            }
        )
    return json.dumps({"retrieved_meetings": details}, ensure_ascii=False)[:60000]
