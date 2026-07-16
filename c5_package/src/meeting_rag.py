from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import dashscope
import numpy as np

from .meeting_library import LIBRARY_ROOT, load_meeting_detail, load_meeting_index


RAG_ROOT = LIBRARY_ROOT / "rag"
META_PATH = RAG_ROOT / "chunks.json"
VECTOR_PATH = RAG_ROOT / "vectors.npz"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
_index_lock = threading.Lock()


def _seconds(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if os.getenv("ALI_DASHSCOPE_BASE_URL"):
        dashscope.base_http_api_url = os.getenv("ALI_DASHSCOPE_BASE_URL")


def _chunk_text(metadata: dict[str, Any], detail: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = detail.get("transcript") or []
    minutes = detail.get("minutes") or {}
    chunks: list[dict[str, Any]] = []
    coverage = "完整会议" if metadata.get("status") == "completed" else f"会议采样，覆盖 {float(metadata.get('coverage_ratio', 0)) * 100:.1f}%"
    summary = (minutes.get("final_minutes") or {}).get("one_sentence_summary") or metadata.get("summary", "")
    if summary:
        chunks.append({
            "chunk_id": f"{metadata['meeting_id']}:summary",
            "meeting_id": metadata["meeting_id"], "title": metadata.get("title", ""), "date": metadata.get("meeting_date", ""),
            "topic": metadata.get("topic", ""), "tags": metadata.get("tags", []), "speaker": "会议纪要", "start": 0.0, "end": 0.0,
            "text": f"会议：{metadata.get('title','')}。数据状态：{coverage}。主题：{metadata.get('topic','')}。纪要：{summary}", "kind": "summary", "coverage": coverage,
        })
    for index in range(0, len(transcript), 4):
        rows = transcript[index : index + 4]
        if not rows:
            continue
        text = "\n".join(
            f"{row.get('speaker') or '参会者识别中'}: {row.get('corrected') or row.get('source') or row.get('en') or ''} {row.get('translation') or row.get('english') or row.get('zh') or ''}".strip()
            for row in rows
        ).strip()
        if not text:
            continue
        chunks.append({
            "chunk_id": f"{metadata['meeting_id']}:turn:{index // 4}",
            "meeting_id": metadata["meeting_id"], "title": metadata.get("title", ""), "date": metadata.get("meeting_date", ""),
            "topic": metadata.get("topic", ""), "tags": metadata.get("tags", []),
            "speaker": ", ".join(sorted({str(r.get('speaker') or '参会者识别中') for r in rows})),
            "start": _seconds(rows[0].get("start", rows[0].get("time", 0))),
            "end": _seconds(rows[-1].get("end", rows[-1].get("time", 0))), "text": text, "kind": "transcript", "coverage": coverage,
        })
    return chunks


def build_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for metadata in load_meeting_index():
        if metadata.get("status") not in {"completed", "partial_processing"}:
            continue
        chunks.extend(_chunk_text(metadata, load_meeting_detail(metadata.get("meeting_id", ""))))
    return chunks


def _embedding_batch(texts: list[str]) -> list[list[float]]:
    _load_env()
    response = dashscope.TextEmbedding.call(model=EMBEDDING_MODEL, input=texts, dimension=1024, output_type="dense")
    if getattr(response, "status_code", 200) != 200:
        raise RuntimeError(f"embedding failed: {getattr(response, 'message', response)}")
    output = getattr(response, "output", {}) or {}
    rows = output.get("embeddings", []) if isinstance(output, dict) else getattr(output, "embeddings", [])
    ordered = sorted(rows, key=lambda item: item.get("text_index", 0))
    vectors = [item.get("embedding") for item in ordered]
    if len(vectors) != len(texts) or any(not row for row in vectors):
        raise RuntimeError("embedding response is incomplete")
    return vectors


def build_index() -> dict[str, Any]:
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError("没有可索引的已完成会议")
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 10):
        vectors.extend(_embedding_batch([row["text"][:8000] for row in chunks[start : start + 10]]))
    array = np.asarray(vectors, dtype="float32")
    array /= np.linalg.norm(array, axis=1, keepdims=True) + 1e-8
    RAG_ROOT.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(VECTOR_PATH, vectors=array)
    return {"success": True, "chunk_count": len(chunks), "dimension": int(array.shape[1]), "model": EMBEDDING_MODEL}


def _query_terms(query: str) -> list[str]:
    """Return useful lexical anchors for mixed Chinese/English meeting queries."""
    terms = re.findall(r"[a-z0-9]{2,}", query.lower())
    for phrase in re.findall(r"[\u4e00-\u9fff]+", query):
        # Chinese text has no spaces. Bigrams retain concrete nouns such as
        # "台风" inside conversational questions like "最近有讨论台风吗".
        terms.extend(phrase[index : index + 2] for index in range(max(0, len(phrase) - 1)))
        terms.extend(phrase[index : index + 3] for index in range(max(0, len(phrase) - 2)))
    stop_terms = {"最近", "讨论", "有讨", "论台", "风吗", "最近有", "有讨论", "吗"}
    return list(dict.fromkeys(term for term in terms if term and term not in stop_terms))


def _keyword_score(query: str, text: str) -> float:
    query_tokens = _query_terms(query)
    haystack = text.lower()
    return float(sum(1 for token in query_tokens if token and token in haystack))


def _index_is_stale() -> bool:
    if not META_PATH.exists() or not VECTOR_PATH.exists():
        return True
    try:
        indexed = {str(row.get("meeting_id", "")) for row in json.loads(META_PATH.read_text(encoding="utf-8"))}
        expected = {
            str(row.get("meeting_id", ""))
            for row in load_meeting_index()
            if row.get("status") in {"completed", "partial_processing"}
        }
        return not expected.issubset(indexed)
    except Exception:
        return True


def ensure_index_current() -> bool:
    """Synchronize newly saved meetings before answering a library question."""
    if not _index_is_stale():
        return False
    with _index_lock:
        if not _index_is_stale():
            return False
        build_index()
        return True


def retrieve(query: str, limit: int = 5) -> list[dict[str, Any]]:
    try:
        ensure_index_current()
    except Exception:
        # Metadata retrieval remains available when embedding refresh fails.
        pass
    if not META_PATH.exists() or not VECTOR_PATH.exists():
        return []
    chunks = json.loads(META_PATH.read_text(encoding="utf-8"))
    vectors = np.load(VECTOR_PATH)["vectors"]
    query_vector = np.asarray(_embedding_batch([query])[0], dtype="float32")
    query_vector /= np.linalg.norm(query_vector) + 1e-8
    semantic = vectors @ query_vector
    keyword = np.asarray([_keyword_score(query, f"{item.get('title','')} {item.get('topic','')} {item.get('text','')}") for item in chunks], dtype="float32")
    keyword = keyword / (keyword.max() + 1e-8) if keyword.max() else keyword
    score = 0.78 * semantic + 0.22 * keyword
    selected = np.argsort(-score)[:limit]
    results = []
    for idx in selected:
        item = dict(chunks[int(idx)])
        item["score"] = round(float(score[int(idx)]), 4)
        item["source"] = f"{item['title']} | {item['date']} | {item['speaker']} | {item['start']:.1f}s"
        results.append(item)
    return results


def context_for_query(query: str, limit: int = 5) -> str:
    evidence = retrieve(query, limit=limit)
    return json.dumps({"rag_evidence": evidence}, ensure_ascii=False)
