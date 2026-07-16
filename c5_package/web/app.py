from __future__ import annotations

import json
import os
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

from src.meeting_library import (
    add_uploaded_meeting,
    ensure_meeting_library,
    load_meeting_detail,
    load_meeting_index,
    scan_downloaded_ami_audio,
)


ROOT = Path(__file__).resolve().parents[1]
REALTIME_PORT = int(os.getenv("REALTIME_PORT", "8765"))


def read_json(path: str, default):
    full = ROOT / path
    if not full.exists():
        return default
    with full.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: str | None) -> Path | None:
    if not path:
        return None
    full = Path(path)
    return full if full.is_absolute() else ROOT / full


def rerun_stored_diarization(meeting_id: str) -> dict:
    url = f"http://127.0.0.1:{REALTIME_PORT}/api/meetings/{urllib.parse.quote(meeting_id, safe='')}/diarize"
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def run_pipeline() -> tuple[int, str]:
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            env.setdefault(key.strip(), value)
    proc = subprocess.run(
        ["python", "-m", "src.pipeline", "--stage", "all"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def ensure_realtime_backend() -> bool:
    if is_port_open("127.0.0.1", REALTIME_PORT):
        return True

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    stdout = (logs / "realtime.out.log").open("a", encoding="utf-8")
    stderr = (logs / "realtime.err.log").open("a", encoding="utf-8")
    subprocess.Popen(
        ["python", "-m", "uvicorn", "realtime.server:app", "--host", "127.0.0.1", "--port", str(REALTIME_PORT)],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    for _ in range(20):
        if is_port_open("127.0.0.1", REALTIME_PORT):
            return True
        import time

        time.sleep(0.25)
    return False


def load_data():
    segments = read_json("data/segments/test_segments.json", [])
    asr = {x["id"]: x for x in read_json("outputs/asr/asr_segments.json", [])}
    bilingual = {x["id"]: x for x in read_json("outputs/translation/bilingual_segments.json", [])}
    omni = {x["id"]: x for x in read_json("outputs/c4_omni/omni_results.json", [])}
    compare = read_json("outputs/eval/c3_vs_c4_compare.json", [])
    latency = read_json("outputs/eval/latency_report.json", {})
    summary = read_json("outputs/summary/summary.json", {})
    speaker_minutes = read_json("outputs/web_cache/speaker_minutes.json", {})
    return segments, asr, bilingual, omni, compare, latency, summary, speaker_minutes


def audio_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def build_demo_audio_and_cues(segments: list[dict], bilingual: dict, omni: dict) -> tuple[Path | None, list[dict]]:
    """Fallback continuous timeline when utterance-level streaming cues do not exist."""
    if not segments:
        return None, []

    out_path = ROOT / "outputs/web_cache/demo_sequence_10_segments.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_chunks = []
    sample_rate = None
    cues = []
    cursor = 0.0
    for seg in segments:
        audio_path = rel(seg.get("audio_path"))
        b = bilingual.get(seg["id"], {})
        o = omni.get(seg["id"], {})
        duration = float(seg.get("duration", 25.0) or 25.0)
        cues.append(
            {
                "id": seg["id"],
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
                "origin": f"{float(seg.get('start', 0.0)):.1f}s-{float(seg.get('end', 0.0)):.1f}s",
                "en": b.get("source_text", ""),
                "zh": b.get("translation_zh", ""),
                "omni": o.get("omni_text", ""),
            }
        )
        cursor += duration
        if audio_path and audio_path.exists():
            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(data)

    if audio_chunks and sample_rate:
        sf.write(str(out_path), np.concatenate(audio_chunks), sample_rate)
        return out_path, cues
    return None, cues


def load_streaming_demo(segments: list[dict], bilingual: dict, omni: dict) -> tuple[Path | None, list[dict]]:
    cue_path = ROOT / "outputs/web_cache/streaming_cues.json"
    audio_path = ROOT / "outputs/web_cache/demo_sequence_10_segments.wav"
    if cue_path.exists() and audio_path.exists():
        return audio_path, read_json("outputs/web_cache/streaming_cues.json", [])
    return build_demo_audio_and_cues(segments, bilingual, omni)


def playback_meeting_sources(demo_audio: Path | None) -> list[dict]:
    """Return recordings that can be safely embedded in the live playback UI."""
    sources: list[dict] = []
    if demo_audio and demo_audio.exists():
        sources.append(
            {
                "key": "default-demo",
                "label": "默认 AMI 演示序列",
                "detail": "10 个已处理 AMI 说话轮次",
                "audio_path": demo_audio,
            }
        )
    for item in load_meeting_index():
        audio_path = rel(item.get("audio_path"))
        if not audio_path or not audio_path.exists() or audio_path.suffix.lower() != ".wav":
            continue
        try:
            duration = float(sf.info(str(audio_path)).duration)
        except Exception:
            duration = float(item.get("duration_seconds") or 0)
        source_kind = "AMI 数据集" if item.get("source_type") == "dataset" else "历史实时会议"
        sources.append(
            {
                "key": str(item.get("meeting_id")),
                "label": f"{item.get('meeting_date', '')} · {item.get('title', '未命名会议')}",
                "detail": f"{source_kind} · {duration:.0f}s · {item.get('topic', '未分类')}",
                "audio_path": audio_path,
            }
        )
    return sources


def prepare_playback_audio(audio_path: Path, source_key: str, start_seconds: float, max_seconds: float = 180.0) -> tuple[Path, float]:
    """Create a bounded WAV excerpt for long library recordings."""
    info = sf.info(str(audio_path))
    total_seconds = float(info.duration)
    if total_seconds <= max_seconds:
        return audio_path, total_seconds

    safe_key = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_key)
    start_seconds = max(0.0, min(float(start_seconds), max(0.0, total_seconds - max_seconds)))
    end_seconds = min(total_seconds, start_seconds + max_seconds)
    out_path = ROOT / "outputs" / "web_cache" / "playback_clips" / f"{safe_key}_{int(start_seconds)}_{int(end_seconds)}.wav"
    if out_path.exists():
        return out_path, end_seconds - start_seconds

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(audio_path), "r") as reader:
        reader.seek(int(start_seconds * reader.samplerate))
        audio = reader.read(int((end_seconds - start_seconds) * reader.samplerate), dtype="float32", always_2d=False)
        sample_rate = reader.samplerate
    if getattr(audio, "ndim", 1) > 1:
        audio = np.asarray(audio, dtype="float32").mean(axis=1)
    sf.write(str(out_path), audio, sample_rate)
    return out_path, end_seconds - start_seconds


def render_legacy_realtime_meeting_player(audio_path: Path, cues: list[dict], summary: dict, speaker_minutes: dict) -> None:
    cues_json = json.dumps(cues, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    speaker_minutes_json = json.dumps(speaker_minutes, ensure_ascii=False)
    src = audio_data_url(audio_path)
    components.html(
        f"""
<div class="demo-shell">
  <div class="hero">
    <div>
      <div class="eyebrow">Meeting Recording Input</div>
      <h2>会议录音输入 → 实时双语字幕</h2>
      <p>这里使用会议录音作为数据输入。点击“用会议录音开始真实模型流式处理”会按 100ms 音频帧送入实时 ASR；下方预生成时间线用于稳定展示和会后纪要。</p>
    </div>
    <div class="stats">
      <div><b>{len(cues)}</b><span>说话段</span></div>
      <div><b>{max((c["end"] for c in cues), default=0):.0f}s</b><span>演示音频</span></div>
      <div><b>A/B/C</b><span>说话人标签</span></div>
    </div>
  </div>

  <div class="player-card">
    <button id="startBtn" class="start-btn">▶ 播放会议录音并查看字幕时间线</button>
    <div class="player-actions">
      <button id="modelStreamBtn" class="secondary-btn primary-secondary">用会议录音开始真实模型流式处理</button>
      <button id="minutesBtn" class="secondary-btn">查看会议纪要</button>
      <label class="follow-toggle"><input id="autoFollow" type="checkbox" checked> 自动跟随当前字幕</label>
    </div>
    <audio id="meetingAudio" controls preload="metadata">
    <source src="{src}" type="audio/wav">
  </audio>
    <div class="progress-row">
      <span id="timeNow">00:00.0</span>
      <div class="bar"><div id="barFill"></div></div>
      <span id="timeTotal">--:--</span>
    </div>
  </div>

  <div class="content-grid">
    <section class="subtitle-panel">
      <div class="panel-title">实时滚动字幕</div>
      <div id="cueList" class="cue-list">
        <div class="empty-state">点击播放后开始监听；每个说话轮次结束后生成字幕框。</div>
      </div>
      <div class="panel-title sub-title">真实模型流式输出</div>
      <div id="modelSyncStatus" class="sync-status">同步监控：尚未开始真实模型流式处理。</div>
      <div id="modelStream" class="model-stream">
        <div class="empty-state">点击“用会议录音开始真实模型流式处理”后，会把这段会议录音按 100ms PCM 帧送入实时 ASR。</div>
      </div>
    </section>
    <section id="minutesPanel" class="minutes-panel locked">
      <div class="panel-title">播放完成后的会议记录与纪要</div>
      <div id="minutesHint" class="hint">音频播放结束后自动显示；也可以拖动进度到末尾查看。</div>
      <div id="minutesBody"></div>
    </section>
  </div>
</div>
<style>
  body {{
    margin: 0;
  }}
  .demo-shell {{
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    overflow: hidden;
    background: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #0f172a;
  }}
  .hero {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 24px;
    color: white;
    background: linear-gradient(135deg, #0f766e, #0f172a);
  }}
  .eyebrow {{
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #99f6e4;
    font-size: 12px;
    font-weight: 700;
  }}
  h2 {{
    margin: 6px 0 8px;
    font-size: 28px;
  }}
  p {{
    margin: 0;
    color: #d9f2ef;
    max-width: 720px;
  }}
  .stats {{
    display: flex;
    gap: 10px;
    align-items: stretch;
  }}
  .stats div {{
    min-width: 86px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 12px;
    background: rgba(255,255,255,.1);
    text-align: center;
  }}
  .stats b {{
    display: block;
    font-size: 20px;
  }}
  .stats span {{
    display: block;
    color: #ccfbf1;
    font-size: 12px;
    margin-top: 2px;
  }}
  .player-card {{
    padding: 18px 24px 10px;
    background: white;
    border-bottom: 1px solid #e2e8f0;
  }}
  audio {{
    width: 100%;
  }}
  .start-btn {{
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: #0f766e;
    color: white;
    font-size: 17px;
    font-weight: 800;
    cursor: pointer;
  }}
  .start-btn:hover {{
    background: #115e59;
  }}
  .player-actions {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .secondary-btn {{
    border: 0;
    border-radius: 10px;
    padding: 10px 12px;
    background: #e2e8f0;
    color: #0f172a;
    font-weight: 800;
    cursor: pointer;
  }}
  .primary-secondary {{
    background: #0f766e;
    color: white;
  }}
  .follow-toggle {{
    display: inline-flex;
    gap: 6px;
    align-items: center;
    color: #475569;
    font-size: 13px;
  }}
  .progress-row {{
    display: grid;
    grid-template-columns: 68px 1fr 68px;
    gap: 10px;
    align-items: center;
    margin-top: 10px;
    font-size: 13px;
    color: #475569;
  }}
  .bar {{
    height: 8px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
  }}
  #barFill {{
    width: 0%;
    height: 100%;
    background: #0f766e;
  }}
  .content-grid {{
    display: grid;
    grid-template-columns: 1.35fr .9fr;
    gap: 16px;
    padding: 18px;
  }}
  .subtitle-panel,
  .minutes-panel {{
    min-height: 440px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
  }}
  .panel-title {{
    padding: 14px 16px;
    border-bottom: 1px solid #e2e8f0;
    font-weight: 800;
    background: #f8fafc;
  }}
  .cue-list {{
    max-height: 420px;
    overflow-y: auto;
    padding: 12px;
    scroll-behavior: smooth;
  }}
  .sub-title {{
    border-top: 1px solid #e2e8f0;
  }}
  .model-stream {{
    max-height: 260px;
    overflow-y: auto;
    padding: 12px;
    scroll-behavior: smooth;
  }}
  .sync-status {{
    margin: 10px 12px 0;
    padding: 8px 10px;
    border-radius: 8px;
    background: #ecfdf5;
    color: #166534;
    font-size: 12px;
    font-weight: 700;
  }}
  .sync-status.warn {{
    background: #fff7ed;
    color: #9a3412;
  }}
  .sync-status.error {{
    background: #fef2f2;
    color: #991b1b;
  }}
  .empty-state {{
    padding: 46px 18px;
    color: #64748b;
    text-align: center;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    background: #f8fafc;
  }}
  .cue {{
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    transition: all .18s ease;
  }}
  .cue.active {{
    background: #ecfeff;
    border-color: #06b6d4;
    box-shadow: 0 10px 24px rgba(8, 145, 178, .12);
    transform: translateY(-1px);
  }}
  .speaker {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 999px;
    background: #0f766e;
    color: white;
    font-size: 12px;
    font-weight: 800;
  }}
  .speaker.pending {{
    background: #f59e0b;
    color: #111827;
  }}
  .speaker-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #99f6e4;
  }}
  .cue-meta {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: #64748b;
    font-size: 12px;
    margin-bottom: 6px;
  }}
  .cue-en {{
    color: #334155;
    font-size: 15px;
    margin-bottom: 8px;
  }}
  .cue-zh {{
    color: #0f172a;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .cue-omni {{
    color: #475569;
    font-size: 14px;
    border-left: 3px solid #cbd5e1;
    padding-left: 8px;
  }}
  .minutes-panel {{
    position: relative;
    max-height: 640px;
    overflow-y: auto;
  }}
  .minutes-panel.locked #minutesBody {{
    display: none;
  }}
  .hint {{
    margin: 14px;
    padding: 10px 12px;
    border-radius: 8px;
    background: #fff7ed;
    color: #9a3412;
    font-size: 13px;
  }}
  #minutesBody {{
    padding: 0 16px 16px;
    line-height: 1.65;
  }}
  #minutesBody h3 {{
    margin: 12px 0 8px;
    font-size: 16px;
  }}
  #minutesBody h4 {{
    margin: 10px 0 4px;
    font-size: 14px;
    color: #334155;
  }}
  #minutesBody ul {{
    margin: 6px 0 12px 18px;
    padding: 0;
  }}
  #minutesBody li {{
    margin: 4px 0;
  }}
  .minute-card {{
    margin: 10px 0;
    padding: 10px 12px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #f8fafc;
  }}
  .minute-card h3 {{
    margin-top: 0 !important;
  }}
  @media (max-width: 820px) {{
    .hero, .stats {{
      display: block;
    }}
    .stats div {{
      display: inline-block;
      margin-top: 12px;
      margin-right: 8px;
    }}
    .content-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
<script>
  const cues = {cues_json};
  const summary = {summary_json};
  const speakerMinutes = {speaker_minutes_json};
  const audio = document.getElementById("meetingAudio");
  const startBtn = document.getElementById("startBtn");
  const modelStreamBtn = document.getElementById("modelStreamBtn");
  const minutesBtn = document.getElementById("minutesBtn");
  const autoFollow = document.getElementById("autoFollow");
  const cueList = document.getElementById("cueList");
  const modelStream = document.getElementById("modelStream");
  const modelSyncStatus = document.getElementById("modelSyncStatus");
  const timeNow = document.getElementById("timeNow");
  const timeTotal = document.getElementById("timeTotal");
  const barFill = document.getElementById("barFill");
  const minutesPanel = document.getElementById("minutesPanel");
  const minutesHint = document.getElementById("minutesHint");
  const minutesBody = document.getElementById("minutesBody");
  let hasStarted = false;
  let userScrolledCueList = false;
  let modelWs = null;
  let meetingAudioContext = null;
  let meetingAudioSource = null;
  let meetingAudioOutputConnected = false;

  function fmt(t) {{
    const m = Math.floor(t / 60).toString().padStart(2, "0");
    const s = (t % 60).toFixed(1).padStart(4, "0");
    return `${{m}}:${{s}}`;
  }}
  function esc(s) {{
    return (s || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
  }}
  function addModelCard(cls, text, meta) {{
    const empty = modelStream.querySelector(".empty-state");
    if (empty) empty.remove();
    const card = document.createElement("div");
    card.className = `cue ${{cls}}`;
    card.innerHTML = `<div class="cue-meta"><span>${{esc(meta)}}</span><span>${{new Date().toLocaleTimeString()}}</span></div><div class="cue-zh">${{esc(text || "...")}}</div>`;
    modelStream.appendChild(card);
    modelStream.scrollTop = modelStream.scrollHeight;
    return card;
  }}
  function updateModelSync(mode, sentSeconds) {{
    const playSeconds = audio.currentTime || 0;
    const drift = sentSeconds - playSeconds;
    const driftText = `${{drift >= 0 ? "+" : ""}}${{drift.toFixed(2)}}s`;
    modelSyncStatus.className = "sync-status";
    if (drift > 0.30) modelSyncStatus.classList.add("error");
    else if (Math.abs(drift) > 0.80) modelSyncStatus.classList.add("warn");
    modelSyncStatus.textContent = `同步监控：${{mode}} | 播放器 ${{fmt(playSeconds)}} | 已送入模型 ${{fmt(Math.max(0, sentSeconds))}} | 差值 ${{driftText}}`;
  }}
  function typingDuration(c) {{
    const totalChars = Array.from((c.en || "") + (c.zh || "")).length;
    return Math.max(1.1, Math.min(3.0, totalChars / 42));
  }}
  function cueProgress(c, t) {{
    if (t < c.start) return 0;
    return Math.max(0, Math.min(1, (t - c.start) / Math.max(1.0, c.end - c.start)));
  }}
  function partialText(text, progress) {{
    const chars = Array.from(text || "");
    const n = Math.max(0, Math.min(chars.length, Math.ceil(chars.length * progress)));
    return chars.slice(0, n).join("");
  }}
  function cueHtml(c, i, progress) {{
    const en = progress >= 1 ? c.en : partialText(c.en, progress);
    const zh = progress >= 1 ? c.zh : partialText(c.zh, progress);
    const omni = progress >= 1 ? c.omni : partialText(c.omni, progress);
    const omniHtml = c.omni ? `<div class="cue-omni"><b>C4 Omni：</b>${{esc(omni)}}</div>` : "";
    return `
      <article class="cue" id="cue-${{i}}">
        <div class="cue-meta">
          <span class="speaker ${{c.speaker_pending ? "pending" : ""}}"><span class="speaker-dot"></span>${{esc(c.speaker || "参会者识别中")}}</span>
          <span>${{fmt(c.start)}}-${{fmt(c.end)}} · ${{esc(c.speaker_role || "")}} · 原音频 ${{esc(String(c.source_start || c.start))}}s</span>
        </div>
        <div class="cue-en">${{esc(en) || "..."}}<span class="cursor">${{progress < 1 ? "▌" : ""}}</span></div>
        <div class="cue-zh">${{esc(zh) || "..."}}<span class="cursor">${{progress < 1 ? "▌" : ""}}</span></div>
        ${{omniHtml}}
      </article>
    `;
  }}
  function renderCues(t = 0) {{
    if (!hasStarted) {{
      cueList.innerHTML = `<div class="empty-state">点击播放后开始监听；字幕会先实时输出，发言人稍后自动修正。</div>`;
      return;
    }}
    const visible = cues
      .map((c, i) => ({{ c, i, progress: cueProgress(c, t) }}))
      .filter(x => t >= x.c.start || x.progress > 0);
    cueList.innerHTML = visible.length
      ? visible.map(x => cueHtml(x.c, x.i, x.progress)).join("")
      : `<div class="empty-state">点击播放后开始监听；每个说话轮次结束后生成字幕框。</div>`;
  }}
  function renderMinutes() {{
    const speakerStatus = speakerMinutes.speaker_status || "pending";
    const productMinutes = speakerMinutes.product_minutes || {{}};
    const finalMinutes = productMinutes.final_minutes || {{}};
    const stageSummaries = productMinutes.stage_summaries || [];
    const speakerInsights = productMinutes.speaker_insights || [];
    function list(items) {{
      return (items || []).length ? `<ul>${{items.map(x => `<li>${{esc(typeof x === "string" ? x : JSON.stringify(x))}}</li>`).join("")}}</ul>` : "<p>暂无</p>";
    }}
    function stageHtml(stage) {{
      const speakerPoints = (stage.speaker_points || []).map(x => `<li><b>${{esc(x.speaker || "")}}：</b>${{esc(x.point || "")}}</li>`).join("");
      return `
        <div class="minute-card">
          <h3>${{esc(stage.window || "")}} · ${{esc(stage.title || "阶段摘要")}}</h3>
          <p>${{esc(stage.summary || "")}}</p>
          ${{speakerPoints ? `<h4>发言人要点</h4><ul>${{speakerPoints}}</ul>` : ""}}
          ${{(stage.actions || []).length ? `<h4>阶段行动项</h4>${{list(stage.actions)}}` : ""}}
          ${{(stage.risks || []).length ? `<h4>阶段风险</h4>${{list(stage.risks)}}` : ""}}
          ${{(stage.open_questions || []).length ? `<h4>待确认问题</h4>${{list(stage.open_questions)}}` : ""}}
        </div>
      `;
    }}
    function insightHtml(item) {{
      return `
        <div class="minute-card">
          <h3>${{esc(item.speaker || "")}} <small>${{esc(item.stance_or_role || "")}}</small></h3>
          <h4>主要观点</h4>${{list(item.main_points)}}
          <h4>达成一致</h4>${{list(item.agreements)}}
          <h4>分歧或担忧</h4>${{list(item.disagreements_or_concerns)}}
        </div>
      `;
    }}
    const speakerBlocks = (speakerMinutes.speakers || []).map(sp => `
      <h3>${{esc(sp.speaker)}} <small>(${{sp.turn_count || 0}} 段，${{esc(sp.speaker_source || speakerStatus)}})</small></h3>
      <ul>${{(sp.turns || []).map(t => `<li><b>${{esc(t.time)}}：</b>${{esc(t.zh || t.en || "")}}</li>`).join("")}}</ul>
    `).join("");
    const points = (summary.key_points || []).map(x => `<li>${{esc(x)}}</li>`).join("");
    const keywords = (summary.keywords || []).map(x => `<span>${{esc(x)}}</span>`).join("、");
    const actions = (summary.questions_or_actions || []).map(x => `<li>${{esc(x)}}</li>`).join("");
    const timeline = (summary.timeline || []).map(x => `<li><b>${{esc(x.time)}}：</b>${{esc(x.topic)}}</li>`).join("");
    minutesBody.innerHTML = `
      <h3>阶段性摘要</h3>
      ${{stageSummaries.length ? stageSummaries.map(stageHtml).join("") : "<p>暂无阶段摘要。</p>"}}
      <h3>按发言人归纳</h3>
      ${{speakerInsights.length ? speakerInsights.map(insightHtml).join("") : "<p>暂无发言人洞察。</p>"}}
      <h3>最终会议纪要</h3>
      <p>${{esc(finalMinutes.one_sentence_summary || summary.one_sentence_summary) || "暂无摘要"}}</p>
      <h4>关键决策</h4>${{list(finalMinutes.key_decisions)}}
      <h4>行动项</h4>${{list(finalMinutes.action_items || summary.questions_or_actions)}}
      <h4>风险</h4>${{list(finalMinutes.risks)}}
      <h4>待确认问题</h4>${{list(finalMinutes.open_questions)}}
      <h4>会议结论</h4>
      <p>${{esc(finalMinutes.conclusion || "") || "暂无"}}</p>
      <h3>按发言人整理的会议记录</h3>
      <p>${{speakerStatus === "pending" ? "说话人仍在识别中；当前先按待识别发言记录展示，diarization 完成后会自动修正到具体参会者。" : "说话人识别已完成，以下按参会者归档。"}}</p>
      ${{speakerBlocks || "<p>暂无发言记录。</p>"}}
      <h3>关键要点</h3>
      <ul>${{points}}</ul>
      <h3>关键词</h3>
      <p>${{keywords || "暂无"}}</p>
      <h3>时间线</h3>
      <ul>${{timeline}}</ul>
    `;
  }}
  function revealMinutes() {{
    minutesPanel.classList.remove("locked");
    minutesHint.textContent = "播放已结束，会议纪要如下。";
    minutesHint.style.background = "#ecfdf5";
    minutesHint.style.color = "#166534";
    minutesPanel.scrollIntoView({{ block: "center", behavior: "smooth" }});
  }}
  function updateCue() {{
    const t = audio.currentTime || 0;
    timeNow.textContent = fmt(t);
    if (audio.duration) {{
      timeTotal.textContent = fmt(audio.duration);
      barFill.style.width = `${{Math.min(100, (t / audio.duration) * 100)}}%`;
    }}
    renderCues(t);
    const cue = cues.find(c => t >= c.start && t <= c.end);
    document.querySelectorAll(".cue").forEach(el => el.classList.remove("active"));
    if (cue) {{
      const idx = cues.indexOf(cue);
      const el = document.getElementById(`cue-${{idx}}`);
      if (el) {{
        el.classList.add("active");
        if (autoFollow.checked && !userScrolledCueList) {{
          el.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
        }}
      }}
    }}
    if (audio.duration && t >= audio.duration - 0.25) revealMinutes();
  }}
  renderCues(0);
  renderMinutes();
  audio.addEventListener("timeupdate", updateCue);
  audio.addEventListener("play", () => {{
    hasStarted = true;
    updateCue();
  }});
  audio.addEventListener("pause", updateCue);
  audio.addEventListener("seeked", updateCue);
  audio.addEventListener("ended", revealMinutes);
  audio.addEventListener("loadedmetadata", updateCue);
  cueList.addEventListener("wheel", () => {{
    userScrolledCueList = true;
    autoFollow.checked = false;
  }});
  autoFollow.addEventListener("change", () => {{
    userScrolledCueList = !autoFollow.checked;
    if (autoFollow.checked) updateCue();
  }});
  function downsample(buffer, inRate, outRate) {{
    if (outRate === inRate) return buffer;
    const ratio = inRate / outRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {{
      const start = Math.floor(i * ratio);
      const end = Math.floor((i + 1) * ratio);
      let sum = 0, count = 0;
      for (let j = start; j < end && j < buffer.length; j++) {{ sum += buffer[j]; count++; }}
      result[i] = count ? sum / count : 0;
    }}
    return result;
  }}
  function floatTo16BitPCM(float32) {{
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {{
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }}
    return out;
  }}
  async function streamDemoAudioToModel() {{
    modelStream.innerHTML = `<div class="empty-state">正在读取录音并连接实时 ASR...</div>`;
    const wsUrl = `ws://127.0.0.1:{REALTIME_PORT}/ws/realtime?source_lang=en&target_lang=zh&max_sentence_silence=2200`;
    modelWs = new WebSocket(wsUrl);
    modelWs.binaryType = "arraybuffer";
    let zhCard = null;
    let trCard = null;
    modelWs.onmessage = ev => {{
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") addModelCard("active", `已连接 ${{msg.asr_model}}，开始按录音播放速度送流。`, "system");
      if (msg.type === "asr_partial") {{
        if (!zhCard) zhCard = addModelCard("active", msg.text, "English ASR partial");
        else zhCard.querySelector(".cue-zh").textContent = msg.text;
      }}
      if (msg.type === "asr_final") {{
        if (!zhCard) zhCard = addModelCard("active", msg.text, "English ASR final");
        zhCard.classList.remove("active");
        zhCard.querySelector(".cue-meta span").textContent = "English ASR final";
        zhCard.querySelector(".cue-zh").textContent = msg.text;
        zhCard = null;
      }}
      if (msg.type === "translation_start") trCard = addModelCard("active", "", "Chinese subtitle streaming");
      if (msg.type === "translation_partial") {{
        if (!trCard) trCard = addModelCard("active", "", "Chinese subtitle streaming");
        trCard.querySelector(".cue-zh").textContent = msg.text || msg.delta || "";
      }}
      if (msg.type === "translation_final") {{
        if (!trCard) trCard = addModelCard("active", msg.text || "", "Chinese subtitle final");
        trCard.classList.remove("active");
        trCard.querySelector(".cue-meta span").textContent = "Chinese subtitle final";
        trCard.querySelector(".cue-zh").textContent = msg.text || "";
        trCard = null;
      }}
      if (msg.type === "error") addModelCard("active", msg.message, "error");
    }};
    await new Promise((resolve, reject) => {{
      modelWs.onopen = resolve;
      modelWs.onerror = reject;
    }});
    audio.pause();
    audio.currentTime = 0;
    hasStarted = true;
    try {{
      await streamSharedAudioGraph();
    }} catch (err) {{
      addModelCard("active", `共享音频图不可用，降级到播放时钟同步：${{String(err)}}`, "system");
      await streamByPlaybackClock();
    }}
  }}
  async function streamSharedAudioGraph() {{
    addModelCard("active", "使用同一个 Web Audio 源同时输出到扬声器和模型：会议录音 -> shared source -> speaker + ASR。", "system");
    meetingAudioContext = meetingAudioContext || new AudioContext();
    const ctx = meetingAudioContext;
    if (ctx.state === "suspended") await ctx.resume();
    if (!meetingAudioSource) meetingAudioSource = ctx.createMediaElementSource(audio);
    const processorNode = ctx.createScriptProcessor(4096, 1, 1);
    const silentSink = ctx.createGain();
    silentSink.gain.value = 0;
    let sentSamples = 0;
    processorNode.onaudioprocess = e => {{
      if (!modelWs || modelWs.readyState !== WebSocket.OPEN || audio.paused || audio.ended) return;
      const input = e.inputBuffer.getChannelData(0);
      const pcm = floatTo16BitPCM(downsample(input, ctx.sampleRate, 16000));
      if (pcm.length) {{
        modelWs.send(pcm.buffer);
        sentSamples += pcm.length;
        updateModelSync("共享 Web Audio 源", sentSamples / 16000);
      }}
    }};
    if (!meetingAudioOutputConnected) {{
      meetingAudioSource.connect(ctx.destination);
      meetingAudioOutputConnected = true;
    }}
    meetingAudioSource.connect(processorNode);
    processorNode.connect(silentSink);
    silentSink.connect(ctx.destination);
    try {{
      await audio.play();
      await new Promise(resolve => {{
        const done = () => resolve();
        audio.addEventListener("ended", done, {{ once: true }});
        modelWs.addEventListener("close", done, {{ once: true }});
      }});
    }} finally {{
      processorNode.disconnect();
      silentSink.disconnect();
      if (modelWs && modelWs.readyState === WebSocket.OPEN) modelWs.send(JSON.stringify({{type:"stop"}}));
    }}
  }}
  async function streamByPlaybackClock() {{
    addModelCard("active", "当前浏览器不支持 audio.captureStream()，降级为按 audio.currentTime 发送；不会发送当前播放时间之后的音频。", "system");
    const response = await fetch(audio.currentSrc || audio.querySelector("source").src);
    const wavBytes = await response.arrayBuffer();
    const ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(wavBytes.slice(0));
    const mono = decoded.numberOfChannels > 1 ? (() => {{
      const left = decoded.getChannelData(0);
      const right = decoded.getChannelData(1);
      const mixed = new Float32Array(left.length);
      for (let i = 0; i < mixed.length; i++) mixed[i] = (left[i] + right[i]) / 2;
      return mixed;
    }})() : decoded.getChannelData(0);
    const pcm = floatTo16BitPCM(downsample(mono, decoded.sampleRate, 16000));
    let sent = 0;
    await audio.play();
    while (!audio.ended && modelWs.readyState === WebSocket.OPEN) {{
      const allowed = Math.min(pcm.length, Math.floor((audio.currentTime || 0) * 16000));
      while (sent + 1600 <= allowed && modelWs.readyState === WebSocket.OPEN) {{
        modelWs.send(pcm.slice(sent, sent + 1600).buffer);
        sent += 1600;
        updateModelSync("currentTime 时钟降级", sent / 16000);
      }}
      await new Promise(r => setTimeout(r, 50));
    }}
    if (modelWs.readyState === WebSocket.OPEN) {{
      const remaining = pcm.slice(sent, Math.min(pcm.length, Math.floor((audio.currentTime || 0) * 16000)));
      if (remaining.length) modelWs.send(remaining.buffer);
      modelWs.send(JSON.stringify({{type:"stop"}}));
    }}
    await ctx.close();
  }}
  startBtn.addEventListener("click", async () => {{
    hasStarted = true;
    await audio.play();
    startBtn.textContent = "正在播放：字幕实时生成中";
  }});
  modelStreamBtn.addEventListener("click", async () => {{
    modelStreamBtn.disabled = true;
    try {{
      await streamDemoAudioToModel();
    }} catch (err) {{
      addModelCard("active", String(err), "error");
    }} finally {{
      revealMinutes();
      modelStreamBtn.disabled = false;
    }}
  }});
  minutesBtn.addEventListener("click", () => {{
    minutesHint.textContent = "会议纪要已展开。播放完成后会自动展开；也可以随时手动查看。";
    revealMinutes();
  }});
</script>
""",
        height=820,
    )


def render_legacy_microphone_realtime_panel() -> None:
    backend_ready = ensure_realtime_backend()
    status = "实时后端已启动" if backend_ready else "实时后端启动失败，请查看 logs/realtime.err.log"
    sync_initial = (
        "尚未开始。当前数据集回放载入了已保存的会议级校正标签，用于验证实时字幕与会后结果的一致性。"
        if reference_meeting_id
        else "尚未开始。实时阶段使用无先验在线声纹聚类；播放结束后自动运行会议级校正。"
    )
    ws_url = f"ws://127.0.0.1:{REALTIME_PORT}/ws/realtime"
    components.html(
        f"""
<div class="mic-shell">
  <section class="mic-hero">
    <div>
      <div class="eyebrow">Real Microphone Stream</div>
      <h2>本地麦克风中文输入 → 英文实时字幕</h2>
      <p>当前页面直接采集浏览器麦克风，将 16k PCM 流发送到本地后端，再调用阿里实时 ASR 和 Qwen 流式翻译输出英文。</p>
    </div>
    <div class="status-pill" id="backendStatus">{status}</div>
  </section>
  <div class="controls">
    <button id="start">开始实时会议</button>
    <button id="stop" disabled>结束会议并保存</button>
    <button id="clear">清空字幕</button>
    <input id="meetingTitle" class="meeting-title" value="实时麦克风会议" aria-label="会议标题">
    <label class="silence-control">
      停顿切句阈值
      <input id="silenceMs" type="range" min="800" max="6000" step="100" value="2200">
      <span id="silenceValue">2200ms</span>
    </label>
  </div>
  <div id="status" class="status">请点击开始，并允许浏览器使用麦克风。</div>
  <div class="live-grid">
    <section class="panel">
      <h3>中文 ASR 真流式输出</h3>
      <div id="zh" class="stream"><div class="empty">等待中文语音输入...</div></div>
    </section>
    <section class="panel">
      <h3>英文字幕真流式输出</h3>
      <div id="en" class="stream"><div class="empty">等待英文翻译输出...</div></div>
    </section>
  </div>
  <section class="panel transcript-panel">
    <h3>本次实时会议记录</h3>
    <div id="records" class="records"><div class="empty">识别完成的句子会记录在这里。</div></div>
  </section>
  <section class="panel transcript-panel">
    <h3>阶段摘要与会后纪要</h3>
    <div id="summaries" class="records"><div class="empty">会议开始后每 2 分钟生成一次阶段摘要；结束会议后保存录音并生成最终纪要。</div></div>
  </section>
</div>
<style>
  .mic-shell {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #0f172a;
    background: #f8fafc;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    overflow: hidden;
  }}
  .mic-hero {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    padding: 24px;
    color: white;
    background: linear-gradient(135deg, #0f766e, #111827);
  }}
  .eyebrow {{ color: #99f6e4; font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
  h2 {{ margin: 6px 0 8px; font-size: 28px; }}
  h3 {{ margin: 0; padding: 14px 16px; background: #f1f5f9; border-bottom: 1px solid #e2e8f0; font-size: 17px; }}
  p {{ margin: 0; color: #d9f2ef; max-width: 760px; }}
  .status-pill {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.24); border-radius: 999px; padding: 8px 12px; white-space: nowrap; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 18px 8px; background: white; }}
  button {{ border: 0; border-radius: 10px; padding: 12px 16px; font-size: 15px; font-weight: 800; cursor: pointer; }}
  #start {{ background: #0f766e; color: white; }}
  #stop, #clear {{ background: #e2e8f0; color: #0f172a; }}
  button:disabled {{ opacity: .45; cursor: not-allowed; }}
  .silence-control {{ display: inline-flex; align-items: center; gap: 8px; color: #475569; font-size: 13px; font-weight: 700; }}
  .silence-control input {{ width: 170px; }}
  .meeting-title {{ border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 12px; min-width: 190px; font: inherit; }}
  .status {{ padding: 0 18px 14px; color: #475569; background: white; }}
  .live-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 16px 18px; }}
  .panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }}
  .stream {{ min-height: 360px; max-height: 430px; overflow-y: auto; padding: 12px; scroll-behavior: smooth; }}
  .transcript-panel {{ margin: 0 18px 18px; }}
  .records {{ max-height: 220px; overflow-y: auto; padding: 12px; }}
  .empty {{ padding: 34px 14px; color: #64748b; text-align: center; border: 1px dashed #cbd5e1; border-radius: 12px; background: #f8fafc; }}
  .card {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #f8fafc; line-height: 1.65; }}
  .partial {{ border-color: #f59e0b; background: #fffbeb; }}
  .final {{ border-color: #0f766e; background: #ecfdf5; }}
  .meta {{ display: flex; justify-content: space-between; gap: 10px; margin-bottom: 6px; color: #64748b; font-size: 12px; }}
  .text {{ font-size: 18px; font-weight: 750; white-space: pre-wrap; }}
  .record {{ border-bottom: 1px solid #e2e8f0; padding: 10px 2px; }}
  .record:last-child {{ border-bottom: 0; }}
  .record b {{ color: #0f766e; }}
  .record small {{ display: block; color: #64748b; margin-top: 3px; }}
  @media (max-width: 820px) {{ .mic-hero, .live-grid {{ display: block; }} .panel {{ margin-bottom: 14px; }} }}
</style>
<script>
  const WS_URL = "{ws_url}";
  const SAVE_URL = "http://127.0.0.1:{REALTIME_PORT}/api/realtime/meeting/save";
  const STAGE_URL = "http://127.0.0.1:{REALTIME_PORT}/api/realtime/meeting/stage-summary";
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const clearBtn = document.getElementById("clear");
  const meetingTitle = document.getElementById("meetingTitle");
  const silenceMs = document.getElementById("silenceMs");
  const silenceValue = document.getElementById("silenceValue");
  const statusEl = document.getElementById("status");
  const zhEl = document.getElementById("zh");
  const enEl = document.getElementById("en");
  const recordsEl = document.getElementById("records");
  const summariesEl = document.getElementById("summaries");

  let ws, audioCtx, source, processor, mediaStream;
  let lastZhCard = null;
  let currentEnCard = null;
  let currentSource = "";
  let currentRawSource = "";
  let meetingStartMs = 0;
  let stageTimer = null;
  let stageIndex = 0;
  let recordedChunks = [];
  let recordedSampleRate = 16000;
  let records = [];
  let activeSpeaker = "参会者识别中";

  function esc(s) {{
    return (s || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
  }}
  function now() {{ return new Date().toLocaleTimeString(); }}
  function resetEmpty(root, text) {{
    root.innerHTML = `<div class="empty">${{esc(text)}}</div>`;
  }}
  function removeEmpty(root) {{
    const empty = root.querySelector(".empty");
    if (empty) empty.remove();
  }}
  function addCard(root, cls, text, meta) {{
    removeEmpty(root);
    const card = document.createElement("div");
    card.className = `card ${{cls}}`;
    card.innerHTML = `<div class="meta"><span>${{esc(meta)}}</span><span>${{now()}}</span></div><div class="text">${{esc(text || "...")}}</div>`;
    root.appendChild(card);
    root.scrollTop = root.scrollHeight;
    return card;
  }}
  function setCard(card, cls, text, meta) {{
    if (!card) return;
    card.className = `card ${{cls}}`;
    card.querySelector(".meta span").textContent = meta;
    card.querySelector(".text").textContent = text || "...";
  }}
  function addRecord(source, english, speaker=activeSpeaker) {{
    removeEmpty(recordsEl);
    const raw = currentRawSource || source;
    const record = {{ source: raw, corrected: source, english, time: now(), speaker }};
    records.push(record);
    const div = document.createElement("div");
    div.className = "record";
    const correction = raw && raw !== source ? `<small>ASR 原文：${{esc(raw)}} → 已按上下文修正</small>` : "";
    div.innerHTML = `<div><b>${{esc(record.time)}}</b> <span class="speaker-name">${{esc(record.speaker)}}</span> 中文：${{esc(source)}}</div><div>英文：${{esc(english || "")}}</div>${{correction}}`;
    recordsEl.appendChild(div);
    recordsEl.scrollTop = recordsEl.scrollHeight;
  }}
  function addSummary(title, text, extra="") {{
    removeEmpty(summariesEl);
    const div = document.createElement("div");
    div.className = "record";
    div.innerHTML = `<div><b>${{esc(title)}}</b></div><div>${{esc(text || "")}}</div>${{extra}}`;
    summariesEl.appendChild(div);
    summariesEl.scrollTop = summariesEl.scrollHeight;
  }}
  function appendRecordedPcm(pcm) {{
    recordedChunks.push(new Int16Array(pcm));
  }}
  function encodeWav(chunks, sampleRate) {{
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const pcm = new Int16Array(length);
    let offset = 0;
    chunks.forEach(chunk => {{ pcm.set(chunk, offset); offset += chunk.length; }});
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    function writeString(pos, str) {{ for (let i = 0; i < str.length; i++) view.setUint8(pos + i, str.charCodeAt(i)); }}
    writeString(0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, pcm.length * 2, true);
    let pos = 44;
    for (let i = 0; i < pcm.length; i++, pos += 2) view.setInt16(pos, pcm[i], true);
    return new Blob([view], {{type: "audio/wav"}});
  }}
  function blobToDataUrl(blob) {{
    return new Promise(resolve => {{
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    }});
  }}
  async function generateStageSummary(force=false) {{
    if (!records.length) return;
    const elapsed = Math.floor(((Date.now() - meetingStartMs) || 0) / 1000);
    if (!force && elapsed < (stageIndex + 1) * 120) return;
    const from = stageIndex * 120;
    const to = force ? elapsed : (stageIndex + 1) * 120;
    stageIndex += 1;
    const windowLabel = `${{Math.floor(from / 60)}}-${{Math.max(1, Math.ceil(to / 60))}}分钟`;
    addSummary(`${{windowLabel}} 阶段摘要`, "正在生成...");
    try {{
      const res = await fetch(STAGE_URL, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{window: windowLabel, transcript: records}})
      }});
      const data = await res.json();
      const summary = data.summary || {{}};
      const last = summariesEl.querySelector(".record:last-child div:nth-child(2)");
      if (last) last.textContent = summary.summary || data.error || "暂无摘要";
    }} catch (err) {{
      const last = summariesEl.querySelector(".record:last-child div:nth-child(2)");
      if (last) last.textContent = `阶段摘要生成失败：${{err}}`;
    }}
  }}
  function floatTo16BitPCM(float32) {{
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {{
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }}
    return out;
  }}
  function downsample(buffer, inRate, outRate) {{
    if (outRate === inRate) return buffer;
    const ratio = inRate / outRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {{
      const start = Math.floor(i * ratio);
      const end = Math.floor((i + 1) * ratio);
      let sum = 0, count = 0;
      for (let j = start; j < end && j < buffer.length; j++) {{ sum += buffer[j]; count++; }}
      result[i] = count ? sum / count : 0;
    }}
    return result;
  }}
  async function start() {{
    records = [];
    activeSpeaker = "参会者识别中";
    recordedChunks = [];
    stageIndex = 0;
    meetingStartMs = Date.now();
    resetEmpty(summariesEl, "会议进行中；每 2 分钟生成一次阶段摘要，结束后保存录音并生成最终纪要。");
    statusEl.textContent = "正在连接实时后端...";
    const url = `${{WS_URL}}?source_lang=zh&target_lang=en&max_sentence_silence=${{encodeURIComponent(silenceMs.value)}}`;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.onmessage = ev => {{
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") statusEl.textContent = `已连接 ${{msg.asr_model}}，停顿切句阈值 ${{msg.max_sentence_silence}}ms，现在可以直接说中文。`;
      if (msg.type === "speaker_update") {{
        if (msg.speaker) activeSpeaker = msg.speaker;
        const state = msg.status === "new" ? "已发现新参会者" : msg.status === "matched" ? "声纹已匹配" : "正在识别声纹";
        statusEl.textContent = `${{state}}：${{activeSpeaker}}`;
      }}
      if (msg.type === "asr_open") statusEl.textContent = "ASR 已打开，正在接收麦克风音频。";
      if (msg.type === "asr_partial") {{
        if (!lastZhCard) lastZhCard = addCard(zhEl, "partial", msg.text, `${{activeSpeaker}} · ASR partial`);
        else setCard(lastZhCard, "partial", msg.text, `${{activeSpeaker}} · ASR partial`);
      }}
      if (msg.type === "asr_final") {{
        const speaker = msg.speaker || activeSpeaker;
        if (lastZhCard) setCard(lastZhCard, "final", msg.text, `${{speaker}} · ASR final`);
        else lastZhCard = addCard(zhEl, "final", msg.text, `${{speaker}} · ASR final`);
        lastZhCard = null;
      }}
      if (msg.type === "correction_start") {{
        statusEl.textContent = "正在结合上下文修正 ASR 细节...";
      }}
      if (msg.type === "correction_final") {{
        currentRawSource = msg.source || "";
        if (msg.changed) addCard(zhEl, "final", msg.corrected || msg.source || "", "上下文纠错后文本");
      }}
      if (msg.type === "translation_start") {{
        currentSource = msg.source || "";
        currentRawSource = msg.raw_source || currentRawSource || currentSource;
        currentEnCard = addCard(enEl, "partial", "", "English subtitle streaming");
      }}
      if (msg.type === "translation_partial") {{
        if (!currentEnCard) currentEnCard = addCard(enEl, "partial", "", "English subtitle streaming");
        setCard(currentEnCard, "partial", msg.text || msg.delta || "", "English subtitle streaming");
      }}
      if (msg.type === "translation_final") {{
        if (!currentEnCard) currentEnCard = addCard(enEl, "final", msg.text || "", "English subtitle final");
        setCard(currentEnCard, "final", msg.text || "", "English subtitle final");
        addRecord(msg.source || currentSource, msg.text || "", msg.speaker || activeSpeaker);
        generateStageSummary(false);
        currentEnCard = null;
        currentSource = "";
        currentRawSource = "";
      }}
      if (msg.type === "error") statusEl.textContent = `错误：${{msg.message}}`;
    }};
    ws.onerror = () => {{ statusEl.textContent = "WebSocket 连接失败，请确认实时后端已启动。"; }};
    ws.onclose = () => {{ statusEl.textContent = "实时连接已关闭。"; }};
    ws.onopen = async () => {{
      mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }} }});
      audioCtx = new AudioContext();
      source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = e => {{
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm = floatTo16BitPCM(downsample(input, audioCtx.sampleRate, 16000));
        appendRecordedPcm(pcm);
        ws.send(pcm.buffer);
      }};
      source.connect(processor);
      processor.connect(audioCtx.destination);
      startBtn.disabled = true;
      stopBtn.disabled = false;
      stageTimer = setInterval(() => generateStageSummary(false), 15000);
    }};
  }}
  async function stop() {{
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    if (audioCtx) await audioCtx.close();
    if (ws && ws.readyState === WebSocket.OPEN) {{
      ws.send(JSON.stringify({{type:"stop"}}));
      ws.close();
    }}
    if (stageTimer) clearInterval(stageTimer);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = "已停止，正在保存录音并生成会议纪要...";
    await generateStageSummary(true);
    try {{
      if (!recordedChunks.length) throw new Error("没有录到音频");
      const wavBlob = encodeWav(recordedChunks, recordedSampleRate);
      const audioB64 = await blobToDataUrl(wavBlob);
      const durationSeconds = Math.max(0, Math.round((Date.now() - meetingStartMs) / 1000));
      const res = await fetch(SAVE_URL, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          title: meetingTitle.value || "实时麦克风会议",
          audio_b64: audioB64,
          transcript: records,
          duration_seconds: durationSeconds
        }})
      }});
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "保存失败");
      const fm = (data.minutes && data.minutes.final_minutes) || {{}};
      addSummary("最终会议纪要", fm.one_sentence_summary || "会议已保存。", `<small>已保存到会议库：${{esc(data.meeting.title)}}；说话人状态：后处理声纹待回写。</small>`);
      statusEl.textContent = `会议已保存到会议库：${{data.meeting.title}}`;
    }} catch (err) {{
      addSummary("保存失败", String(err));
      statusEl.textContent = `保存失败：${{err}}`;
    }}
  }}
  startBtn.onclick = start;
  stopBtn.onclick = stop;
  silenceMs.oninput = () => {{
    silenceValue.textContent = `${{silenceMs.value}}ms`;
  }};
  clearBtn.onclick = () => {{
    resetEmpty(zhEl, "等待中文语音输入...");
    resetEmpty(enEl, "等待英文翻译输出...");
    resetEmpty(recordsEl, "识别完成的句子会记录在这里。");
    resetEmpty(summariesEl, "会议开始后每 2 分钟生成一次阶段摘要；结束会议后保存录音并生成最终纪要。");
    lastZhCard = null;
    currentEnCard = null;
    records = [];
    recordedChunks = [];
  }};
</script>
""",
        height=1000,
    )


def render_live_route_workspace(
    input_kind: str,
    title: str,
    description: str,
    source_lang: str,
    target_lang: str,
    audio_src: str = "",
    reference_meeting_id: str = "",
    reference_offset_seconds: float = 0.0,
) -> None:
    """Render the shared PCM bus used by recording playback and microphone input."""
    backend_ready = ensure_realtime_backend()
    status = "实时后端已启动" if backend_ready else "实时后端启动失败，请查看 logs/realtime.err.log"
    sync_initial = (
        "尚未开始。当前数据集回放载入了已保存的会议级校正标签，用于验证实时字幕与会后结果的一致性。"
        if reference_meeting_id
        else "尚未开始。实时阶段使用无先验在线声纹聚类；播放结束后自动运行会议级校正。"
    )
    audio_markup = (
        f'<audio id="inputAudio" controls preload="metadata"><source src="{audio_src}" type="audio/wav"></audio>'
        if input_kind == "recording"
        else ""
    )
    summary_markup = "" if input_kind == "microphone" else '<section class="panel meeting-minutes"><h3>会后会议纪要</h3><div id="minutes" class="records"><div class="empty">处理结束后将基于本次真实流式记录生成会议纪要。</div></div></section>'
    html = """
<div class="route-shell">
  <section class="route-hero">
    <div><div class="eyebrow">LIVE MEETING INTELLIGENCE</div><h2>__TITLE__</h2><p>__DESCRIPTION__</p></div>
    <div class="status-pill">__STATUS__</div>
  </section>
  <section class="route-toolbar">
    <div class="mode-group" aria-label="翻译路线">
      <label>处理路线</label>
      <select id="routeMode"><option value="cascade">级联：ASR + 上下文纠错 + 翻译</option><option value="e2e">端到端：实时语音翻译</option><option value="compare">对比：级联与端到端</option></select>
    </div>
    <div class="mode-group"><label>纪要依据</label><select id="canonicalRoute"><option value="cascade">级联路线</option><option value="e2e">端到端路线</option></select></div>
    <div class="mode-group silence"><label>停顿切句</label><input id="silenceMs" type="range" min="800" max="6000" step="100" value="2200"><span id="silenceLabel">2200ms</span></div>
    <input id="meetingTitle" value="__MEETING_TITLE__" aria-label="会议标题">
    <button id="start" class="primary">__START_LABEL__</button><button id="stop" disabled>结束并归档</button><button id="clear">清空</button>
  </section>
  <div class="sync-bar" id="syncStatus">__SYNC_INIT__</div>
  <section id="latencyCompare" class="latency-compare"></section>
  <div class="audio-wrap">__AUDIO__</div>
  <section id="routeGrid" class="route-grid"></section>
  <section class="panel record-panel"><h3>统一会议记录</h3><div id="records" class="records"><div class="empty">完成的双语句段会按所选纪要依据写入这里。</div></div></section>
  <section class="panel summary-panel"><h3>阶段摘要与最终纪要</h3><div id="summaries" class="records"><div class="empty">会议每两分钟生成阶段摘要；结束后生成最终纪要并保存。</div></div></section>
  __SUMMARY__
</div>
<style>
  .route-shell{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#10233e;background:#f6f9fc;border:1px solid #d9e4ef;border-radius:14px;overflow:hidden}
  .route-hero{display:flex;justify-content:space-between;gap:20px;padding:25px 26px;background:linear-gradient(118deg,#123b70,#087f91);color:white}.route-hero h2{margin:5px 0 8px;font-size:27px}.route-hero p{margin:0;max-width:760px;color:#dceffc;line-height:1.55}.eyebrow{color:#8fe9ef;font-size:11px;font-weight:800;letter-spacing:.12em}.status-pill{align-self:flex-start;padding:8px 10px;border:1px solid #77cbd4;border-radius:999px;background:#ffffff18;font-size:12px;white-space:nowrap}
  .route-toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:11px;padding:15px 18px;background:white;border-bottom:1px solid #dfe8f1}.mode-group{display:grid;gap:5px;color:#4d6078;font-size:12px;font-weight:750}.mode-group select,.route-toolbar input{height:36px;border:1px solid #cbd9e7;border-radius:7px;padding:0 9px;background:white;color:#10233e;font:inherit}.mode-group select{min-width:195px}.silence input{width:126px;padding:0}.route-toolbar>input{min-width:160px}.route-toolbar button{border:0;border-radius:7px;padding:10px 13px;background:#e7eef5;color:#17324f;font-weight:800;cursor:pointer}.route-toolbar .primary{background:#087f91;color:white}.route-toolbar button:disabled{opacity:.45;cursor:not-allowed}
  .sync-bar{margin:14px 18px 0;padding:10px 12px;border-radius:7px;background:#e8f6f8;color:#0d5967;font-size:12px;font-weight:700}.sync-bar.warn{background:#fff7e6;color:#8a5a00}.audio-wrap{padding:12px 18px 0}.audio-wrap:empty{display:none}.audio-wrap audio{width:100%}
  .latency-compare{display:none;margin:12px 18px 0;padding:13px 14px;border:1px solid #dce7ef;border-radius:9px;background:white}.latency-compare.visible{display:block}.latency-title{margin-bottom:10px;color:#193957;font-size:13px;font-weight:800}.latency-row{display:grid;grid-template-columns:106px 1fr 126px;gap:10px;align-items:center;margin:8px 0;color:#4e6780;font-size:12px}.latency-track{height:9px;overflow:hidden;border-radius:999px;background:#e7eef4}.latency-fill{height:100%;min-width:3px;border-radius:999px;background:#197f94}.latency-fill.e2e{background:#1d95a5}.latency-value{color:#173651;font-variant-numeric:tabular-nums;text-align:right}.latency-note{margin-top:8px;color:#718397;font-size:11px}
  .route-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding:15px 18px}.route-grid.single{grid-template-columns:minmax(0,1fr)}.route-panel,.panel{background:white;border:1px solid #dbe5ee;border-radius:10px;overflow:hidden}.route-panel{min-width:0}.route-panel.accent{border-color:#1698aa;box-shadow:0 8px 20px #0d76810e}.route-header{display:flex;justify-content:space-between;align-items:center;padding:13px 15px;border-bottom:1px solid #e3ebf1}.route-header h3,.panel h3{margin:0;color:#153453;font-size:16px}.route-header p{margin:3px 0 0;color:#6a7e93;font-size:12px}.route-badge,.speaker-chip{border-radius:999px;padding:4px 8px;background:#e8f6f8;color:#087f91;font-size:11px;font-weight:800}.speaker-chip{background:#e8f1ff;color:#22558a;white-space:nowrap}.stream{height:188px;min-height:150px;max-height:220px;overflow-y:auto;overscroll-behavior:contain;padding:11px;background:#fbfdff;scrollbar-color:#a7becd transparent;scrollbar-width:thin}.stream+.stream{border-top:1px solid #e3ebf1}.stream-label{position:sticky;top:-11px;z-index:1;margin:-11px -11px 8px;padding:9px 11px 7px;background:#fbfdff;color:#698096;font-size:11px;font-weight:800;letter-spacing:.04em}.card{border:1px solid #dce7ef;border-radius:7px;margin-bottom:9px;padding:9px 10px;background:white}.card.partial{border-color:#e7bd5a;background:#fffbed}.card.final{border-color:#8ed0bd;background:#f1fbf7}.meta{display:flex;align-items:center;gap:8px;color:#667e92;font-size:11px}.card-kind{margin-right:auto}.text{margin-top:5px;color:#17324f;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}.target .text{font-size:16px;font-weight:700}.empty{padding:26px 14px;border:1px dashed #cbd8e4;border-radius:7px;background:#fbfdff;color:#71859a;text-align:center}.metrics{display:flex;flex-wrap:wrap;gap:8px;padding:10px 14px;background:#f7fafc;border-top:1px solid #e2eaf1;color:#536c83;font-size:11px}.metric{padding:3px 6px;border-radius:5px;background:white;border:1px solid #dce6ee}.record-panel,.summary-panel,.meeting-minutes{margin:0 18px 15px}.panel h3{padding:13px 15px;background:#f7fafc;border-bottom:1px solid #e1e9f0}.records{max-height:270px;overflow-y:auto;padding:11px}.record{padding:9px 2px;border-bottom:1px solid #e4ebf1;line-height:1.5}.record:last-child{border-bottom:0}.record b{color:#087f91}.record small{display:block;color:#708397}.route-hidden{display:none}
  @media(max-width:720px){.route-hero{display:block}.status-pill{display:inline-block;margin-top:14px}.route-grid{grid-template-columns:1fr}.route-toolbar{align-items:stretch}.route-toolbar>input,.mode-group select{width:100%;box-sizing:border-box}.mode-group{flex:1 1 190px}}
</style>
<script>
const INPUT_KIND="__INPUT_KIND__", WS_BASE="__WS_BASE__", SAVE_URL="__SAVE_URL__", DIARIZE_URL="__DIARIZE_URL__", STAGE_URL="__STAGE_URL__", SOURCE_LANG="__SOURCE_LANG__", TARGET_LANG="__TARGET_LANG__", REFERENCE_MEETING_ID="__REFERENCE_MEETING_ID__", REFERENCE_OFFSET_SECONDS="__REFERENCE_OFFSET_SECONDS__";
const startBtn=document.getElementById("start"),stopBtn=document.getElementById("stop"),clearBtn=document.getElementById("clear"),modeEl=document.getElementById("routeMode"),canonicalEl=document.getElementById("canonicalRoute"),silenceEl=document.getElementById("silenceMs"),silenceLabel=document.getElementById("silenceLabel"),grid=document.getElementById("routeGrid"),syncEl=document.getElementById("syncStatus"),latencyEl=document.getElementById("latencyCompare"),recordsEl=document.getElementById("records"),summariesEl=document.getElementById("summaries"),minutesEl=document.getElementById("minutes"),audio=document.getElementById("inputAudio");
let sockets={},streaming=false,audioCtx,sourceNode,processor,silentGain,mediaStream,recordingOutputConnected=false,activeSpeaker="参会者识别中",currentTurn=0,startedAt=0,sentSamples=0,frameCount=0,frameHash=2166136261,records=[],routeRecords={cascade:[],e2e:[]},routeState={},stageTimer,stageIndex=0,recordedChunks=[],turnAudioStarts={0:0},turnAudioEnds={},reconciledSpeakers={};
function esc(v){return String(v||"").replace(/[&<>"']/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":c.charCodeAt(0)===34?"&quot;":"&#39;");}
function activeRoutes(){return modeEl.value==="compare"?["cascade","e2e"]:[modeEl.value];}
function connectionRoutes(){return modeEl.value==="compare"?["compare"]:activeRoutes();}
function routeName(r){return r==="cascade"?"级联路线 · C3":"端到端路线 · C4";}
function routeDetail(r){return r==="cascade"?"实时 ASR → 上下文修正 → 翻译":"Qwen LiveTranslate 直接语音翻译";}
function resetEmpty(root,text){root.innerHTML='<div class="empty">'+esc(text)+'</div>';}
function removeEmpty(root){const e=root.querySelector(".empty");if(e)e.remove();}
function addCard(root,cls,text,meta,speaker=activeSpeaker,turnId=currentTurn){removeEmpty(root);const d=document.createElement("div");d.className="card "+cls;d.dataset.turn=String(turnId);const pending=speaker==="参会者识别中"?' data-pending-speaker="true"':"";d.innerHTML='<div class="meta"><span class="speaker-chip"'+pending+'>'+esc(speaker)+'</span><span class="card-kind">'+esc(meta)+'</span><span>'+new Date().toLocaleTimeString()+'</span></div><div class="text">'+esc(text||"...")+'</div>';root.appendChild(d);root.scrollTop=root.scrollHeight;return d;}
function updateCard(card,cls,text,meta,speaker=activeSpeaker){if(!card)return;card.className="card "+cls;const chip=card.querySelector(".speaker-chip");chip.textContent=speaker;if(speaker!=="参会者识别中")chip.removeAttribute("data-pending-speaker");card.querySelector(".card-kind").textContent=meta;card.querySelector(".text").textContent=text||"...";}
function renderRoutes(){const routes=activeRoutes();grid.className="route-grid "+(routes.length===1?"single":"");grid.innerHTML=routes.map(r=>'<section class="route-panel '+(r==="e2e"?"accent":"")+'" data-route="'+r+'"><div class="route-header"><div><h3>'+routeName(r)+'</h3><p>'+routeDetail(r)+'</p></div><span class="route-badge">等待连接</span></div><div class="stream" id="source-'+r+'"><div class="stream-label">'+(SOURCE_LANG==="zh"?"原文 / 中文":"SOURCE / ENGLISH")+'</div><div class="empty">等待真实音频帧...</div></div><div class="stream target" id="target-'+r+'"><div class="stream-label">'+(TARGET_LANG==="zh"?"翻译 / 中文":"TRANSLATION / ENGLISH")+'</div><div class="empty">等待模型输出...</div></div><div class="metrics" id="metrics-'+r+'"></div></section>').join("");routes.forEach(r=>routeState[r]={source:"",raw:"",sourceCard:null,targetCard:null,first:null,final:null});refreshMetrics();}
function setBadge(r,text){const el=document.querySelector('[data-route="'+r+'"] .route-badge');if(el)el.textContent=text;}
function elapsed(v){return v==null?"--":((v-startedAt)/1000).toFixed(2)+"s";}
function routeAgreement(){if(modeEl.value!=="compare")return "--";const a=(routeRecords.cascade||[]).map(x=>x.translation||"").join("").replace(/\\s/g,""),b=(routeRecords.e2e||[]).map(x=>x.translation||"").join("").replace(/\\s/g,"");if(!a||!b)return "--";const left=new Set(Array.from(a)),right=new Set(Array.from(b));let same=0;left.forEach(ch=>{if(right.has(ch))same++;});return Math.round(100*same/Math.max(1,new Set([...left,...right]).size))+"%";}
function renderLatencyCompare(){if(modeEl.value!=="compare"){latencyEl.className="latency-compare";latencyEl.innerHTML="";return;}const rows=[{route:"cascade",label:"C3 级联"},{route:"e2e",label:"C4 端到端"}],values=rows.map(x=>{const s=routeState[x.route]||{};return Math.max(0,s.final?s.final-startedAt:s.first?s.first-startedAt:0);}),max=Math.max(1,...values);latencyEl.className="latency-compare visible";latencyEl.innerHTML='<div class="latency-title">双路线真实延迟对比</div>'+rows.map((x,i)=>{const s=routeState[x.route]||{},first=elapsed(s.first),final=elapsed(s.final),value=values[i],width=value?Math.max(4,Math.round(100*value/max)):0;return '<div class="latency-row"><b>'+x.label+'</b><div class="latency-track"><div class="latency-fill '+(x.route==="e2e"?"e2e":"")+'" style="width:'+width+'%"></div></div><div class="latency-value">首 '+first+' · 最终 '+final+'</div></div>';}).join("")+'<div class="latency-note">从首个共享 PCM 帧开始计时；最终译文延迟包含当前句段结束与模型收尾时间。</div>';}
function refreshMetrics(){const play=audio?(audio.currentTime||0):sentSamples/16000;const sent=sentSamples/16000;const drift=sent-play;syncEl.className="sync-bar"+(Math.abs(drift)>.35?" warn":"");syncEl.textContent='共享帧总线：'+frameCount+' 帧 · '+sent.toFixed(2)+'s 16 kHz PCM · FNV '+(frameHash>>>0).toString(16)+' · '+(INPUT_KIND==="recording"?'播放器':'麦克风')+'时间 '+play.toFixed(2)+'s · 差值 '+(drift>=0?"+":"")+drift.toFixed(2)+'s。每条激活路线收到同一帧副本。';activeRoutes().forEach(r=>{const m=document.getElementById("metrics-"+r);if(m){const s=routeState[r]||{};m.innerHTML='<span class="metric">首文本 '+elapsed(s.first)+'</span><span class="metric">最终译文 '+elapsed(s.final)+'</span><span class="metric">共享 '+frameCount+' 帧</span><span class="metric">FNV '+(frameHash>>>0).toString(16)+'</span>'+(modeEl.value==="compare"?'<span class="metric">结果一致度 '+routeAgreement()+'</span>':'');}});renderLatencyCompare();}
function hashFrame(pcm){for(let i=0;i<pcm.length;i+=19){frameHash^=(pcm[i]&255);frameHash=Math.imul(frameHash,16777619);}}
function floatToPcm(f){const out=new Int16Array(f.length);for(let i=0;i<f.length;i++){const s=Math.max(-1,Math.min(1,f[i]));out[i]=s<0?s*32768:s*32767;}return out;}
function downsample(buffer,inRate,outRate){if(inRate===outRate)return buffer;const ratio=inRate/outRate,n=Math.round(buffer.length/ratio),out=new Float32Array(n);for(let i=0;i<n;i++){const a=Math.floor(i*ratio),b=Math.floor((i+1)*ratio);let sum=0,count=0;for(let j=a;j<b&&j<buffer.length;j++){sum+=buffer[j];count++;}out[i]=count?sum/count:0;}return out;}
function sendFrame(pcm){if(!streaming||!pcm.length)return;recordedChunks.push(new Int16Array(pcm));sentSamples+=pcm.length;frameCount++;hashFrame(pcm);connectionRoutes().forEach(r=>{const ws=sockets[r];if(ws&&ws.readyState===WebSocket.OPEN)ws.send(pcm.buffer.slice(0));});refreshMetrics();}
function playbackSeconds(){return INPUT_KIND==="recording"&&audio?Math.max(0,Number(audio.currentTime||0)):sentSamples/16000;}
function syncTurnTimes(turnId){const start=Number(turnAudioStarts[turnId]),end=Number(turnAudioEnds[turnId]);if(!Number.isFinite(start)||!Number.isFinite(end))return;[records,routeRecords.cascade,routeRecords.e2e].forEach(rows=>(rows||[]).forEach(row=>{if(Number(row.turn_id)!==Number(turnId))return;row.audio_start_seconds=Number(start.toFixed(3));row.audio_end_seconds=Number(Math.max(start,end).toFixed(3));}));recordsEl.querySelectorAll(".record").forEach(node=>{if(String(node.dataset.turn||"")!==String(turnId))return;const row=records.find(item=>Number(item.turn_id)===Number(turnId)),small=node.querySelector("small");if(row&&small)small.textContent=routeName(row.route)+" · "+row.audio_start_seconds+"-"+row.audio_end_seconds+"s";});}
function finalizePlaybackTimes(){if(INPUT_KIND!=="recording")return;const end=playbackSeconds();if(Number.isFinite(Number(currentTurn))){turnAudioEnds[currentTurn]=end;syncTurnTimes(currentTurn);}}
function recordSegment(route,source,translation,speaker,turnId=currentTurn){const inputEnd=INPUT_KIND==="recording"&&audio?Math.min(sentSamples/16000,playbackSeconds()||sentSamples/16000):sentSamples/16000;const previous=(routeRecords[route]||[]).slice(-1)[0],fallbackStart=previous&&Number.isFinite(Number(previous.audio_end_seconds))?Number(previous.audio_end_seconds):Math.max(0,inputEnd-2.5),turnStart=Number(turnAudioStarts[turnId]),inputStart=Number.isFinite(turnStart)?turnStart:fallbackStart,turnEnd=Number(turnAudioEnds[turnId]),fixedEnd=Number.isFinite(turnEnd)?turnEnd:inputEnd;const item={route,speaker:speaker||activeSpeaker,turn_id:turnId,audio_start_seconds:Number(inputStart.toFixed(3)),audio_end_seconds:Number(Math.max(inputStart,fixedEnd).toFixed(3)),source,corrected:source,english:TARGET_LANG==="en"?translation:"",translation,time:new Date().toLocaleTimeString()};routeRecords[route].push(item);if(route!==canonicalEl.value)return;removeEmpty(recordsEl);records.push(item);const d=document.createElement("div");d.className="record";d.dataset.turn=String(turnId);d.innerHTML='<b>'+esc(item.time)+' · '+esc(item.speaker)+'</b><div>'+esc(source)+'</div><div>'+esc(translation)+'</div><small>'+routeName(route)+' · '+item.audio_start_seconds+'-'+item.audio_end_seconds+'s</small>';recordsEl.appendChild(d);recordsEl.scrollTop=recordsEl.scrollHeight;}
function addSummary(title,text,extra=""){removeEmpty(summariesEl);const d=document.createElement("div");d.className="record";d.innerHTML='<b>'+esc(title)+'</b><div>'+esc(text||"")+'</div>'+extra;summariesEl.appendChild(d);summariesEl.scrollTop=summariesEl.scrollHeight;}
function onEvent(route,msg){const st=routeState[route];if(!st)return;if(msg.type==="ready"){const vad=msg.speaker_vad?" · 声纹 "+msg.speaker_vad:"";setBadge(route,"音频通道已连接"+vad);return;}if(msg.type==="e2e_ready"){setBadge(route,"LiveTranslate 就绪");return;}if(msg.type==="speaker_turn_start"){currentTurn=Number(msg.turn_id||currentTurn+1);activeSpeaker="参会者识别中";Object.values(routeState).forEach(lane=>{lane.sourceCard=null;lane.targetCard=null;});document.querySelectorAll(".route-badge").forEach(x=>x.textContent="新发言轮次 · 声纹识别中");return;}if(msg.type==="speaker_update"){if(msg.applies_to_active_turn===false)return;activeSpeaker=msg.speaker||activeSpeaker;document.querySelectorAll(".route-badge").forEach(x=>x.textContent=activeSpeaker+" · 声纹 "+(msg.status||"识别中"));const from=Number(msg.backfill_from_turn);if(msg.status==="provisional"||Number.isFinite(from)){const shouldFill=card=>msg.status==="provisional"||Number(card.dataset.turn||-1)>=from;document.querySelectorAll('.card .speaker-chip[data-pending-speaker="true"]').forEach(chip=>{const card=chip.closest(".card");if(card&&shouldFill(card)){chip.textContent=activeSpeaker;chip.removeAttribute("data-pending-speaker");}});[records,routeRecords.cascade,routeRecords.e2e].forEach(rows=>(rows||[]).forEach(row=>{if(row.speaker==="参会者识别中"&&(msg.status==="provisional"||Number(row.turn_id||-1)>=from))row.speaker=activeSpeaker;}));}return;}const src=document.getElementById("source-"+route),tgt=document.getElementById("target-"+route),speaker=msg.speaker||activeSpeaker,turnId=Number.isFinite(Number(msg.turn_id))?Number(msg.turn_id):currentTurn;if(["asr_partial","e2e_source_partial"].includes(msg.type)){const text=(msg.text||"")+(msg.stash||"");if(!st.first&&text)st.first=performance.now();if(!st.sourceCard)st.sourceCard=addCard(src,"partial",text,"原文实时输出",speaker,turnId);else updateCard(st.sourceCard,"partial",text,"原文实时输出",speaker);refreshMetrics();return;}if(["asr_final","e2e_source_final"].includes(msg.type)){st.source=msg.text||st.source;if(!st.sourceCard)st.sourceCard=addCard(src,"final",st.source,"原文最终句段",speaker,turnId);else updateCard(st.sourceCard,"final",st.source,"原文最终句段",speaker);st.sourceCard=null;return;}if(msg.type==="correction_final"){st.source=msg.corrected||msg.source||st.source;addCard(src,"final",st.source,msg.changed?"上下文修正后":"原文确认",speaker,turnId);return;}if(["translation_partial","e2e_translation_partial"].includes(msg.type)){const text=(msg.text||"")+(msg.stash||"");if(!st.first&&text)st.first=performance.now();if(!st.targetCard)st.targetCard=addCard(tgt,"partial",text,"翻译实时输出",speaker,turnId);else updateCard(st.targetCard,"partial",text,"翻译实时输出",speaker);refreshMetrics();return;}if(["translation_final","e2e_translation_final"].includes(msg.type)){const text=msg.text||"";st.final=performance.now();if(!st.targetCard)st.targetCard=addCard(tgt,"final",text,"翻译最终句段",speaker,turnId);else updateCard(st.targetCard,"final",text,"翻译最终句段",speaker);recordSegment(route,msg.corrected_source||msg.source||st.source,text,speaker,turnId);st.targetCard=null;refreshMetrics();return;}if(msg.type==="error"||msg.type==="e2e_error"){setBadge(route,"连接异常");addCard(tgt,"partial",msg.message||"模型错误","错误",speaker,turnId);}}
function connectRoute(route){return new Promise((resolve,reject)=>{const params=new URLSearchParams({route,source_lang:SOURCE_LANG,target_lang:TARGET_LANG,max_sentence_silence:silenceEl.value});const ws=new WebSocket(WS_BASE+"?"+params.toString());ws.binaryType="arraybuffer";sockets[route]=ws;ws.onmessage=e=>{const msg=JSON.parse(e.data);if(route==="compare"&&msg.type==="ready"){["cascade","e2e"].forEach(lane=>onEvent(lane,msg));return;}const lane=route==="compare"?(msg.route||(String(msg.type||"").startsWith("e2e_")?"e2e":"cascade")):route;onEvent(lane,msg);};ws.onopen=resolve;ws.onerror=()=>reject(new Error(route==="compare"?"比较会话连接失败":routeName(route)+"连接失败"));ws.onclose=()=>{if(streaming)(route==="compare"?["cascade","e2e"]:[route]).forEach(lane=>setBadge(lane,"通道已关闭"));};});}
async function setupInput(){if(INPUT_KIND==="recording"){audio.pause();audio.currentTime=0;audioCtx=audioCtx||new AudioContext();if(audioCtx.state==="suspended")await audioCtx.resume();if(!sourceNode)sourceNode=audioCtx.createMediaElementSource(audio);processor=audioCtx.createScriptProcessor(2048,1,1);silentGain=audioCtx.createGain();silentGain.gain.value=0;processor.onaudioprocess=e=>{if(!audio.paused&&!audio.ended)sendFrame(floatToPcm(downsample(e.inputBuffer.getChannelData(0),audioCtx.sampleRate,16000)));};if(!recordingOutputConnected){sourceNode.connect(audioCtx.destination);recordingOutputConnected=true;}sourceNode.connect(processor);processor.connect(silentGain);silentGain.connect(audioCtx.destination);await audio.play();audio.onended=()=>finish(false);return;}mediaStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});audioCtx=new AudioContext();sourceNode=audioCtx.createMediaStreamSource(mediaStream);processor=audioCtx.createScriptProcessor(2048,1,1);silentGain=audioCtx.createGain();silentGain.gain.value=0;processor.onaudioprocess=e=>sendFrame(floatToPcm(downsample(e.inputBuffer.getChannelData(0),audioCtx.sampleRate,16000)));sourceNode.connect(processor);processor.connect(silentGain);silentGain.connect(audioCtx.destination);}
function encodeWav(chunks){const n=chunks.reduce((a,c)=>a+c.length,0),pcm=new Int16Array(n);let off=0;chunks.forEach(c=>{pcm.set(c,off);off+=c.length;});const b=new ArrayBuffer(44+n*2),v=new DataView(b),w=(p,s)=>{for(let i=0;i<s.length;i++)v.setUint8(p+i,s.charCodeAt(i));};w(0,"RIFF");v.setUint32(4,36+n*2,true);w(8,"WAVE");w(12,"fmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,"data");v.setUint32(40,n*2,true);for(let i=0,p=44;i<n;i++,p+=2)v.setInt16(p,pcm[i],true);return new Blob([v],{type:"audio/wav"});}
function dataUrl(blob){return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result);f.readAsDataURL(blob);});}
function updateTurnSpeaker(turnId,speaker){if(!speaker||turnId==null)return;document.querySelectorAll(".card").forEach(card=>{if(String(card.dataset.turn||"")!==String(turnId))return;const chip=card.querySelector(".speaker-chip");if(chip){chip.textContent=speaker;chip.removeAttribute("data-pending-speaker");}const kind=card.querySelector(".card-kind");if(kind)kind.textContent="会后校正标签";});document.querySelectorAll(".record").forEach(node=>{if(String(node.dataset.turn||"")!==String(turnId))return;const label=node.querySelector("b");if(!label)return;const parts=label.textContent.split(" · ");label.textContent=(parts[0]||"")+" · "+speaker;});}
async function reconcilePlayback(){if(INPUT_KIND!=="recording"||!recordedChunks.length)return;try{syncEl.textContent="播放结束，正在运行本地会议级声纹校正；实时标签暂不代表最终结果...";const audioB64=await dataUrl(encodeWav(recordedChunks));const r=await fetch(DIARIZE_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({audio_b64:audioB64,transcript:records})}),d=await r.json();if(!r.ok||d.error)throw new Error(d.error||"声纹校正失败");const corrected=d.transcript||[],canonicalRows=routeRecords[canonicalEl.value]||[],speakerByTurn=new Map();corrected.forEach((row,i)=>{if(row.speaker){const key=String(row.turn_id==null?"":row.turn_id);if(key)speakerByTurn.set(key,row.speaker);if(records[i])records[i].speaker=row.speaker;if(canonicalRows[i])canonicalRows[i].speaker=row.speaker;updateTurnSpeaker(row.turn_id,row.speaker);}});Object.values(routeRecords).forEach(rows=>(rows||[]).forEach(row=>{const speaker=speakerByTurn.get(String(row.turn_id));if(speaker)row.speaker=speaker;}));recordsEl.querySelectorAll(".record").forEach((node,i)=>{const row=records[i],label=node.querySelector("b");if(row&&label)label.textContent=row.time+" · "+row.speaker;});const ds=d.diarization||{},labels=[...new Set(corrected.map(row=>row.speaker).filter(Boolean))],resolved=ds.status==="resolved";document.querySelectorAll(".route-badge").forEach(x=>x.textContent=labels.length?"会后校正 · "+labels.join(" / "):"会后校正完成");addSummary("会后声纹校正",resolved?"已完成整段录音的全局声纹校正，级联与端到端字幕卡片及统一会议记录均已回写最终参会者标签。":"已完成整段录音的声纹时间线，但当前字幕缺少可对齐的音频时间戳，因此保留实时临时标签。");syncEl.textContent="播放结束；"+(resolved?"会议级声纹校正已完成，字幕卡片已同步更新。":"已完成声纹分析，当前字幕缺少可对齐时间戳。");}catch(e){addSummary("声纹校正失败",String(e));syncEl.textContent="声纹校正失败："+e;}}
if(audio)audio.addEventListener("ended",()=>{if(INPUT_KIND==="recording")setTimeout(reconcilePlayback,1200);});
async function generateStage(force=false){if(!records.length)return;const elapsed=Math.floor((Date.now()-startedAt)/1000);if(!force&&elapsed<(stageIndex+1)*120)return;const from=stageIndex*120,to=force?elapsed:(stageIndex+1)*120;stageIndex++;const label=Math.floor(from/60)+"-"+Math.max(1,Math.ceil(to/60))+"分钟";addSummary(label+" 阶段摘要","正在生成...");try{const r=await fetch(STAGE_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({window:label,transcript:records})}),d=await r.json(),last=summariesEl.querySelector(".record:last-child div");if(last)last.textContent=(d.summary||{}).summary||d.error||"暂无摘要";}catch(e){addSummary(label+" 阶段摘要",String(e));}}
async function saveMicrophone(){if(INPUT_KIND!=="microphone"||!recordedChunks.length)return;try{const audioB64=await dataUrl(encodeWav(recordedChunks));const latency=Object.fromEntries(activeRoutes().map(route=>[route,{first_text_seconds:routeState[route]?.first?Number(((routeState[route].first-startedAt)/1000).toFixed(3)):null,final_translation_seconds:routeState[route]?.final?Number(((routeState[route].final-startedAt)/1000).toFixed(3)):null}]));syncEl.textContent="录音已停止，正在运行本地 FSMN-VAD + 会议级声纹校正，请稍候...";const r=await fetch(SAVE_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:document.getElementById("meetingTitle").value||"实时麦克风会议",audio_b64:audioB64,transcript:records,duration_seconds:sentSamples/16000,route_mode:modeEl.value,canonical_route:canonicalEl.value,route_transcripts:routeRecords,comparison_metrics:{frame_count:frameCount,pcm_seconds:sentSamples/16000,frame_hash:(frameHash>>>0).toString(16),routes:latency}})}),d=await r.json();if(!r.ok||d.error)throw new Error(d.error||"保存失败");const fm=(d.minutes||{}).final_minutes||{};const ds=d.diarization||{};const diarizationText=ds.status==="resolved"?"已完成会后声纹校正，会议记录已回写参会者标签。":ds.status==="audio_diarized"?"已完成录音声纹分析，但当前字幕缺少可对齐时间戳。":"声纹校正暂未完成，已保留实时临时标签。";addSummary("最终会议纪要",fm.one_sentence_summary||"会议已归档。",'<small>已保存到会议库：'+esc(d.meeting.title)+'；'+esc(diarizationText)+'</small>');syncEl.textContent="会议已保存；"+diarizationText;}catch(e){addSummary("保存失败",String(e));syncEl.textContent="保存失败："+e;}}
async function finish(save=true){if(!streaming)return;streaming=false;if(audio)audio.pause();if(processor)processor.disconnect();if(sourceNode&&INPUT_KIND==="microphone")sourceNode.disconnect();if(mediaStream)mediaStream.getTracks().forEach(t=>t.stop());connectionRoutes().forEach(r=>{const ws=sockets[r];if(ws&&ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify({type:"stop"}));});if(stageTimer)clearInterval(stageTimer);startBtn.disabled=false;stopBtn.disabled=true;syncEl.textContent="输入已停止，正在等待两条路线完成最后一个句段...";await new Promise(r=>setTimeout(r,900));await generateStage(true);if(minutesEl&&records.length){resetEmpty(minutesEl,"");addSummary("最终会议纪要", "已基于本次真实流式记录生成，可在上方阶段摘要查看。");}if(save)await saveMicrophone();setTimeout(()=>Object.values(sockets).forEach(ws=>{if(ws.readyState===WebSocket.OPEN)ws.close();}),18000);}
async function begin(){try{records=[];routeRecords={cascade:[],e2e:[]};activeSpeaker="参会者识别中";recordedChunks=[];sentSamples=0;frameCount=0;frameHash=2166136261;stageIndex=0;startedAt=performance.now();resetEmpty(recordsEl,"等待完成句段...");resetEmpty(summariesEl,"会议进行中；每两分钟生成阶段摘要。");if(minutesEl)resetEmpty(minutesEl,"真实音频流结束后生成会议纪要。");renderRoutes();await Promise.all(connectionRoutes().map(connectRoute));streaming=true;await setupInput();startBtn.disabled=true;stopBtn.disabled=false;syncEl.textContent="已启动共享 PCM 总线。音频只从当前输入源产生，未使用预生成字幕或预送音频。";stageTimer=setInterval(()=>generateStage(false),15000);}catch(e){streaming=false;syncEl.className="sync-bar warn";syncEl.textContent="启动失败："+e.message;Object.values(sockets).forEach(ws=>ws.close());}}
modeEl.onchange=()=>{canonicalEl.value=modeEl.value==="e2e"?"e2e":"cascade";renderRoutes();};canonicalEl.onchange=()=>{records=[];resetEmpty(recordsEl,"纪要依据已切换；后续最终句段将记录到这里。");};silenceEl.oninput=()=>silenceLabel.textContent=silenceEl.value+"ms";startBtn.onclick=begin;stopBtn.onclick=()=>finish(true);clearBtn.onclick=()=>{records=[];recordedChunks=[];resetEmpty(recordsEl,"识别完成的句子会记录在这里。");resetEmpty(summariesEl,"会议开始后每两分钟生成阶段摘要；结束后生成最终纪要。");activeRoutes().forEach(r=>{const a=document.getElementById("source-"+r),b=document.getElementById("target-"+r);if(a)resetEmpty(a,"等待真实音频帧...");if(b)resetEmpty(b,"等待模型输出...");});};renderRoutes();
const baseEventHandler=onEvent;
onEvent=function(route,msg){if(msg.type==="speaker_update"&&msg.status==="reconciled"){const turn=Number(msg.turn_id),speaker=msg.speaker;reconciledSpeakers[String(turn)]=speaker;[records,routeRecords.cascade,routeRecords.e2e].forEach(rows=>(rows||[]).forEach(row=>{if(Number(row.turn_id)===turn)row.speaker=speaker;}));updateTurnSpeaker(turn,speaker);document.querySelectorAll(".route-badge").forEach(x=>x.textContent=speaker+" · 会后增量校正");return;}const known=msg.turn_id!=null?reconciledSpeakers[String(msg.turn_id)]:null;if(known&&msg.type!=="speaker_turn_start")msg={...msg,speaker:known};if(msg.type==="speaker_turn_start"){const previousTurn=currentTurn,nextTurn=Number.isFinite(Number(msg.turn_id))?Number(msg.turn_id):previousTurn+1,boundary=playbackSeconds();if(nextTurn!==previousTurn){turnAudioEnds[previousTurn]=boundary;syncTurnTimes(previousTurn);}turnAudioStarts[nextTurn]=boundary;}baseEventHandler(route,msg);};
const baseConnectRoute=connectRoute;
connectRoute=function(route){return new Promise((resolve,reject)=>{const params=new URLSearchParams({route,source_lang:SOURCE_LANG,target_lang:TARGET_LANG,max_sentence_silence:silenceEl.value});if(REFERENCE_MEETING_ID){params.set("reference_meeting_id",REFERENCE_MEETING_ID);if(REFERENCE_OFFSET_SECONDS)params.set("reference_offset_seconds",REFERENCE_OFFSET_SECONDS);}const ws=new WebSocket(WS_BASE+"?"+params.toString());ws.binaryType="arraybuffer";sockets[route]=ws;ws.onmessage=e=>{const msg=JSON.parse(e.data);if(route==="compare"&&msg.type==="ready"){["cascade","e2e"].forEach(lane=>onEvent(lane,msg));return;}const lane=route==="compare"?(msg.route||(String(msg.type||"").startsWith("e2e_")?"e2e":"cascade")):route;onEvent(lane,msg);};ws.onopen=resolve;ws.onerror=()=>reject(new Error(route==="compare"?"比较会话连接失败":routeName(route)+"连接失败"));ws.onclose=()=>{if(streaming)(route==="compare"?["cascade","e2e"]:[route]).forEach(lane=>setBadge(lane,"通道已关闭"));};});};
const baseReferenceBadgeHandler=onEvent;
onEvent=function(route,msg){baseReferenceBadgeHandler(route,msg);if(msg.type==="ready"&&String(msg.speaker_profile||"").startsWith("meeting_calibrated"))setBadge(route,msg.speaker_profile==="meeting_calibrated_timeline"?"会议级校正标签已载入":"会议级声纹参考已载入");};
const originalDataUrl=dataUrl;
dataUrl=function(blob){if(INPUT_KIND==="recording"&&audio&&audio.currentSrc&&audio.currentSrc.startsWith("data:"))return Promise.resolve(audio.currentSrc);return originalDataUrl(blob);};
if(startBtn)startBtn.addEventListener("click",()=>{turnAudioStarts={0:0};turnAudioEnds={};reconciledSpeakers={};},{capture:true});
if(stopBtn)stopBtn.addEventListener("click",finalizePlaybackTimes,{capture:true});
if(audio)audio.addEventListener("ended",finalizePlaybackTimes);
</script>
"""
    html = (
        html.replace("__TITLE__", title)
        .replace("__DESCRIPTION__", description)
        .replace("__STATUS__", status)
        .replace("__AUDIO__", audio_markup)
        .replace("__SUMMARY__", summary_markup)
        .replace("__SYNC_INIT__", sync_initial)
        .replace("__INPUT_KIND__", input_kind)
        .replace("__WS_BASE__", f"ws://127.0.0.1:{REALTIME_PORT}/ws/realtime")
        .replace("__SAVE_URL__", f"http://127.0.0.1:{REALTIME_PORT}/api/realtime/meeting/save")
        .replace("__DIARIZE_URL__", f"http://127.0.0.1:{REALTIME_PORT}/api/realtime/meeting/diarize")
        .replace("__STAGE_URL__", f"http://127.0.0.1:{REALTIME_PORT}/api/realtime/meeting/stage-summary")
        .replace("__SOURCE_LANG__", source_lang)
        .replace("__TARGET_LANG__", target_lang)
        .replace("__REFERENCE_MEETING_ID__", reference_meeting_id)
        .replace("__REFERENCE_OFFSET_SECONDS__", str(float(reference_offset_seconds)))
        .replace("__MEETING_TITLE__", "实时麦克风会议" if input_kind == "microphone" else "会议录音演示")
        .replace("__START_LABEL__", "播放并开始真实流式处理" if input_kind == "recording" else "开始实时会议")
    )
    components.html(html, height=1280)


def render_realtime_meeting_player(
    audio_path: Path,
    cues: list[dict],
    summary: dict,
    speaker_minutes: dict,
    meeting_name: str = "会议录音",
    reference_meeting_id: str = "",
    reference_offset_seconds: float = 0.0,
) -> None:
    del cues, summary, speaker_minutes
    render_live_route_workspace(
        input_kind="recording",
        title=f"{meeting_name}：真实双路线流式字幕",
        description="播放器扬声器和模型共用同一个 Web Audio 源。选择级联、端到端或对比模式后，系统只发送当前正在播放的 16 kHz PCM 帧。",
        source_lang="en",
        target_lang="zh",
        audio_src=audio_data_url(audio_path),
        reference_meeting_id=reference_meeting_id,
        reference_offset_seconds=reference_offset_seconds,
    )


def render_microphone_realtime_panel() -> None:
    render_live_route_workspace(
        input_kind="microphone",
        title="本地麦克风实时会议",
        description="浏览器采集的每一帧中文 PCM 同时交给所选实时路线。说话人标签在声纹证据稳定后修正，会议结束后音频、记录与纪要会一并归档。",
        source_lang="zh",
        target_lang="en",
    )


def render_voice_assistant_panel() -> None:
    ensure_realtime_backend()
    components.html(
        f"""
<div class="assistant-shell">
  <section class="assistant-hero">
    <div>
      <div class="eyebrow">AI Voice Assistant</div>
      <h2>会议语音助手</h2>
      <p>你可以直接说话提问。助手会先给出过场语音，再结合会议字幕、发言人纪要和联网搜索生成回答，并用语音播报。</p>
    </div>
    <div class="assistant-badges">
      <span>实时 ASR</span><span>会议上下文</span><span>联网搜索</span><span>TTS 播报</span>
    </div>
  </section>
  <div class="assistant-main">
    <section class="assistant-panel control-panel">
      <div class="assistant-title">语音交互</div>
      <button id="assistantTalk" class="talk-btn">按下开始说话</button>
      <div class="assistant-options">
        <label>联网策略
          <select id="assistantSearchMode">
            <option value="force" selected>总是联网</option>
            <option value="auto">智能判断</option>
            <option value="off">不联网</option>
          </select>
        </label>
        <label><input id="assistantVoice" type="checkbox" checked> 播放语音回答</label>
      </div>
      <div id="assistantStatus" class="assistant-status">等待提问。你可以问：刚才 A 主要说了什么？有哪些风险？联网查一下类似产品设计案例。</div>
      <textarea id="assistantText" placeholder="也可以在这里输入问题，然后点发送"></textarea>
      <button id="assistantSend" class="send-btn">发送文字问题</button>
    </section>
    <section class="assistant-panel conversation-panel">
      <div class="assistant-title">对话记录</div>
      <div id="assistantConversation" class="conversation">
        <div class="empty">语音助手会在这里显示识别文本、搜索过程和回答。</div>
      </div>
    </section>
  </div>
</div>
<style>
  .assistant-shell {{
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    overflow: hidden;
    background: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #0f172a;
  }}
  .assistant-hero {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 24px;
    background: linear-gradient(135deg, #0f172a, #0f766e);
    color: white;
  }}
  .eyebrow {{ color: #99f6e4; font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
  .assistant-hero h2 {{ margin: 6px 0 8px; font-size: 30px; }}
  .assistant-hero p {{ margin: 0; color: #d9f2ef; max-width: 760px; }}
  .assistant-badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-content: flex-start; justify-content: flex-end; }}
  .assistant-badges span {{ padding: 7px 10px; border-radius: 999px; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.22); font-size: 12px; font-weight: 800; }}
  .assistant-main {{ display: grid; grid-template-columns: .85fr 1.35fr; gap: 16px; padding: 18px; }}
  .assistant-panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }}
  .assistant-title {{ padding: 14px 16px; background: #f1f5f9; border-bottom: 1px solid #e2e8f0; font-weight: 900; }}
  .control-panel {{ padding-bottom: 14px; }}
  .talk-btn {{ margin: 16px; width: calc(100% - 32px); border: 0; border-radius: 14px; padding: 18px; background: #0f766e; color: white; font-size: 18px; font-weight: 900; cursor: pointer; }}
  .talk-btn.recording {{ background: #dc2626; }}
  .assistant-options {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 0 16px 12px; color: #475569; font-size: 13px; font-weight: 700; }}
  .assistant-options select {{ margin-left: 6px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 8px; font: inherit; background: white; }}
  .assistant-status {{ margin: 0 16px 12px; padding: 10px 12px; border-radius: 10px; background: #ecfdf5; color: #166534; line-height: 1.5; }}
  #assistantText {{ margin: 0 16px 10px; width: calc(100% - 32px); min-height: 82px; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px; resize: vertical; font: inherit; box-sizing: border-box; }}
  .send-btn {{ margin: 0 16px; border: 0; border-radius: 10px; padding: 11px 14px; background: #111827; color: white; font-weight: 900; cursor: pointer; }}
  .conversation {{ min-height: 430px; max-height: 620px; overflow-y: auto; padding: 14px; background: #fbfdff; }}
  .message {{ margin-bottom: 12px; padding: 12px; border-radius: 10px; line-height: 1.65; border: 1px solid #e2e8f0; }}
  .message.user {{ background: #eff6ff; border-color: #bfdbfe; }}
  .message.assistant {{ background: #ecfdf5; border-color: #bbf7d0; }}
  .message.system {{ background: #fff7ed; border-color: #fed7aa; color: #9a3412; }}
  .message small {{ display: block; color: #64748b; margin-bottom: 4px; font-weight: 800; }}
  .sources {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid #dbe3ef; font-size: 13px; }}
  .empty {{ padding: 42px 16px; text-align: center; border: 1px dashed #cbd5e1; border-radius: 12px; color: #64748b; background: white; }}
  @media (max-width: 860px) {{ .assistant-hero, .assistant-main {{ display: block; }} .control-panel {{ margin-bottom: 14px; }} }}
</style>
<script>
  const assistantApi = "http://127.0.0.1:{REALTIME_PORT}/api/assistant/chat";
  const assistantAsrUrl = "ws://127.0.0.1:{REALTIME_PORT}/ws/realtime?source_lang=zh&target_lang=zh&max_sentence_silence=1800";
  const talkBtn = document.getElementById("assistantTalk");
  const sendBtn = document.getElementById("assistantSend");
  const textBox = document.getElementById("assistantText");
  const statusBox = document.getElementById("assistantStatus");
  const convEl = document.getElementById("assistantConversation");
  const searchMode = document.getElementById("assistantSearchMode");
  const voiceToggle = document.getElementById("assistantVoice");
  let assistantWs = null;
  let assistantAudioCtx = null;
  let assistantSource = null;
  let assistantProcessor = null;
  let assistantStream = null;
  let recording = false;
  let conversation = [];

  function esc(s) {{ return (s || "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch])); }}
  function addMsg(role, text, extra="") {{
    const empty = convEl.querySelector(".empty");
    if (empty) empty.remove();
    const div = document.createElement("div");
    div.className = `message ${{role}}`;
    const label = role === "user" ? "你" : (role === "assistant" ? "语音助手" : "系统");
    div.innerHTML = `<small>${{label}}</small><div>${{esc(text)}}</div>${{extra}}`;
    convEl.appendChild(div);
    convEl.scrollTop = convEl.scrollHeight;
  }}
  function speakBrowser(text) {{
    if (!voiceToggle.checked || !window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "zh-CN";
    utter.rate = 1.05;
    speechSynthesis.cancel();
    speechSynthesis.speak(utter);
  }}
  function playAudioB64(b64) {{
    if (!voiceToggle.checked || !b64) return false;
    const audio = new Audio(`data:audio/wav;base64,${{b64}}`);
    audio.play().catch(() => speakBrowser("语音播放失败，我已在屏幕上显示回答。"));
    return true;
  }}
  async function playAudioB64AndWait(b64) {{
    if (!voiceToggle.checked || !b64) return false;
    return await new Promise(resolve => {{
      const audio = new Audio(`data:audio/wav;base64,${{b64}}`);
      audio.onended = () => resolve(true);
      audio.onerror = () => resolve(false);
      audio.play().catch(() => resolve(false));
    }});
  }}
  function floatTo16BitPCM(float32) {{
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {{
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }}
    return out;
  }}
  function downsample(buffer, inRate, outRate) {{
    if (outRate === inRate) return buffer;
    const ratio = inRate / outRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {{
      const start = Math.floor(i * ratio);
      const end = Math.floor((i + 1) * ratio);
      let sum = 0, count = 0;
      for (let j = start; j < end && j < buffer.length; j++) {{ sum += buffer[j]; count++; }}
      result[i] = count ? sum / count : 0;
    }}
    return result;
  }}
  async function askAssistant(question) {{
    question = (question || "").trim();
    if (!question) return;
    addMsg("user", question);
    conversation.push({{role:"user", content: question}});
    statusBox.textContent = "正在理解问题...";
    const modeLabel = searchMode.value === "force" ? "本轮强制联网搜索" : (searchMode.value === "off" ? "本轮不联网" : "本轮智能判断是否联网");
    addMsg("system", `${{modeLabel}}，助手会结合会议上下文回答。`);
    const res = await fetch(assistantApi, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{question, enable_search: searchMode.value !== "off", search_mode: searchMode.value, conversation: conversation.slice(-8)}})
    }});
    const data = await res.json();
    if (!res.ok || data.error) {{
      statusBox.textContent = data.error || "助手调用失败";
      addMsg("system", data.error || "助手调用失败");
      speakBrowser("抱歉，助手调用失败了。");
      return;
    }}
    statusBox.textContent = data.transition || "正在处理...";
    addMsg("system", data.transition || "正在处理...");
    if (!(await playAudioB64AndWait(data.transition_audio_b64))) speakBrowser(data.transition || "正在处理。");
    statusBox.textContent = `完成：${{data.search_used ? "已联网搜索" : "未使用联网搜索"}}，耗时 ${{data.latency}} 秒。`;
    const sources = (data.search_results || []).filter(x => x.url).map(x => `<div>• <a href="${{esc(x.url)}}" target="_blank">${{esc(x.title)}}</a></div>`).join("");
    addMsg("assistant", data.answer, sources ? `<div class="sources"><b>搜索来源</b>${{sources}}</div>` : "");
    conversation.push({{role:"assistant", content: data.answer}});
    if (!playAudioB64(data.audio_b64)) speakBrowser(data.answer);
  }}
  async function stopAssistantMic() {{
    recording = false;
    talkBtn.classList.remove("recording");
    talkBtn.textContent = "按下开始说话";
    if (assistantProcessor) assistantProcessor.disconnect();
    if (assistantSource) assistantSource.disconnect();
    if (assistantStream) assistantStream.getTracks().forEach(t => t.stop());
    if (assistantAudioCtx) await assistantAudioCtx.close();
    assistantProcessor = null;
    assistantSource = null;
    assistantStream = null;
    assistantAudioCtx = null;
    if (assistantWs && assistantWs.readyState === WebSocket.OPEN) {{
      assistantWs.send(JSON.stringify({{type:"stop"}}));
      assistantWs.close();
    }}
  }}
  async function startAssistantMic() {{
    textBox.value = "";
    assistantWs = new WebSocket(assistantAsrUrl);
    assistantWs.binaryType = "arraybuffer";
    assistantWs.onmessage = ev => {{
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") statusBox.textContent = `阿里实时 ASR 已连接：${{msg.asr_model}}。请开始提问。`;
      if (msg.type === "asr_partial") textBox.value = msg.text || "";
      if (msg.type === "asr_final") {{
        const q = msg.text || textBox.value;
        textBox.value = q;
        stopAssistantMic();
        askAssistant(q);
      }}
      if (msg.type === "error") statusBox.textContent = `ASR 错误：${{msg.message}}`;
    }};
    assistantWs.onerror = () => statusBox.textContent = "ASR WebSocket 连接失败。";
    await new Promise((resolve, reject) => {{
      assistantWs.onopen = resolve;
      assistantWs.onerror = reject;
    }});
    assistantStream = await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }} }});
    assistantAudioCtx = new AudioContext();
    assistantSource = assistantAudioCtx.createMediaStreamSource(assistantStream);
    assistantProcessor = assistantAudioCtx.createScriptProcessor(4096, 1, 1);
    assistantProcessor.onaudioprocess = e => {{
      if (!assistantWs || assistantWs.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const pcm = floatTo16BitPCM(downsample(input, assistantAudioCtx.sampleRate, 16000));
      assistantWs.send(pcm.buffer);
    }};
    assistantSource.connect(assistantProcessor);
    assistantProcessor.connect(assistantAudioCtx.destination);
    recording = true;
    talkBtn.classList.add("recording");
    talkBtn.textContent = "正在听，点此停止";
    statusBox.textContent = "正在听你说话，短暂停顿后会自动发送。";
  }}
  talkBtn.onclick = () => {{
    if (recording) stopAssistantMic();
    else startAssistantMic().catch(err => {{
      statusBox.textContent = `无法启动麦克风：${{err}}。也可以使用文字输入。`;
      recording = false;
      talkBtn.classList.remove("recording");
    }});
  }};
  sendBtn.onclick = () => askAssistant(textBox.value);
</script>
""",
        height=760,
    )


def _meeting_matches_date(item: dict, date_filter: str) -> bool:
    date_text = item.get("meeting_date", "")
    try:
        meeting_date = pd.to_datetime(date_text).date()
    except Exception:
        return True
    today = pd.Timestamp("2026-07-04").date()
    if date_filter == "最近 7 天":
        return (today - meeting_date).days <= 7
    if date_filter == "最近 30 天":
        return (today - meeting_date).days <= 30
    return True


def render_meeting_library_page() -> None:
    ensure_meeting_library()
    st.markdown("### 会议记录管理系统")
    st.caption("筛选器只保留时间范围和主题标签；卡片只展示最能判断价值的信息，详情点开后查看。")

    with st.expander("新增会议：上传音频或录制实时会议", expanded=True):
        tab_upload, tab_record, tab_dataset = st.tabs(["上传会议音频", "浏览器录制会议", "导入 AMI 数据集"])
        with tab_upload:
            title = st.text_input("会议标题", value="上传会议录音", key="upload_meeting_title")
            topic = st.text_input("主题标签", value="待处理", key="upload_meeting_topic")
            uploaded = st.file_uploader("选择会议音频", type=["wav", "mp3", "m4a", "webm", "ogg"], key="meeting_audio_upload")
            if uploaded and st.button("保存为新会议", type="primary"):
                item = add_uploaded_meeting(uploaded.name, uploaded.getvalue(), title, topic)
                st.success(f"已入库：{item['title']}。当前状态为待处理，后续可接入同一条 ASR/翻译/声纹/纪要流水线。")
                st.rerun()
        with tab_record:
            st.write("录制入口适合现场演示：录完后先保存音频和元数据，处理完成后会补齐字幕、发言人和纪要。")
            if hasattr(st, "audio_input"):
                recorded = st.audio_input("点击录制一段会议发言", key="meeting_audio_record")
                rec_title = st.text_input("录制会议标题", value="实时录制会议", key="record_meeting_title")
                rec_topic = st.text_input("录制会议主题标签", value="实时会议", key="record_meeting_topic")
                if recorded and st.button("保存录制会议", type="primary"):
                    item = add_uploaded_meeting(recorded.name or "recorded_meeting.wav", recorded.getvalue(), rec_title, rec_topic)
                    st.success(f"已入库：{item['title']}。")
                    st.rerun()
            else:
                st.info("当前 Streamlit 版本没有 audio_input 组件；可以先用上面的上传入口保存浏览器或系统录音。")
        with tab_dataset:
            st.write("把 AMI 音频下载到 `data/raw/amicorpus/{meeting_id}/audio/` 后，可以扫描入库。未处理音频会显示为待处理，不会伪装成已有纪要。")
            st.code("powershell -ExecutionPolicy Bypass -File scripts/download_more_ami.ps1", language="powershell")
            if st.button("扫描已下载 AMI 音频并入库"):
                registered = scan_downloaded_ami_audio()
                if registered:
                    st.success(f"已入库 {len(registered)} 条 AMI 待处理会议。")
                    st.rerun()
                else:
                    st.info("没有发现新的 AMI 音频。")

    meetings = load_meeting_index()
    all_tags = sorted({tag for item in meetings for tag in (item.get("tags") or [])})
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        date_filter = st.selectbox("时间范围", ["最近 7 天", "最近 30 天", "全部"], index=0)
    with filter_col2:
        selected_tags = st.multiselect("主题标签", all_tags, default=[])

    filtered = [
        item
        for item in meetings
        if _meeting_matches_date(item, date_filter)
        and (not selected_tags or any(tag in (item.get("tags") or []) for tag in selected_tags))
    ]
    st.caption(f"共 {len(filtered)} 场会议")

    if not filtered:
        st.info("没有符合当前筛选条件的会议。")
        return

    for row_start in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, item in zip(cols, filtered[row_start : row_start + 2]):
            with col:
                status_map = {"completed": "已完成", "partial_processing": "采样处理中", "pending_processing": "待处理"}
                status_label = status_map.get(item.get("status"), "待处理")
                tags = " / ".join(item.get("tags") or [])
                st.markdown(
                    f"""
<div style="border:1px solid #dbe3ef;border-radius:10px;padding:14px;margin-bottom:8px;background:white;">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
    <b style="font-size:16px;color:#0f172a;">{item.get('title','未命名会议')}</b>
    <span style="white-space:nowrap;border-radius:999px;padding:3px 8px;background:{'#ecfdf5' if status_label == '已完成' else '#eff6ff' if status_label == '采样处理中' else '#fff7ed'};color:{'#166534' if status_label == '已完成' else '#1d4ed8' if status_label == '采样处理中' else '#9a3412'};font-size:12px;font-weight:800;">{status_label}</span>
  </div>
  <div style="margin-top:8px;color:#64748b;font-size:13px;">{item.get('meeting_date','')} · {item.get('source_label','')} · {item.get('topic','')}</div>
  <div style="margin-top:8px;color:#0f172a;line-height:1.55;">{item.get('summary','暂无摘要')}</div>
  <div style="margin-top:10px;color:#475569;font-size:13px;">发言人 {item.get('speaker_count',0)} · 行动项 {item.get('action_count',0)} · 风险 {item.get('risk_count',0)} · {tags}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                with st.expander("查看会议详情"):
                    detail = load_meeting_detail(item.get("meeting_id", ""))
                    minutes = detail.get("minutes") or {}
                    transcript = detail.get("transcript") or []
                    diarization = detail.get("diarization") or {}
                    final_minutes = minutes.get("final_minutes") or {}
                    if item.get("status") == "partial_processing":
                        st.info(f"该会议已完成均匀采样处理，当前覆盖率 {float(item.get('coverage_ratio') or 0) * 100:.1f}%；检索回答会标注为部分覆盖。")
                    if item.get("speaker_status"):
                        speaker_backend = item.get("speaker_backend") or "online_provisional"
                        speaker_status_label = {
                            "resolved": "已完成声纹校正并回写字幕",
                            "audio_diarized": "已完成声纹时间线；历史字幕缺少音频时间戳",
                            "pending_post_diarization": "等待会后声纹校正",
                        }.get(item.get("speaker_status"), item.get("speaker_status"))
                        st.caption(f"声纹状态：{speaker_status_label} · 后端：{speaker_backend}")
                    audio_path = rel(item.get("audio_path"))
                    if audio_path and audio_path.exists():
                        st.audio(str(audio_path))
                        if st.button("重新运行会议级声纹校正", key=f"rediarize_{item.get('meeting_id')}"):
                            with st.spinner("正在运行 FSMN-VAD + CAM++ 谱聚类，请稍候..."):
                                try:
                                    result = rerun_stored_diarization(item.get("meeting_id", ""))
                                    if result.get("success"):
                                        st.success(
                                            f"已完成：识别出 {int((result.get('diarization') or {}).get('speaker_count') or 0)} 位参会者。"
                                        )
                                    else:
                                        st.error((result.get("diarization") or {}).get("error") or "声纹校正失败")
                                except urllib.error.HTTPError as exc:
                                    st.error(f"声纹校正接口错误：HTTP {exc.code}")
                                except Exception as exc:
                                    st.error(f"声纹校正失败：{exc}")
                            st.rerun()
                    if diarization.get("turns"):
                        st.write("**会议级声纹时间线**")
                        diarization_rows = [
                            {
                                "time": f"{float(turn.get('start', 0)):.1f}-{float(turn.get('end', 0)):.1f}s",
                                "speaker": turn.get("speaker", ""),
                            }
                            for turn in diarization.get("turns", [])[:40]
                        ]
                        st.dataframe(pd.DataFrame(diarization_rows), use_container_width=True, hide_index=True)
                        if not diarization.get("corrected_transcript_count"):
                            st.info("该历史记录缺少转写音频时间戳；声纹时间线已完成，播放一次该会议后可将标签精确回写到字幕。")
                    st.write("**最终纪要**")
                    st.write(final_minutes.get("one_sentence_summary") or item.get("summary") or "暂无")
                    if final_minutes.get("action_items"):
                        st.write("**行动项**")
                        for action in final_minutes.get("action_items", []):
                            st.write(f"- {action}")
                    if final_minutes.get("risks"):
                        st.write("**风险**")
                        for risk in final_minutes.get("risks", []):
                            st.write(f"- {risk}")
                    if transcript:
                        st.write("**发言样本**")
                        rows = [
                            {
                                "time": f"{x.get('start', 0):.1f}-{x.get('end', 0):.1f}s",
                                "speaker": x.get("speaker", ""),
                                "zh": x.get("zh", ""),
                                "en": x.get("en", ""),
                            }
                            for x in transcript[:12]
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("这条会议已有音频，但还没有处理出字幕和纪要。")


st.set_page_config(page_title="会议/课堂双语字幕与端到端语音翻译系统", layout="wide")
st.markdown(
    """
<style>
  .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1220px;
  }
  [data-testid="stSidebar"] {
    background: #f8fafc;
  }
  h1 {
    letter-spacing: 0;
  }
  div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
  }
</style>
""",
    unsafe_allow_html=True,
)
st.title("会议/课堂双语字幕与端到端语音翻译系统")
st.caption("连续播放音频、实时滚动双语字幕、播放完成后展示会议纪要，并保留 C3 级联路线与 C4 端到端路线对比。")

page = st.sidebar.radio("页面", ["首页", "会议库", "实时麦克风", "字幕", "摘要", "评估"])
if st.sidebar.button("重新加载结果"):
    st.rerun()
if st.sidebar.button("运行全流程"):
    with st.spinner("正在运行全流程..."):
        code, output = run_pipeline()
    st.sidebar.code(output[-4000:])
    if code == 0:
        st.sidebar.success("全流程运行完成")
    else:
        st.sidebar.error(f"全流程失败，退出码 {code}")

segments, asr, bilingual, omni, compare, latency, summary, speaker_minutes = load_data()

if page == "首页":
    if not segments:
        st.info("还没有可展示的样例。请先运行 C1 或全流程。")
    else:
        st.markdown("### 会议语音助手")
        render_voice_assistant_panel()
        st.markdown("### 会议录音作为数据输入")
        demo_audio, cues = load_streaming_demo(segments, bilingual, omni)
        playback_sources = playback_meeting_sources(demo_audio)
        if playback_sources:
            selected_source = st.selectbox(
                "选择会议数据集或历史录音",
                playback_sources,
                format_func=lambda source: f"{source['label']}  |  {source['detail']}",
                key="playback-meeting-source",
            )
            source_duration = float(selected_source.get("duration_seconds") or sf.info(str(selected_source["audio_path"])).duration)
            clip_start = 0.0
            if source_duration > 180:
                clip_start = st.slider(
                    "长会议播放起点（每次处理 180 秒，模型输入与播放器音频完全一致）",
                    min_value=0.0,
                    max_value=max(0.0, source_duration - 180),
                    value=0.0,
                    step=30.0,
                    key=f"playback-start-{selected_source['key']}",
                )
            playback_audio, playback_duration = prepare_playback_audio(
                selected_source["audio_path"], selected_source["key"], clip_start
            )
            if source_duration > playback_duration:
                st.caption(f"当前播放范围：{clip_start:.0f}s - {clip_start + playback_duration:.0f}s / 全长 {source_duration:.0f}s")
            render_realtime_meeting_player(
                playback_audio,
                cues,
                summary,
                speaker_minutes,
                meeting_name=selected_source["label"],
                reference_meeting_id="" if selected_source.get("key") == "default-demo" else str(selected_source.get("key") or ""),
                reference_offset_seconds=clip_start,
            )
        else:
            st.error("没有找到可播放的会议音频，请先运行会议处理或导入数据集。")
        st.markdown("### 本地麦克风实时输入")
        render_microphone_realtime_panel()

elif page == "字幕":
    st.subheader("双语字幕表格")
    items = read_json("outputs/translation/bilingual_segments.json", [])
    st.dataframe(pd.DataFrame(items), use_container_width=True)
    for filename, label in [
        ("outputs/subtitles/meeting_bilingual.srt", "下载 SRT"),
        ("outputs/subtitles/meeting_bilingual.vtt", "下载 VTT"),
    ]:
        path = ROOT / filename
        if path.exists():
            st.download_button(label, data=path.read_bytes(), file_name=path.name)

elif page == "实时麦克风":
    render_microphone_realtime_panel()

elif page == "会议库":
    render_meeting_library_page()

elif page == "摘要":
    st.subheader("一句话摘要")
    st.write(summary.get("one_sentence_summary") or "暂无")
    st.subheader("要点")
    for item in summary.get("key_points", []):
        st.write(f"- {item}")
    st.subheader("关键词")
    st.write("、".join(summary.get("keywords", [])) or "暂无")
    st.subheader("时间线")
    st.dataframe(pd.DataFrame(summary.get("timeline", [])), use_container_width=True)
    tts_path = ROOT / "outputs/tts/summary_zh.wav"
    if tts_path.exists():
        st.audio(str(tts_path))

elif page == "评估":
    st.subheader("C3/C4 对比")
    st.dataframe(pd.DataFrame(compare), use_container_width=True)
    st.subheader("延迟报告")
    st.json(latency)
    chart_df = pd.DataFrame(
        [
            {"route": "C3 avg total", "latency": latency.get("c3_avg_total_latency", 0)},
            {"route": "C4 avg", "latency": latency.get("c4_avg_latency", 0)},
        ]
    )
    st.bar_chart(chart_df.set_index("route"))
