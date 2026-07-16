from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
from pathlib import Path

from .io_utils import project_path


def audio_to_data_url(path: str | Path, max_mb: float = 10) -> str:
    full = project_path(path)
    data = full.read_bytes()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"base64 音频输入约 {size_mb:.2f}MB，超过 {max_mb}MB；请改用 OSS URL 或缩短切片。"
        )
    mime = mimetypes.guess_type(str(full))[0] or "audio/wav"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def data_url_to_raw_base64(data_url: str) -> tuple[str, str]:
    header, payload = data_url.split(",", 1)
    fmt = "wav"
    if "audio/" in header:
        fmt = header.split("audio/", 1)[1].split(";", 1)[0]
    return payload, fmt


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("未找到 ffmpeg。请安装 ffmpeg 或 pip install imageio-ffmpeg。") from exc
    return ffmpeg


def convert_to_16k_mono(input_path: str | Path, output_path: str | Path) -> None:
    full_in = project_path(input_path)
    full_out = project_path(output_path)
    if not full_in.exists():
        raise FileNotFoundError(f"原始音频不存在: {full_in}")
    full_out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(full_in),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(full_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def cut_wav_segment(input_path: str | Path, output_path: str | Path, start: float, duration: float) -> None:
    full_in = project_path(input_path)
    full_out = project_path(output_path)
    full_out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(full_in),
        "-acodec",
        "copy",
        str(full_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
