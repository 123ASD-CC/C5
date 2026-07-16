from __future__ import annotations

import argparse

from .utils.api_clients import AliOpenAIClient, MissingConfiguration
from .utils.io_utils import load_config, project_path, read_json, write_json
from .utils.logger import setup_logging


def _script(summary: dict) -> str:
    parts = [summary.get("one_sentence_summary", "")]
    key_points = summary.get("key_points", [])
    if key_points:
        parts.append("主要要点包括：" + "；".join(key_points))
    return "\n".join(x for x in parts if x).strip()


def run(config_path: str = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    logger = setup_logging()
    summary = read_json(cfg["paths"]["summary_json"])
    model = cfg["models"]["tts_model"]
    voice = cfg["api"]["tts_voice"]
    fmt = cfg["api"]["tts_format"]
    text = _script(summary)
    request = {"model": model, "voice": voice, "format": fmt, "text": text}
    write_json("outputs/tts/tts_request.json", request)
    result = {"audio_path": None, "latency": 0.0, "success": False, "error": None, **request}

    if not text:
        result["error"] = "摘要播报稿为空，跳过 TTS"
        write_json("outputs/tts/tts_result.json", result)
        return result

    try:
        client = AliOpenAIClient(cfg)
        ready, preflight_error = client.preflight()
        if not ready:
            raise RuntimeError(preflight_error)
        audio_bytes, latency = client.tts(model=model, text=text, voice=voice, audio_format=fmt)
        out_path = project_path(cfg["paths"]["tts_audio"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        result.update({"audio_path": cfg["paths"]["tts_audio"], "latency": latency, "success": True})
    except MissingConfiguration as exc:
        logger.error(str(exc))
        result["error"] = "缺少 API 环境变量，跳过真实 TTS 调用"
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        result["error"] = str(exc)

    write_json("outputs/tts/tts_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
