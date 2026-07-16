from __future__ import annotations

import argparse
import json

from .utils.api_clients import AliOpenAIClient, MissingConfiguration
from .utils.io_utils import load_config, read_json, write_json, write_text
from .utils.logger import setup_logging
from .utils.text_utils import extract_json_object
from .utils.time_utils import seconds_to_clock_range


EMPTY_SUMMARY = {
    "one_sentence_summary": "",
    "key_points": [],
    "keywords": [],
    "timeline": [],
    "questions_or_actions": [],
}


def _summary_markdown(data: dict) -> str:
    lines = ["# 中文会议/课堂摘要", "", f"## 一句话摘要", data.get("one_sentence_summary", ""), ""]
    lines.append("## 要点")
    lines.extend(f"- {x}" for x in data.get("key_points", []))
    lines.extend(["", "## 关键词", "、".join(data.get("keywords", [])), "", "## 时间线"])
    for item in data.get("timeline", []):
        lines.append(f"- {item.get('time', '')}: {item.get('topic', '')}")
    lines.extend(["", "## 问题或行动项"])
    lines.extend(f"- {x}" for x in data.get("questions_or_actions", []))
    return "\n".join(lines).strip() + "\n"


def run(config_path: str = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    logger = setup_logging()
    items = read_json(cfg["paths"]["bilingual_json"])
    model = cfg["models"]["llm_strong_model"] or cfg["models"]["llm_model"]
    content = "\n".join(
        f"[{seconds_to_clock_range(x['start'], x['end'])}] EN: {x.get('source_text','')} ZH: {x.get('translation_zh','')}"
        for x in items
        if x.get("translation_zh") or x.get("source_text")
    )
    data = dict(EMPTY_SUMMARY)

    try:
        client = AliOpenAIClient(cfg)
        ready, preflight_error = client.preflight()
        if not ready:
            raise RuntimeError(preflight_error)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是会议/课堂纪要助手。只能基于字幕内容总结，不要编造；保留专业术语；"
                    "中文输出；必须输出 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请按以下结构输出 JSON："
                    '{"one_sentence_summary":"","key_points":[],"keywords":[],"timeline":[{"time":"","topic":""}],"questions_or_actions":[]}'
                    f"\n\n字幕内容：\n{content}"
                ),
            },
        ]
        text, latency = client.chat_text(model=model, messages=messages, temperature=0.2)
        data = json.loads(extract_json_object(text))
        data["summary_model"] = model
        data["latency"] = latency
        data["success"] = True
        data["error"] = None
    except MissingConfiguration as exc:
        logger.error(str(exc))
        data.update({"success": False, "error": "缺少 API 环境变量，跳过真实摘要调用", "summary_model": model})
    except Exception as exc:
        logger.error("Summary failed: %s", exc)
        data.update({"success": False, "error": str(exc), "summary_model": model})

    write_json(cfg["paths"]["summary_json"], data)
    write_text(cfg["paths"]["summary_md"], _summary_markdown(data))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
