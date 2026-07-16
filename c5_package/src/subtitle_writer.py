from __future__ import annotations

import argparse

from .utils.io_utils import load_config, read_json, write_text
from .utils.time_utils import seconds_to_srt_time, seconds_to_vtt_time


def build_srt(items: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(items, start=1):
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{seconds_to_srt_time(item['start'])} --> {seconds_to_srt_time(item['end'])}",
                    item.get("source_text", ""),
                    item.get("translation_zh", ""),
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def build_vtt(items: list[dict]) -> str:
    blocks = ["WEBVTT\n"]
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"{seconds_to_vtt_time(item['start'])} --> {seconds_to_vtt_time(item['end'])}",
                    item.get("source_text", ""),
                    item.get("translation_zh", ""),
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def run(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    items = read_json(cfg["paths"]["bilingual_json"])
    write_text(cfg["paths"]["subtitle_srt"], build_srt(items))
    write_text(cfg["paths"]["subtitle_vtt"], build_vtt(items))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

