from __future__ import annotations

import argparse
from statistics import mean

from .utils.io_utils import load_config, read_json, write_json


def _avg(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 4) if values else 0.0


def run(config_path: str = "config/config.yaml") -> tuple[dict, list[dict]]:
    cfg = load_config(config_path)
    c3 = read_json(cfg["paths"]["bilingual_json"], default=[])
    c4 = read_json(cfg["paths"]["omni_json"], default=[])
    c4_by_id = {x["id"]: x for x in c4}

    compare = []
    failures = []
    for item in c3:
        omni = c4_by_id.get(item["id"], {})
        row = {
            "id": item["id"],
            "start": item.get("start"),
            "end": item.get("end"),
            "source_text": item.get("source_text", ""),
            "c3_translation_zh": item.get("translation_zh", ""),
            "c4_omni_text": omni.get("omni_text", ""),
            "asr_latency": item.get("asr_latency", 0.0),
            "c3_translate_latency": item.get("latency", 0.0),
            "c3_total_latency": round(float(item.get("asr_latency", 0.0)) + float(item.get("latency", 0.0)), 4),
            "c4_latency": omni.get("latency", 0.0),
            "c4_chinese_ratio": omni.get("chinese_ratio", 0.0),
            "c4_language_check": omni.get("language_check", "fail"),
            "c3_success": item.get("success", False),
            "c4_success": omni.get("success", False),
        }
        compare.append(row)
        if not row["c3_success"] or not row["c4_success"]:
            failures.append({"id": row["id"], "c3_success": row["c3_success"], "c4_success": row["c4_success"]})

    c4_pass = [1.0 if x.get("language_check") == "pass" else 0.0 for x in c4]
    report = {
        "c3_avg_asr_latency": _avg([x.get("asr_latency", 0.0) for x in c3]),
        "c3_avg_translate_latency": _avg([x.get("latency", 0.0) for x in c3]),
        "c3_avg_total_latency": _avg([r["c3_total_latency"] for r in compare]),
        "c4_avg_latency": _avg([x.get("latency", 0.0) for x in c4]),
        "c4_avg_chinese_ratio": _avg([x.get("chinese_ratio", 0.0) for x in c4]),
        "c4_language_check_pass_rate": _avg(c4_pass),
        "failure_items": failures,
    }
    write_json(cfg["paths"]["eval_latency_json"], report)
    write_json(cfg["paths"]["eval_compare_json"], compare)
    return report, compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

