from __future__ import annotations

import argparse
import sys

from .utils.logger import setup_logging


STAGES = {"c1", "c2", "c3", "subtitles", "summary", "tts", "c4", "eval", "diarize", "diarize_fetch", "stream", "all"}


def run_stage(stage: str, config: str = "config/config.yaml", limit: int | None = None) -> None:
    if stage == "c1":
        from .c1_preprocess import run

        run(config)
    elif stage == "c2":
        from .c2_asr_ali import run

        run(config, limit=limit)
    elif stage == "c3":
        from .c3_translate_ali import run

        run(config, limit=limit)
    elif stage == "subtitles":
        from .subtitle_writer import run

        run(config)
    elif stage == "summary":
        from .summary_ali import run

        run(config)
    elif stage == "tts":
        from .c5_tts_ali import run

        run(config)
    elif stage == "c4":
        from .c4_omni_ali import run

        run(config, limit=limit)
    elif stage == "eval":
        from .evaluator import run

        run(config)
    elif stage == "stream":
        from .streaming_cues import run

        run(config)
    elif stage == "diarize":
        from .diarization_ali import run

        run(config)
    elif stage == "diarize_fetch":
        from .diarization_ali import run

        run(config, mode="fetch")
    elif stage == "all":
        for name in ["c1", "c2", "c3", "subtitles", "summary", "tts", "c4", "eval", "stream"]:
            run_stage(name, config=config, limit=limit)
    else:
        raise ValueError(f"未知 stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="仅用于 API 批处理阶段的调试条数")
    args = parser.parse_args()
    logger = setup_logging()
    try:
        run_stage(args.stage, config=args.config, limit=args.limit)
    except Exception as exc:
        logger.error("Stage %s failed: %s", args.stage, exc)
        print(f"ERROR: stage {args.stage} failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
