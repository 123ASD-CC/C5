from __future__ import annotations

import logging
import os
from pathlib import Path


SECRET_ENV_NAMES = ("DASHSCOPE_API_KEY",)


def mask_key(value: str | None) -> str:
    """Return a printable masked API key such as sk-****abcd."""
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    prefix = value[:3] if value.startswith("sk-") else value[:2]
    return f"{prefix}****{value[-4:]}"


class SecretMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for env_name in SECRET_ENV_NAMES:
            secret = os.getenv(env_name)
            if secret and secret in message:
                message = message.replace(secret, mask_key(secret))
        record.msg = message
        record.args = ()
        return True


def setup_logging(log_dir: str | Path = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("meeting_speech_system")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    masking = SecretMaskingFilter()

    app_handler = logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    app_handler.addFilter(masking)

    error_handler = logging.FileHandler(Path(log_dir) / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(masking)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(masking)

    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stream_handler)
    return logger

