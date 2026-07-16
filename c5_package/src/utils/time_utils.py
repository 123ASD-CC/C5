from __future__ import annotations


def seconds_to_srt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    return seconds_to_srt_time(seconds).replace(",", ".")


def seconds_to_clock_range(start: float, end: float) -> str:
    def fmt(value: float) -> str:
        total = int(round(value))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{fmt(start)}-{fmt(end)}"

