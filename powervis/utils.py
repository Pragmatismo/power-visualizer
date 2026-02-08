import os
import re
from typing import Optional

MINUTES_PER_DAY = 24 * 60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def time_to_minutes(hhmm: str) -> int:
    # "HH:MM"
    try:
        hh, mm = hhmm.strip().split(":")
        return clamp(int(hh) * 60 + int(mm), 0, MINUTES_PER_DAY)
    except Exception:
        return 0


def minutes_to_time(m: int) -> str:
    m = clamp(int(m), 0, MINUTES_PER_DAY)
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


def format_duration_minutes(m: int) -> str:
    m = clamp(int(m), 1, MINUTES_PER_DAY)
    hh = m // 60
    mm = m % 60
    return f"{hh:d}:{mm:02d}"


def format_duration_hhmm(m: int) -> str:
    m = clamp(int(m), 1, MINUTES_PER_DAY)
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


def parse_duration_text(text: str) -> Optional[int]:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) != 2:
                return None
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if parts[1] else 0
            if minutes < 0:
                return None
            hours += minutes // 60
            minutes = minutes % 60
            total = hours * 60 + minutes
        else:
            hours = float(cleaned)
            if hours < 0:
                return None
            total = int(hours * 60)
        if total >= MINUTES_PER_DAY:
            return MINUTES_PER_DAY
        return max(1, total)
    except Exception:
        return None


def format_duration_export(m: int) -> str:
    return format_duration_minutes(m)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-")


def derive_modified_path(path: str) -> str:
    if path.lower().endswith("_modified.json"):
        return path
    base, ext = os.path.splitext(path)
    if ext.lower() != ".json":
        ext = ".json"
    return f"{base}_modified{ext}"


def parse_hhmm(text: str) -> Optional[int]:
    text = text.strip()
    if not re.match(r"^\d{2}:\d{2}$", text):
        return None
    hh = int(text[:2])
    mm = int(text[3:])
    if hh < 0 or hh > 24 or mm < 0 or mm > 59:
        return None
    if hh == 24 and mm != 0:
        return None
    return hh * 60 + mm
