from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.horses.config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def ensure_day_folder(base: Path | None = None) -> Path:
    base_dir = base or settings.DATA_DIR
    day_dir = base_dir / today_str()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def hhmm_to_today_iso(hhmm: str) -> str:
    try:
        hour, minute = [int(x) for x in hhmm.strip()[:5].split(":")]
        now = datetime.now()
        dt = datetime(now.year, now.month, now.day, hour, minute)
        return dt.isoformat(timespec="minutes")
    except Exception:
        return datetime.now().isoformat(timespec="minutes")


def iso_to_hhmm(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M")
    except Exception:
        return ""
