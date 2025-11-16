from dataclasses import dataclass
from pathlib import Path

from src.core.config import DATA_DIR, ensure_data_dir

ensure_data_dir()
HORSES_DATA_DIR = DATA_DIR / "horses"
HORSES_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = HORSES_DATA_DIR
    BETFAIR_BASE_URL: str = "https://www.betfair.com/exchange/plus/"
    BETFAIR_HORSE_RACING_URL: str = "https://www.betfair.com/exchange/plus/horse-racing"
    TIMEFORM_BASE_URL: str = "https://www.timeform.com/horse-racing"
    SPORTINGLIFE_BASE_URL: str = "https://www.sportinglife.com/racing/results"
    SELENIUM_HEADLESS: bool = False
    SELENIUM_PAGELOAD_TIMEOUT_SEC: int = 5
    SELENIUM_IMPLICIT_WAIT_SEC: int = 3
    SELENIUM_EXPLICIT_WAIT_SEC: int = 2
    TIMEFORM_MIN_DELAY_SEC: float = 0.5
    TIMEFORM_MAX_DELAY_SEC: float = 1.0
    SPORTINGLIFE_MIN_DELAY_SEC: float = 0.3
    SPORTINGLIFE_MAX_DELAY_SEC: float = 0.6
    CSV_ENCODING: str = "utf-8-sig"
    LOG_LEVEL: str = "INFO"


settings = Settings()

__all__ = ["settings", "Settings", "HORSES_DATA_DIR"]
