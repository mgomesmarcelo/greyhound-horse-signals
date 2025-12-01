from dataclasses import dataclass
from pathlib import Path

from src.core.config import DATA_DIR, ensure_data_dir

ENTRY_TYPE_LABELS: dict[str, str] = {
    "back": "back",
    "lay": "lay",
}
ENTRY_TYPE_LABELS_INV: dict[str, str] = {v: k for k, v in ENTRY_TYPE_LABELS.items()}
SOURCE_LABELS: dict[str, str] = {
    "top3": "Timeform Top 3",
    "forecast": "Timeform Forecast",
    "betfair_resultado": "Betfair Resultado Direto",
}
SOURCE_LABELS_INV: dict[str, str] = {v: k for k, v in SOURCE_LABELS.items()}

RULE_LABELS: dict[str, str] = {
    "lider_volume_total": "líder volume total",
    "terceiro_queda50": "terceiro_queda50",
}
RULE_LABELS_INV: dict[str, str] = {v: k for k, v in RULE_LABELS.items()}

ensure_data_dir()
GREYHOUNDS_DATA_DIR = ensure_data_dir("greyhounds")
SIGNALS_DIR = ensure_data_dir("greyhounds", "signals")
TIMEFORM_TOP3_DIR = ensure_data_dir("greyhounds", "timeform_top3")
TIMEFORM_FORECAST_DIR = ensure_data_dir("greyhounds", "TimeformForecast")
RESULT_DIR = ensure_data_dir("greyhounds", "Result")
RACE_LINKS_DIR = ensure_data_dir("greyhounds", "race_links")

PROCESSED_DIR = ensure_data_dir("greyhounds", "processed")
PROCESSED_SIGNALS_DIR = ensure_data_dir("greyhounds", "processed", "signals")
PROCESSED_TIMEFORM_TOP3_DIR = ensure_data_dir("greyhounds", "processed", "timeform_top3")
PROCESSED_TIMEFORM_FORECAST_DIR = ensure_data_dir("greyhounds", "processed", "TimeformForecast")
PROCESSED_RESULT_DIR = ensure_data_dir("greyhounds", "processed", "Result")
PROCESSED_RACE_LINKS_DIR = ensure_data_dir("greyhounds", "processed", "race_links")


@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = GREYHOUNDS_DATA_DIR
    RAW_SIGNALS_DIR: Path = SIGNALS_DIR
    RAW_TIMEFORM_TOP3_DIR: Path = TIMEFORM_TOP3_DIR
    RAW_TIMEFORM_FORECAST_DIR: Path = TIMEFORM_FORECAST_DIR
    RAW_RESULT_DIR: Path = RESULT_DIR
    RAW_RACE_LINKS_DIR: Path = RACE_LINKS_DIR
    PROCESSED_DIR: Path = PROCESSED_DIR
    PROCESSED_SIGNALS_DIR: Path = PROCESSED_SIGNALS_DIR
    PROCESSED_TIMEFORM_TOP3_DIR: Path = PROCESSED_TIMEFORM_TOP3_DIR
    PROCESSED_TIMEFORM_FORECAST_DIR: Path = PROCESSED_TIMEFORM_FORECAST_DIR
    PROCESSED_RESULT_DIR: Path = PROCESSED_RESULT_DIR
    PROCESSED_RACE_LINKS_DIR: Path = PROCESSED_RACE_LINKS_DIR
    BETFAIR_BASE_URL: str = "https://www.betfair.com/exchange/plus/"
    BETFAIR_GREYHOUND_RACING_URL: str = "https://www.betfair.com/exchange/plus/en/greyhound-racing-betting-4339"
    TIMEFORM_BASE_URL: str = "https://www.timeform.com/greyhound-racing"
    SELENIUM_HEADLESS: bool = False
    SELENIUM_PAGELOAD_TIMEOUT_SEC: int = 45
    SELENIUM_IMPLICIT_WAIT_SEC: int = 5
    SELENIUM_EXPLICIT_WAIT_SEC: int = 15
    TIMEFORM_MIN_DELAY_SEC: float = 0.5
    TIMEFORM_MAX_DELAY_SEC: float = 1.0
    CSV_ENCODING: str = "utf-8-sig"
    LOG_LEVEL: str = "INFO"


settings = Settings()

__all__ = [
    "ENTRY_TYPE_LABELS",
    "ENTRY_TYPE_LABELS_INV",
    "SOURCE_LABELS",
    "SOURCE_LABELS_INV",
    "SIGNALS_DIR",
    "TIMEFORM_TOP3_DIR",
    "TIMEFORM_FORECAST_DIR",
    "RESULT_DIR",
    "RACE_LINKS_DIR",
    "PROCESSED_DIR",
    "PROCESSED_SIGNALS_DIR",
    "PROCESSED_TIMEFORM_TOP3_DIR",
    "PROCESSED_TIMEFORM_FORECAST_DIR",
    "PROCESSED_RESULT_DIR",
    "PROCESSED_RACE_LINKS_DIR",
    "RULE_LABELS",
    "RULE_LABELS_INV",
    "Settings",
    "settings",
    "GREYHOUNDS_DATA_DIR",
]