from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.core.config import settings
from src.horses.scrapers.sportinglife import scrape_day


def _out_paths(date_str: str) -> tuple[Path, Path, Path]:
    base = settings.DATA_DIR / "horses"
    p_top3 = base / "sportinglife_top3"
    p_fc = base / "SportingLifeForecast"
    p_links = base / "race_links"
    for path in (p_top3, p_fc, p_links):
        path.mkdir(parents=True, exist_ok=True)
    return (
        p_top3 / f"sportinglife_top3_{date_str}.csv",
        p_fc / f"SportingLifeForecast_{date_str}.csv",
        p_links / f"sportinglife_race_links_{date_str}.csv",
    )


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_top3, out_fc, out_links = _out_paths(date_str)

    logger.info("Iniciando Sporting Life (update) para {}", date_str)
    rows_top3, rows_fc, links = scrape_day(date_str)

    df_top3 = pd.DataFrame(rows_top3) if rows_top3 else pd.DataFrame(
        columns=["track_name", "race_time_iso", "TimeformTop1", "TimeformTop2", "TimeformTop3"]
    )
    df_fc = pd.DataFrame(rows_fc) if rows_fc else pd.DataFrame(
        columns=["track_name", "race_time_iso", "SportingLifeForecast"]
    )
    df_links = pd.DataFrame({"url": links}) if links else pd.DataFrame(columns=["url"])

    df_top3.to_csv(out_top3, index=False, encoding=settings.CSV_ENCODING)
    df_fc.to_csv(out_fc, index=False, encoding=settings.CSV_ENCODING)
    df_links.to_csv(out_links, index=False, encoding=settings.CSV_ENCODING)

    logger.info(
        "Concluído Sporting Life (update). Salvos:\n  top3: {}\n  forecast: {}\n  links: {}",
        out_top3,
        out_fc,
        out_links,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

