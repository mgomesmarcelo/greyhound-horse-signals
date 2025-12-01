import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

import pandas as pd
from loguru import logger

from src.core.config import ensure_data_dir
from src.greyhounds.config import settings
from src.greyhounds.utils.files import write_dataframe_snapshots
from src.greyhounds.scrapers.betfair_index import scrape_betfair_index


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    today_str = date.today().isoformat()

    ensure_data_dir("greyhounds")  # garante base existente
    raw_dir = settings.RAW_RACE_LINKS_DIR
    parquet_dir = settings.PROCESSED_RACE_LINKS_DIR

    logger.info(
        "Iniciando scrape do indice da Betfair: https://www.betfair.com/exchange/plus/en/greyhound-racing-betting-4339"
    )
    rows = scrape_betfair_index()

    df = pd.DataFrame(rows)
    raw_path = raw_dir / f"race_links_{today_str}.csv"
    parquet_path = parquet_dir / f"race_links_{today_str}.parquet"
    write_dataframe_snapshots(df, raw_path=raw_path, parquet_path=parquet_path)
    logger.info(
        "race_links processado. CSV bruto: {} | Parquet: {}",
        raw_path,
        parquet_path,
    )


if __name__ == "__main__":
    main()