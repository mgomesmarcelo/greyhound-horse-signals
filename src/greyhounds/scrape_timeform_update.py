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
from src.greyhounds.scrapers.betfair_index import scrape_betfair_index
from src.greyhounds.scrapers.timeform import (
    save_timeform_forecast,
    save_timeform_top3,
    scrape_timeform_for_races,
)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    logger.info("Iniciando atualizacao Timeform para greyhounds...")

    today_str = date.today().isoformat()

    base_dir: Path = ensure_data_dir("greyhounds")

    forecast_dir = settings.RAW_TIMEFORM_FORECAST_DIR
    top3_dir = settings.RAW_TIMEFORM_TOP3_DIR

    race_links_dir = base_dir / "race_links"
    race_links_dir.mkdir(parents=True, exist_ok=True)

    race_links_path = race_links_dir / f"race_links_{today_str}.csv"

    if not race_links_path.exists():
        logger.warning(f"Arquivo nÃ£o encontrado: {race_links_path}. Gerando Ã­ndice da Betfair agora...")
        rows = scrape_betfair_index()
        pd.DataFrame(rows).to_csv(race_links_path, index=False)
        logger.info(f"race_links gerado em {race_links_path}")

    df = pd.read_csv(race_links_path)
    rows = df.to_dict("records")

    updates = list(scrape_timeform_for_races(rows))

    forecast_path = forecast_dir / f"TimeformForecast_{today_str}.csv"
    forecast_parquet_path = settings.PROCESSED_TIMEFORM_FORECAST_DIR / f"TimeformForecast_{today_str}.parquet"
    top3_path = top3_dir / f"timeform_top3_{today_str}.csv"
    top3_parquet_path = settings.PROCESSED_TIMEFORM_TOP3_DIR / f"timeform_top3_{today_str}.parquet"

    save_timeform_forecast(updates, raw_path=forecast_path, parquet_path=forecast_parquet_path)
    save_timeform_top3(updates, raw_path=top3_path, parquet_path=top3_parquet_path)

    logger.info(
        "Atualizacao Timeform concluida. CSVs/Parquet gerados: {}, {}",
        forecast_path,
        top3_path,
    )


if __name__ == "__main__":
    main()