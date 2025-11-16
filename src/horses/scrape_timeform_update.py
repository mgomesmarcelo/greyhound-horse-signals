import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

from loguru import logger

from src.core.config import ensure_data_dir
from src.horses.config import HORSES_DATA_DIR, settings
from src.horses.scrapers.betfair_index import scrape_betfair_index
from src.horses.scrapers.timeform import scrape_timeform_for_races
from src.horses.utils.dates import today_str
from src.horses.utils.files import ensure_dir, upsert_row_by_keys

ensure_data_dir()
HORSES_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _forecast_output_csv_path() -> Path:
    forecast_dir = HORSES_DATA_DIR / "TimeformForecast"
    ensure_dir(forecast_dir)
    return forecast_dir / f"TimeformForecast_{today_str()}.csv"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    logger.info("Iniciando atualizacao Timeform para horses...")

    races = scrape_betfair_index()
    if not races:
        logger.warning("Nenhuma corrida para processar no Timeform.")
        return

    updates = scrape_timeform_for_races(races)
    if not updates:
        logger.warning("Nenhuma atualizacao do Timeform foi encontrada.")
        return

    forecast_path = _forecast_output_csv_path()
    for update in updates:
        update["source"] = "timeform"
        upsert_row_by_keys(forecast_path, update, key_fields=["track_name", "race_time_iso", "source"])

    logger.info("Atualizacao Timeform concluida. CSV gerado: {}", forecast_path)


if __name__ == "__main__":
    main()