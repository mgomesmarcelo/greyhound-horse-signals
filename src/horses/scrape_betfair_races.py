import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

from loguru import logger

from src.core.config import DATA_DIR, ensure_data_dir
from src.horses.config import settings
from src.horses.scrapers.betfair_index import scrape_betfair_index
from src.horses.scrapers.betfair_race import scrape_betfair_race
from src.horses.utils.dates import today_str
from src.horses.utils.files import ensure_dir, upsert_row_by_keys
from src.horses.utils.selenium_driver import build_chrome_driver

ensure_data_dir()
HORSES_DATA_DIR = DATA_DIR / "horses"
HORSES_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _daily_output_csv_path() -> Path:
    output_dir = HORSES_DATA_DIR / "betfair_top3"
    ensure_dir(output_dir)
    return output_dir / f"betfair_top3_{today_str()}.csv"


def _timeform_top3_csv_path() -> Path:
    """CSV onde load_timeform_top3() le os dados (para horses a fonte e Betfair)."""
    output_dir = HORSES_DATA_DIR / "timeform_top3"
    ensure_dir(output_dir)
    return output_dir / f"timeform_top3_{today_str()}.csv"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    logger.info("Iniciando coleta Betfair Timeform Top3...")

    races = scrape_betfair_index()
    if not races:
        logger.warning("Nenhuma corrida encontrada no indice da Betfair.")
        return

    driver = build_chrome_driver()
    try:
        output_csv = _daily_output_csv_path()
        top3_csv = _timeform_top3_csv_path()
        for row in races:
            track = str(row.get("track_name", "unknown_track"))
            race_url = str(row.get("race_url", ""))
            race_time_iso = str(row.get("race_time_iso", ""))
            if not race_url:
                continue

            result = scrape_betfair_race(race_url, driver=driver)
            if result is None:
                logger.warning(f"Sem dados para esta corrida. Pulando: {race_url}")
                time.sleep(0.5)
                continue

            result.update(
                {
                    "track_name": track,
                    "race_time_iso": race_time_iso,
                    "source": "betfair",
                }
            )

            upsert_row_by_keys(output_csv, result, key_fields=["track_name", "race_time_iso", "source"])

            names = result.get("TimeformPrev_list")
            if isinstance(names, str):
                names = [n.strip() for n in names.split(";") if n.strip()]
            if not isinstance(names, list):
                names = []
            if names:
                top3_row = {
                    "track_name": track,
                    "race_time_iso": race_time_iso,
                    "TimeformTop1": names[0] if len(names) > 0 else "",
                    "TimeformTop2": names[1] if len(names) > 1 else "",
                    "TimeformTop3": names[2] if len(names) > 2 else "",
                }
                upsert_row_by_keys(top3_csv, top3_row, key_fields=["track_name", "race_time_iso"])

            logger.debug(f"Atualizado: {output_csv}")
            time.sleep(1.0)
    finally:
        driver.quit()

    logger.info("Coleta Betfair concluida. CSVs gerados: {} e {}", output_csv, top3_csv)


if __name__ == "__main__":
    main()