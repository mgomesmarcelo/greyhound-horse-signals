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
    save_timeform_cards,
    scrape_timeform_for_races,
)

def scrape_all_races_with_fallback(rows):
    from src.greyhounds.scrapers.betfair_race import scrape_betfair_race
    from src.greyhounds.utils.selenium_driver import build_chrome_driver
    
    failed_rows = []
    
    driver = build_chrome_driver()
    try:
        for row in rows:
            race_url = row.get("race_url")
            track_name = row.get("track_name")
            race_time_iso = row.get("race_time_iso")
            
            if not race_url:
                continue
                
            try:
                result = scrape_betfair_race(race_url, track_name, race_time_iso, driver=driver)
                # Verifica se extraiu o Forecast
                if result.get("TimeformForecast"):
                    yield result
                else:
                    logger.warning(f"Forecast nao encontrado na Betfair para {track_name} {race_time_iso}. Marcado para fallback.")
                    failed_rows.append(row)
            except Exception as e:
                logger.error(f"Erro ao extrair da Betfair para {track_name} {race_time_iso}: {e}")
                failed_rows.append(row)
    finally:
        driver.quit()
        
    if failed_rows:
        logger.info(f"{len(failed_rows)} corridas marcadas para fallback no Timeform. Iniciando fallback...")
        yield from scrape_timeform_for_races(failed_rows)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    logger.info("Iniciando atualizacao Betfair/Timeform para greyhounds...")

    today_str = date.today().isoformat()

    base_dir: Path = ensure_data_dir("greyhounds")

    forecast_dir = settings.RAW_TIMEFORM_FORECAST_DIR
    top3_dir = settings.RAW_TIMEFORM_TOP3_DIR
    cards_dir = base_dir / "timeform_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    race_links_dir = base_dir / "race_links"
    race_links_dir.mkdir(parents=True, exist_ok=True)

    race_links_path = race_links_dir / f"race_links_{today_str}.csv"

    if not race_links_path.exists():
        logger.warning(f"Arquivo não encontrado: {race_links_path}. Gerando índice da Betfair agora...")
        rows = scrape_betfair_index()
        pd.DataFrame(rows).to_csv(race_links_path, index=False)
        logger.info(f"race_links gerado em {race_links_path}")

    df = pd.read_csv(race_links_path)
    rows = df.to_dict("records")

    out_forecast = forecast_dir / f"TimeformForecast_{today_str}.csv"
    out_forecast_pq = settings.PROCESSED_TIMEFORM_FORECAST_DIR / f"TimeformForecast_{today_str}.parquet"

    out_top3 = top3_dir / f"timeform_top3_{today_str}.csv"
    out_top3_pq = settings.PROCESSED_TIMEFORM_TOP3_DIR / f"timeform_top3_{today_str}.parquet"

    out_cards = cards_dir / f"timeform_cards_{today_str}.csv"
    out_cards_pq = cards_dir / f"timeform_cards_{today_str}.parquet"

    from itertools import tee
    gen_forecast, gen_top3, gen_cards = tee(scrape_all_races_with_fallback(rows), 3)

    logger.info("Salvando TimeformForecast...")
    save_timeform_forecast(gen_forecast, raw_path=out_forecast, parquet_path=out_forecast_pq)

    logger.info("Salvando TimeformTop3...")
    save_timeform_top3(gen_top3, raw_path=out_top3, parquet_path=out_top3_pq)

    logger.info("Salvando TimeformCards (Traps e Categorias)...")
    save_timeform_cards(gen_cards, raw_path=out_cards, parquet_path=out_cards_pq)

    logger.info("Processo concluido com sucesso.")


if __name__ == "__main__":
    main()