import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)

from datetime import date

import pandas as pd
from loguru import logger

from src.core.config import ensure_data_dir
from src.horses.config import settings
from src.horses.scrapers.betfair_index import scrape_betfair_index


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    today_str = date.today().isoformat()
    base_dir = ensure_data_dir("horses")
    race_links_dir = base_dir / "race_links"
    race_links_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Iniciando scrape do indice da Betfair: {}", settings.BETFAIR_HORSE_RACING_URL)
    rows = scrape_betfair_index()

    out_path = race_links_dir / f"race_links_{today_str}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("race_links salvo em: {} ({} corridas)", out_path, len(rows))


if __name__ == "__main__":
    main()
