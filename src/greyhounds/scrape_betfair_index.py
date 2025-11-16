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


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    today_str = date.today().isoformat()

    base_dir: Path = ensure_data_dir("greyhounds")
    race_links_dir = base_dir / "race_links"
    race_links_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Iniciando scrape do indice da Betfair: https://www.betfair.com/exchange/plus/en/greyhound-racing-betting-4339"
    )
    rows = scrape_betfair_index()

    out_path = race_links_dir / f"race_links_{today_str}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"race_links.csv salvo em: {out_path}")


if __name__ == "__main__":
    main()