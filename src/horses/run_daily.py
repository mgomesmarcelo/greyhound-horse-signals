from __future__ import annotations

import sys
from typing import Callable, Iterable, Tuple

from loguru import logger

from src.horses.config import settings
from src.horses.scrape_betfair_races import main as scrape_betfair_races
from src.horses.scrape_timeform_update import main as scrape_timeform_update

Step = Tuple[str, Callable[[], None]]


def _run_steps(steps: Iterable[Step]) -> None:
    for label, func in steps:
        logger.info("Iniciando passo: {}", label)
        try:
            func()
        except SystemExit as exc:
            code = int(exc.code or 0)
            if code != 0:
                logger.error("Passo '{}' finalizou com codigo {}", label, code)
                raise
        except Exception as exc:
            logger.exception("Erro inesperado ao executar '{}': {}", label, exc)
            raise SystemExit(1)
        else:
            logger.info("Passo '{}' concluido com sucesso.", label)


from scripts.generate_daily_entries import process_strategies_for_sport
from datetime import date

def generate_daily_entries_horses():
    process_strategies_for_sport("horses", date.today().isoformat())

def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)
    logger.add(settings.DATA_DIR / "daily_run.log", level=settings.LOG_LEVEL, rotation="10 MB", retention="10 days")

    steps: list[Step] = [
        ("scrape_betfair_races", scrape_betfair_races),
        ("scrape_timeform_update", scrape_timeform_update),
        ("generate_daily_entries", generate_daily_entries_horses),
    ]

    _run_steps(steps)
    logger.info("Pipeline diario de horses concluido.")


if __name__ == "__main__":
    main()
