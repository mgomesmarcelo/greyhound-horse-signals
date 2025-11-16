from __future__ import annotations

import sys
from typing import Callable, Iterable, Tuple

from loguru import logger

from src.greyhounds.config import settings
from src.greyhounds.scrape_betfair_index import main as scrape_betfair_index
from src.greyhounds.scrape_timeform_update import main as scrape_timeform_update


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


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    steps: list[Step] = [
        ("scrape_betfair_index", scrape_betfair_index),
        ("scrape_timeform_update", scrape_timeform_update),
    ]

    _run_steps(steps)
    logger.info("Pipeline diario de greyhounds concluido.")


if __name__ == "__main__":
    main()
