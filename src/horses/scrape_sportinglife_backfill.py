from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

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


def _checkpoint_file() -> Path:
    checkpoint = settings.DATA_DIR / "horses" / "sportinglife_backfill_checkpoint.txt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint


def _load_checkpoint() -> set[str]:
    ckpt = _checkpoint_file()
    if ckpt.exists():
        try:
            return {line.strip() for line in ckpt.read_text(encoding="utf-8").splitlines() if line.strip()}
        except Exception:
            return set()
    return set()


def _append_checkpoint(date_str: str) -> None:
    ckpt = _checkpoint_file()
    with ckpt.open("a", encoding="utf-8") as fp:
        fp.write(date_str + "\n")


def _date_range(start: datetime, end: datetime) -> Iterator[datetime]:
    step = timedelta(days=1)
    if start <= end:
        current = start
        while current <= end:
            yield current
            current += step
    else:
        current = start
        while current >= end:
            yield current
            current -= step


def _write_outputs(
    date_str: str,
    rows_top3: Iterable[dict],
    rows_fc: Iterable[dict],
    links: Iterable[str],
    out_top3: Path,
    out_fc: Path,
    out_links: Path,
) -> tuple[int, int, int]:
    list_top3 = list(rows_top3)
    list_fc = list(rows_fc)
    list_links = list(links)

    df_top3 = pd.DataFrame(list_top3) if list_top3 else pd.DataFrame(
        columns=["track_name", "race_time_iso", "TimeformTop1", "TimeformTop2", "TimeformTop3"]
    )
    df_fc = pd.DataFrame(list_fc) if list_fc else pd.DataFrame(
        columns=["track_name", "race_time_iso", "SportingLifeForecast"]
    )
    df_links = pd.DataFrame({"url": list_links}) if list_links else pd.DataFrame(columns=["url"])

    df_top3.to_csv(out_top3, index=False, encoding=settings.CSV_ENCODING)
    df_fc.to_csv(out_fc, index=False, encoding=settings.CSV_ENCODING)
    df_links.to_csv(out_links, index=False, encoding=settings.CSV_ENCODING)

    return len(df_top3), len(df_fc), len(df_links)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    parser = argparse.ArgumentParser(description="Backfill Sporting Life para intervalo de datas.")
    parser.add_argument("--years", type=int, default=3, help="Quantidade padrão de anos para retroceder.")
    parser.add_argument("--start", type=str, default=None, help="Data inicial (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Data final (YYYY-MM-DD).")
    parser.add_argument("--max-workers", type=int, default=1, help="Concorrência por dia (>=1).")
    args = parser.parse_args(argv)

    today = datetime.now().date()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = today
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = today - timedelta(days=365 * int(args.years))

    done_dates = _load_checkpoint()
    workers = max(1, int(getattr(args, "max_workers", 1)))

    for current in _date_range(start_date, end_date):
        date_str = current.strftime("%Y-%m-%d")
        if date_str in done_dates:
            logger.info("Pulado (checkpoint): {}", date_str)
            continue

        logger.info("Processando Sporting Life {} (max_workers={})", date_str, workers)
        out_top3, out_fc, out_links = _out_paths(date_str)

        try:
            rows_top3, rows_fc, links = scrape_day(date_str, max_workers=workers)
        except Exception as exc:
            logger.error("Falha ao raspar {}: {}", date_str, exc)
            continue

        len_top3, len_fc, len_links = _write_outputs(date_str, rows_top3, rows_fc, links, out_top3, out_fc, out_links)
        logger.info(
            "Dia {} salvo: {} top3, {} forecasts, {} links",
            date_str,
            len_top3,
            len_fc,
            len_links,
        )

        if len_top3 > 0 or len_fc > 0 or len_links > 0:
            _append_checkpoint(date_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

