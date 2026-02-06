from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    import sys

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.greyhounds.config import settings


DatasetConfig = tuple[str, Path, Path, str]


DATASETS: list[DatasetConfig] = [
    ("signals", settings.RAW_SIGNALS_DIR, settings.PROCESSED_SIGNALS_DIR, "signals_*.csv"),
    ("timeform_top3", settings.RAW_TIMEFORM_TOP3_DIR, settings.PROCESSED_TIMEFORM_TOP3_DIR, "timeform_top3_*.csv"),
    ("timeform_forecast", settings.RAW_TIMEFORM_FORECAST_DIR, settings.PROCESSED_TIMEFORM_FORECAST_DIR, "TimeformForecast_*.csv"),
    ("race_links", settings.RAW_RACE_LINKS_DIR, settings.PROCESSED_RACE_LINKS_DIR, "race_links_*.csv"),
    ("betfair_result", settings.RAW_RESULT_DIR, settings.PROCESSED_RESULT_DIR, "*.csv"),
]


def convert_file(csv_path: Path, parquet_path: Path, *, force: bool, compression: str) -> tuple[int, str]:
    if not parquet_path.exists():
        action_on_success = "convert"
    elif force:
        action_on_success = "reconvert_force"
    else:
        csv_mtime = os.path.getmtime(csv_path)
        parquet_mtime = os.path.getmtime(parquet_path)
        if parquet_mtime >= csv_mtime:
            logger.debug("Parquet já existe e está atualizado, pulando: {}", parquet_path)
            return 0, "skip"
        logger.debug("CSV mais novo que parquet, reconvertendo: {}", csv_path)
        action_on_success = "reconvert"
    try:
        df = pd.read_csv(
            csv_path,
            encoding=settings.CSV_ENCODING,
            engine="python",
            on_bad_lines="skip",
        )
    except Exception as exc:
        logger.error("Falha ao ler {}: {}", csv_path, exc)
        return 0, "error"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression=compression)
    logger.info(
        "Convertido para Parquet: {} -> {} ({} linhas)",
        csv_path.name,
        parquet_path.name,
        len(df),
    )
    return len(df), action_on_success


def convert_dataset(dataset: DatasetConfig, *, force: bool, compression: str) -> int:
    name, raw_dir, processed_dir, pattern = dataset
    if not raw_dir.exists():
        logger.warning("Diretório bruto inexistente para {}: {}", name, raw_dir)
        return 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    matched = list(raw_dir.glob(pattern))
    if not matched:
        logger.info("Nenhum arquivo encontrado para {} em {}", name, raw_dir)
        return 0

    skipped = reconverted = converted = 0
    for csv_path in sorted(matched):
        parquet_path = processed_dir / f"{csv_path.stem}.parquet"
        lines, action = convert_file(csv_path, parquet_path, force=force, compression=compression)
        total_rows += lines
        if action == "skip":
            skipped += 1
        elif action == "reconvert":
            reconverted += 1
        elif action == "convert":
            converted += 1
        # "reconvert_force" e "error" não entram no resumo
    logger.success("Conversão {} concluída. Total de linhas: {}", name, total_rows)
    logger.debug(
        "Arquivos pulados: {} | Reconversões por CSV mais novo: {} | Convertidos do zero: {}",
        skipped,
        reconverted,
        converted,
    )
    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte históricos CSV dos galgos para Parquet mantendo instantâneos brutos.",
    )
    parser.add_argument(
        "--dataset",
        choices=[name for name, *_ in DATASETS],
        help="Converte apenas um dataset específico.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa mesmo que o Parquet já exista.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Algoritmo de compressão Parquet (padrão: snappy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Iniciando conversão CSV -> Parquet (greyhounds)...")

    targets: Iterable[DatasetConfig]
    if args.dataset:
        targets = [dataset for dataset in DATASETS if dataset[0] == args.dataset]
    else:
        targets = DATASETS

    total = 0
    for dataset in targets:
        total += convert_dataset(dataset, force=args.force, compression=args.compression)

    logger.success("Conversão finalizada. Linhas totais processadas: {}", total)


if __name__ == "__main__":
    main()

