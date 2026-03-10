"""
Gera parquets enriquecidos (category, category_token, num_runners, date_dt, race_ts, track_key, race_iso)
a partir do indice result_index e dos parquets de sinais, para carregamento rapido no Streamlit.

Uso:
  python scripts/horses/build_horses_signals_enriched.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.horses.config import settings
from src.horses.utils.text import normalize_track_name

_RACE_TIME_ISO_FORMAT = "%Y-%m-%dT%H:%M"
PROCESSED_SIGNALS_DIR = settings.DATA_DIR / "processed" / "signals"
CACHE_DIR = settings.DATA_DIR / "processed" / "cache"
SIGNALS_ENRICHED_DIR = settings.DATA_DIR / "processed" / "signals_enriched"


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    index_path = CACHE_DIR / "result_index.parquet"
    if not index_path.exists():
        logger.error(
            "Indice nao encontrado: {}. Execute antes: python scripts/horses/build_horses_result_index.py",
            index_path,
        )
        return 1

    result_index = pd.read_parquet(index_path)

    for provider in ["timeform", "sportinglife"]:
        signals_dir = PROCESSED_SIGNALS_DIR / provider
        if not signals_dir.exists():
            logger.debug("Diretorio inexistente: {}", signals_dir)
            continue

        parquet_files = sorted(signals_dir.glob("signals_*.parquet"))
        if not parquet_files:
            logger.debug("Nenhum signals_*.parquet em {}", signals_dir)
            continue

        out_dir = SIGNALS_ENRICHED_DIR / provider
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in parquet_files:
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Falha ao ler {}: {}", path.name, exc)
                continue

            if df.empty:
                df.to_parquet(out_dir / path.name, index=False, compression="snappy")
                continue

            df = df.copy()

            if "date" in df.columns and "date_dt" not in df.columns:
                df["date_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            if "race_time_iso" in df.columns and "race_ts" not in df.columns:
                df["race_ts"] = pd.to_datetime(
                    df["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce"
                )

            if "track_name" not in df.columns or "race_time_iso" not in df.columns:
                df.to_parquet(out_dir / path.name, index=False, compression="snappy")
                continue

            df["track_key"] = df["track_name"].astype(str).map(normalize_track_name)
            df["race_iso"] = df["race_time_iso"].astype(str).str.strip()

            for col in ("category", "category_token", "num_runners"):
                if col in df.columns:
                    df = df.drop(columns=[col])

            merge_cols = ["track_key", "race_iso"]
            index_sub = result_index[merge_cols + ["category", "category_token", "num_runners"]].drop_duplicates(
                subset=merge_cols, keep="first"
            )
            df = df.merge(index_sub, on=merge_cols, how="left")

            df["category"] = df["category"].fillna("").astype(str)
            df["category_token"] = df["category_token"].fillna("").astype(str)
            df["num_runners"] = pd.to_numeric(df["num_runners"], errors="coerce")

            out_path = out_dir / path.name
            df.to_parquet(out_path, index=False, compression="snappy")
            logger.info("Enriched {} -> {} ({} linhas)", path.name, out_path, len(df))

    logger.info("Enriquecimento concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
