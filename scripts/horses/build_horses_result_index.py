"""
Pre-processa todos os parquets de Result em um unico indice (track_key, race_iso) com
category, category_token e num_runners, para evitar varrer milhares de arquivos no Streamlit.

Uso:
  python scripts/horses/build_horses_result_index.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.horses.config import settings
from src.horses.analysis.signals import _extract_track_from_menu_hint, _to_iso_series


def _classify_category_from_event_name(event_name: str) -> tuple[str, str]:
    txt = str(event_name or "").strip()
    txt_wo_dist = re.sub(r"^\s*\d+(?:m\d+f|m|f)\s+", "", txt, flags=re.IGNORECASE)
    s = txt_wo_dist.lower()

    def _result(token: str) -> tuple[str, str]:
        letter_map = {
            "G1": "G", "G2": "G", "G3": "G",
            "LISTED": "L",
            "HCP": "H", "HCP_CHS": "H", "HCP_HRD": "H",
            "NOV": "N", "NOV_CHS": "N", "NOV_HRD": "N",
            "MDN": "M",
            "NURSERY": "Y",
            "COND": "C",
            "STKS": "S",
            "NHF": "F",
            "CHS": "C",
            "HRD": "R",
            "OTHER": "O",
        }
        return (letter_map.get(token, "O"), token)

    if re.search(r"\bgroup\s*(1|2|3)\b", s):
        m = re.search(r"\bgroup\s*(1|2|3)\b", s)
        if m:
            return _result(f"G{m.group(1)}")
    if re.search(r"\bg([123])\b", s):
        m = re.search(r"\bg([123])\b", s)
        if m:
            return _result(f"G{m.group(1)}")
    if re.search(r"\blisted\b", s):
        return _result("LISTED")
    if re.search(r"\b(nhf|inhf|bumper)\b", s):
        return _result("NHF")
    has_chs = bool(re.search(r"\b(chase|chases|chs)\b", s))
    has_hrd = bool(re.search(r"\b(hurdle|hurdles|hrd|hdl)\b", s))
    if re.search(r"\b(handicap|hcap|hcp)\b", s):
        if has_chs:
            return _result("HCP_CHS")
        if has_hrd:
            return _result("HCP_HRD")
        return _result("HCP")
    if re.search(r"\bnov(?:ice)?\b", s):
        if has_chs:
            return _result("NOV_CHS")
        if has_hrd:
            return _result("NOV_HRD")
        return _result("NOV")
    if re.search(r"\b(maiden|mdn)\b", s):
        return _result("MDN")
    if re.search(r"\bnursery\b", s):
        return _result("NURSERY")
    if re.search(r"\b(conditions|cond)\b", s):
        return _result("COND")
    if re.search(r"\b(stakes|stks)\b", s):
        return _result("STKS")
    if has_chs:
        return _result("CHS")
    if has_hrd:
        return _result("HRD")
    return _result("OTHER")


def _extract_category_letter(event_name: str) -> str:
    letter, _ = _classify_category_from_event_name(event_name)
    return letter


def _extract_category_token(event_name: str) -> str:
    _, token = _classify_category_from_event_name(event_name)
    return token


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    result_dir = settings.DATA_DIR / "processed" / "Result"
    if not result_dir.exists():
        logger.error("Diretorio nao encontrado: {}", result_dir)
        return 1

    parquet_files = sorted(result_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("Nenhum parquet em {}", result_dir)
        return 0

    base_columns = ["menu_hint", "event_dt", "event_name"]
    optional_columns = ["num_runners", "runner"]

    t0 = time.perf_counter()
    chunks = []

    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    for path in parquet_files:
        try:
            if pq is not None:
                schema = pq.ParquetFile(path).schema.names
                usecols = [c for c in base_columns if c in schema]
                if "num_runners" in schema:
                    usecols.append("num_runners")
                elif "runner" in schema:
                    usecols.append("runner")
            else:
                usecols = list(base_columns)
            if len(usecols) < len(base_columns):
                logger.debug("Pulando {} (colunas insuficientes)", path.name)
                continue
            df = pd.read_parquet(path, columns=usecols)
        except Exception as exc:
            logger.warning("Falha ao ler {}: {}", path.name, exc)
            continue

        if df.empty:
            continue

        if "runner" in df.columns and "num_runners" not in df.columns:
            df["num_runners"] = df["runner"]

        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = _to_iso_series(df["event_dt"].astype(str))
        df["category"] = df["event_name"].astype(str).map(_extract_category_letter)
        df["category_token"] = df["event_name"].astype(str).map(_extract_category_token)

        mask = (df["track_key"].astype(str).str.strip() != "") & (df["race_iso"].astype(str).str.strip() != "")
        df = df.loc[mask]
        if df.empty:
            continue
        chunks.append(df)

    if not chunks:
        logger.warning("Nenhum dado valido encontrado.")
        return 0

    combined = pd.concat(chunks, ignore_index=True)

    has_num_runners_col = "num_runners" in combined.columns
    if has_num_runners_col:
        combined["num_runners"] = pd.to_numeric(combined["num_runners"], errors="coerce")

    agg_dict = {"category": "first", "category_token": "first"}
    if has_num_runners_col:
        agg_dict["num_runners"] = "first"
    else:
        agg_dict["menu_hint"] = "count"

    index_df = combined.groupby(["track_key", "race_iso"], as_index=False).agg(agg_dict)
    if not has_num_runners_col:
        index_df["num_runners"] = index_df["menu_hint"].astype("Int64")
        index_df = index_df.drop(columns=["menu_hint"])
    else:
        index_df["num_runners"] = index_df["num_runners"].fillna(0).astype("Int64")

    out_dir = settings.DATA_DIR / "processed" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result_index.parquet"
    index_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Indice construido: {} corridas em {:.2f}s -> {}",
        len(index_df),
        elapsed,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
