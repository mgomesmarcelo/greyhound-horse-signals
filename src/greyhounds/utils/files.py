from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.greyhounds.config import settings


_INVALID_CHARS = r"[^\w\-\. ]+"
_WHITESPACE = re.compile(r"\s+")


def sanitize_name(name: str) -> str:
    clean = re.sub(_INVALID_CHARS, " ", name)
    clean = _WHITESPACE.sub(" ", clean).strip()
    return clean.replace(" ", "_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_links_csv(day_dir: Path, rows: Iterable[Dict[str, object]]) -> Path:
    csv_path = day_dir / "race_links.csv"
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
    else:
        pd.DataFrame([]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
    return csv_path


def append_or_create_csv(csv_path: Path, row: Dict[str, object]) -> None:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
    else:
        pd.DataFrame([row]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)


def upsert_single_row_csv(csv_path: Path, update_row: Dict[str, object]) -> None:
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            base: Dict[str, object] = {}
            if not df_existing.empty:
                base = df_existing.iloc[-1].to_dict()
            base.update(update_row)
            pd.DataFrame([base]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
            return
        except Exception:
            pass
    pd.DataFrame([update_row]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)


def condense_csv_to_single_row(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
            return
        aggregated: Dict[str, object] = {}
        for col in df.columns:
            series = df[col]
            val = None
            for item in series[::-1]:
                if pd.notna(item) and str(item).strip() != "":
                    val = item
                    break
            aggregated[col] = val if val is not None else (series.iloc[-1] if len(series) > 0 else None)
        pd.DataFrame([aggregated]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
    except Exception:
        return


def upsert_row_by_keys(csv_path: Path, new_row: Dict[str, object], key_fields: List[str]) -> None:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and all(k in df.columns for k in key_fields):
                mask = pd.Series([True] * len(df))
                for key in key_fields:
                    mask &= df[key].astype(str) == str(new_row.get(key, ""))
                if mask.any():
                    out_df = pd.concat([df[~mask], pd.DataFrame([new_row])], ignore_index=True)
                    out_df = out_df[list(new_row.keys())]
                    out_df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
                    return
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
            return
        except Exception:
            pass
    pd.DataFrame([new_row]).to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)