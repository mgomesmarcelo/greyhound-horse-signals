from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.greyhounds.config import settings

TARGET_COLUMNS = [
    "event_id",
    "menu_hint",
    "event_name",
    "event_dt",
    "selection_id",
    "selection_name",
    "win_lose",
    "bsp",
    "pptradedvol",
]

_BSP_TWO_DEC_REGEX = re.compile(r"^\d+\.\d{2}$")
_BANNED_REGEX = re.compile(r"\((?:AUS|NZL)\)")


def _canonicalize_column(name: str) -> str | None:
    raw = (name or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", raw)
    mapping = {
        "eventid": "event_id",
        "_eventid": "event_id",
        "menuhint": "menu_hint",
        "eventname": "event_name",
        "eventdt": "event_dt",
        "selectionid": "selection_id",
        "selectionname": "selection_name",
        "winlose": "win_lose",
        "win_lose": "win_lose",
        "bsp": "bsp",
        "pptradedvol": "pptradedvol",
    }
    return mapping.get(normalized, None)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        canonical = _canonicalize_column(str(col))
        if canonical:
            rename_map[col] = canonical
    return df.rename(columns=rename_map)


def is_already_clean(df: pd.DataFrame) -> bool:
    if df.columns.tolist() != TARGET_COLUMNS:
        return False
    if "bsp" not in df.columns:
        return False
    df_as_str = df.astype(str, copy=False)
    if df_as_str.apply(lambda col: col.str.contains(_BANNED_REGEX, na=False)).any(axis=1).any():
        return False
    series = df["bsp"]
    if len(series) == 0:
        return True
    mask_notna = series.notna()
    if not mask_notna.any():
        return True
    bsp_as_str = series.astype(str)
    return _BSP_TWO_DEC_REGEX.fullmatch("0.00") is not None and bsp_as_str[mask_notna].map(
        lambda text: bool(_BSP_TWO_DEC_REGEX.fullmatch(text))
    ).all()


def format_bsp_to_two_decimals(value) -> str:
    if pd.isna(value):
        return ""
    try:
        num = float(value)
        return f"{num:.2f}"
    except Exception:
        return ""


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[TARGET_COLUMNS]
    mask_banned = out.astype(str).apply(lambda col: col.str.contains(_BANNED_REGEX, na=False)).any(axis=1)
    if mask_banned.any():
        out = out.loc[~mask_banned].reset_index(drop=True)
    out["bsp"] = out["bsp"].map(format_bsp_to_two_decimals)
    return out


def clean_results_dir(result_dir: Path, force: bool = False) -> int:
    changed = 0
    for csv_path in sorted(result_dir.glob("*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                encoding=settings.CSV_ENCODING,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", csv_path.name, exc)
            continue

        df = normalize_columns(df)

        missing = [col for col in TARGET_COLUMNS if col not in df.columns]
        if missing:
            logger.error(
                "Pulado {}: colunas ausentes após normalização ({})",
                csv_path.name,
                ", ".join(missing),
            )
            continue

        if not force and is_already_clean(df):
            logger.debug("Pulado (já limpo): {}", csv_path.name)
            continue

        clean_df = clean_dataframe(df)
        try:
            clean_df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
            changed += 1
            logger.info("Arquivo limpo: {} ({} linhas)", csv_path.name, len(clean_df))
        except Exception as exc:
            logger.error("Falha ao escrever {}: {}", csv_path.name, exc)

    return changed


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    force = False
    if "--force" in argv:
        force = True

    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    result_dir = settings.RAW_RESULT_DIR
    if not result_dir.exists():
        logger.error("Diretório não encontrado: {}", result_dir)
        return 1

    logger.info("Limpando CSVs em: {} (force={})", result_dir, force)
    changed = clean_results_dir(result_dir, force=force)
    logger.info("Concluído. Arquivos alterados: {}", changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

