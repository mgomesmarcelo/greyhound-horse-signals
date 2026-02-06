from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.greyhounds.config import settings
from src.greyhounds.utils.text import normalize_track_name

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

BANNED_TRACK_PREFIXES = (
    "aus ",
    "australia ",
    "nz ",
    "new zealand ",
)
BANNED_TRACK_NAMES = {
    "",
    "none",
    "murray bridge",
}

STATE_FILE_NAME = ".clean_state.json"


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


def _load_state(result_dir: Path) -> dict[str, Any]:
    path = result_dir / STATE_FILE_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Falha ao ler {} (ignorando cache): {}", path.name, exc)
        return {}


def _save_state(result_dir: Path, state: dict[str, Any]) -> None:
    path = result_dir / STATE_FILE_NAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _file_fingerprint(p: Path) -> dict[str, int]:
    st = p.stat()
    return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def is_already_clean(df: pd.DataFrame) -> bool:
    # Colunas exatamente na ordem esperada
    if df.columns.tolist() != TARGET_COLUMNS:
        return False

    # BANNED_REGEX só nas colunas textuais relevantes (bem mais barato)
    for col in ("menu_hint", "event_name", "selection_name"):
        if col in df.columns:
            if df[col].astype(str).str.contains(_BANNED_REGEX, na=False).any():
                return False

    # BSP precisa estar no formato X.XX (ou vazio)
    if "bsp" not in df.columns:
        return False

    bsp = df["bsp"].astype(str)

    # normaliza possíveis "nan" vindos de leitura ruim; aqui a leitura já é string,
    # mas deixo robusto
    bsp = bsp.replace({"nan": "", "None": ""})

    # permite vazio; valida os não-vazios
    non_empty = bsp.str.strip().ne("")
    if not non_empty.any():
        return True

    return bsp[non_empty].map(lambda t: bool(_BSP_TWO_DEC_REGEX.fullmatch(t.strip()))).all()


def format_bsp_to_two_decimals(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    try:
        num = float(text)
        return f"{num:.2f}"
    except Exception:
        return ""


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # garante todas as colunas
    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    # mantém somente as colunas alvo e na ordem
    out = out[TARGET_COLUMNS]

    # remove linhas com (AUS|NZL) em campos textuais
    mask_banned = False
    for col in ("menu_hint", "event_name", "selection_name"):
        mask_banned_col = out[col].astype(str).str.contains(_BANNED_REGEX, na=False)
        mask_banned = mask_banned_col if isinstance(mask_banned, bool) else (mask_banned | mask_banned_col)

    if not isinstance(mask_banned, bool) and mask_banned.any():
        out = out.loc[~mask_banned].reset_index(drop=True)

    def _is_banned_track(menu_hint: str) -> bool:
        text = str(menu_hint or "").strip()
        lower = text.lower()
        if any(lower.startswith(prefix) for prefix in BANNED_TRACK_PREFIXES):
            return True
        normalized = normalize_track_name(text)
        return normalized.lower() in BANNED_TRACK_NAMES

    track_mask = out["menu_hint"].apply(_is_banned_track)
    if track_mask.any():
        out = out.loc[~track_mask].reset_index(drop=True)

    # formata BSP como texto X.XX
    out["bsp"] = out["bsp"].map(format_bsp_to_two_decimals)

    return out


def clean_results_dir(result_dir: Path, force: bool = False) -> int:
    changed = 0
    state = _load_state(result_dir)

    csv_paths = sorted(result_dir.glob("*.csv"))
    for csv_path in csv_paths:
        name = csv_path.name

        # FAST SKIP: se arquivo não mudou desde a última vez, nem abre.
        if not force and name in state:
            fp_now = _file_fingerprint(csv_path)
            fp_old = state.get(name, {})
            if fp_old.get("mtime_ns") == fp_now["mtime_ns"] and fp_old.get("size") == fp_now["size"]:
                logger.debug("Pulado (cache, sem mudanças): {}", name)
                continue

        try:
            # MUITO IMPORTANTE: ler como string para não destruir "2.00" -> 2.0
            df = pd.read_csv(
                csv_path,
                encoding=settings.CSV_ENCODING,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                keep_default_na=False,  # mantém "" como ""
            )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", name, exc)
            continue

        df = normalize_columns(df)

        missing = [col for col in TARGET_COLUMNS if col not in df.columns]
        if missing:
            logger.error(
                "Pulado {}: colunas ausentes após normalização ({})",
                name,
                ", ".join(missing),
            )
            continue

        if not force and is_already_clean(df):
            logger.debug("Pulado (já limpo): {}", name)
            # mesmo assim atualiza cache (caso ele tenha sido criado fora do script)
            state[name] = _file_fingerprint(csv_path)
            continue

        clean_df = clean_dataframe(df)

        try:
            clean_df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
            changed += 1
            logger.info("Arquivo limpo: {} ({} linhas)", name, len(clean_df))
            state[name] = _file_fingerprint(csv_path)
        except Exception as exc:
            logger.error("Falha ao escrever {}: {}", name, exc)

    # salva cache no final (bem mais rápido que salvar a cada arquivo)
    try:
        _save_state(result_dir, state)
    except Exception as exc:
        logger.warning("Falha ao salvar {}: {}", STATE_FILE_NAME, exc)

    return changed


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    force = "--force" in argv

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
