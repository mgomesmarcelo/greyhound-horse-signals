from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dateutil import parser as date_parser
from loguru import logger

from src.greyhounds.config import RULE_LABELS, settings
from src.greyhounds.utils.text import clean_greyhound_name, normalize_track_name
from src.greyhounds.utils.files import write_dataframe_snapshots

_TRAP_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")
_TRAP_NUMBER_RE = re.compile(r"^\s*([1-6])[\.\s]+")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _strip_trap_prefix(name: str) -> str:
    return _TRAP_PREFIX_RE.sub("", name or "").strip()


def _extract_trap_number(name: str) -> int | None:
    """Extrai número da trap a partir do prefixo do nome, limitado a 1-6."""
    match = _TRAP_NUMBER_RE.match(name or "")
    if not match:
        return None
    try:
        val = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return val if 1 <= val <= 6 else None


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    return normalize_track_name(str(menu_hint or ""))


def _to_iso_yyyy_mm_dd_thh_mm(value: str) -> str:
    try:
        dt = date_parser.parse(value, dayfirst=True)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _extract_category_letter(event_name: str) -> str:
    txt = str(event_name or "").strip()
    m = re.match(r"^([A-Za-z]+)", txt)
    token = m.group(1).upper() if m else ""
    return token[:1] if token else ""


def _extract_category_token(event_name: str) -> str:
    txt = str(event_name or "").strip()
    m = re.match(r"^([A-Za-z]+\d*)", txt)
    return m.group(1).upper() if m else ""


_FORECAST_ITEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(.+)$", re.IGNORECASE)


def _parse_forecast_all(text: str) -> List[Dict]:
    """Parseia o texto completo do TimeformForecast e retorna lista de itens com odds e rank.

    Formato esperado: "TimeformForecast : 2.88 Coolruss Izzy, 5.00 Day Tripper, ..."
    Nao ha trap no forecast. Itens malformados sao ignorados (apenas o item, nao a corrida).
    """
    if not isinstance(text, str):
        return []
    stripped = re.sub(r"(?i)\btimeformforecast\s*:\s*", "", text.strip())
    parts = [part.strip() for part in stripped.split(",") if isinstance(part, str) and part.strip()]
    items: List[Dict] = []
    for rank, part in enumerate(parts, start=1):
        match = _FORECAST_ITEM_RE.match(part)
        if not match:
            continue
        try:
            odds_val = float(match.group(1))
        except (TypeError, ValueError):
            continue
        name_raw = (match.group(2) or "").strip()
        if not name_raw:
            continue
        name_clean = clean_greyhound_name(name_raw)
        if not name_clean:
            continue
        items.append({
            "forecast_rank": rank,
            "forecast_odds": odds_val,
            "forecast_name_clean": name_clean,
            "forecast_name_raw": name_raw,
        })
    return items


def _parse_forecast_top3(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    stripped = re.sub(r"(?i)\btimeformforecast\s*:\s*", "", text.strip())
    parts = [part.strip() for part in stripped.split(",") if isinstance(part, str) and part.strip()]
    names: List[str] = []
    for part in parts:
        match = re.match(r"^\s*\d+(?:\.\d+)?\s+(.+)$", part)
        if match:
            candidate = match.group(1).strip()
        else:
            candidate = re.sub(r"\s*\([^\)]*\)\s*$", "", part).strip()
        candidate = _strip_trap_prefix(candidate)
        cleaned = clean_greyhound_name(candidate)
        if cleaned and cleaned not in names:
            names.append(cleaned)
        if len(names) >= 3:
            break
    return names


@dataclass
class RunnerBF:
    selection_name_raw: str
    selection_name_clean: str
    pptradedvol: float
    bsp: float
    win_lose: int
    trap_number: int | None


def _iter_result_paths(pattern: str) -> List[Path]:
    parquet_paths = sorted(settings.PROCESSED_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_paths:
        return parquet_paths
    return sorted(settings.RAW_RESULT_DIR.glob(f"{pattern}.csv"))


def load_betfair_win() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    all_files = _iter_result_paths("dwbfgreyhoundwin*")
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for path in all_files:
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", path.name, exc)
            continue

        required_cols = ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = df["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = (
            df["selection_name_raw"].map(_strip_trap_prefix).map(clean_greyhound_name)
        )
        df["trap_number"] = df["selection_name_raw"].map(_extract_trap_number)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), group in df.groupby(["track_key", "race_iso"], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, row in group.iterrows():
                name_clean = row["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                runners[name_clean] = RunnerBF(
                    selection_name_raw=row["selection_name_raw"],
                    selection_name_clean=name_clean,
                    pptradedvol=float(row["pptradedvol"]),
                    bsp=float(row["bsp"]) if pd.notna(row["bsp"]) else float("nan"),
                    win_lose=int(row["win_lose"]),
                    trap_number=int(row["trap_number"]) if pd.notna(row["trap_number"]) else None,
                )

    logger.info("Betfair WIN index criado: {} corridas", len(index))
    return index


def load_betfair_place() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    all_files = _iter_result_paths("dwbfgreyhoundplace*")
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for path in all_files:
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", path.name, exc)
            continue

        required_cols = ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = df["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = (
            df["selection_name_raw"].map(_strip_trap_prefix).map(clean_greyhound_name)
        )
        df["trap_number"] = df["selection_name_raw"].map(_extract_trap_number)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), group in df.groupby(["track_key", "race_iso"], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, row in group.iterrows():
                name_clean = row["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                runners[name_clean] = RunnerBF(
                    selection_name_raw=row["selection_name_raw"],
                    selection_name_clean=name_clean,
                    pptradedvol=float(row["pptradedvol"]),
                    bsp=float(row["bsp"]) if pd.notna(row["bsp"]) else float("nan"),
                    win_lose=int(row["win_lose"]),
                    trap_number=int(row["trap_number"]) if pd.notna(row["trap_number"]) else None,
                )

    logger.info("Betfair PLACE index criado: {} corridas", len(index))
    return index


def build_category_index_from_results() -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Constroi indice (track_key, race_iso) -> {"letter": ..., "token": ...} a partir dos
    arquivos de resultado Betfair WIN (dwbfgreyhoundwin*). Parquet preferencial, csv fallback.
    Usado uma vez por execucao em generate_signals() para enriquecer e persistir no Parquet.
    """
    all_files = _iter_result_paths("dwbfgreyhoundwin*")
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    columns = ("menu_hint", "event_dt", "event_name")

    for path in all_files:
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.warning("build_category_index: falha ao ler {}: {}", path.name, exc)
            continue
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        if df.empty:
            continue
        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = df["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        df["cat_letter"] = df["event_name"].astype(str).map(_extract_category_letter)
        df["cat_token"] = df["event_name"].astype(str).map(_extract_category_token)
        mask = (df["track_key"].astype(str) != "") & (df["race_iso"].astype(str) != "")
        df = df.loc[mask].drop_duplicates(subset=["track_key", "race_iso"], keep="first")
        if df.empty:
            continue
        keys = list(zip(df["track_key"].astype(str), df["race_iso"].astype(str)))
        values = [
            {"letter": str(a), "token": str(b)}
            for a, b in zip(df["cat_letter"], df["cat_token"])
        ]
        for k, v in zip(keys, values):
            if k not in mapping:
                mapping[k] = v

    logger.info("Indice de categoria (WIN) criado: {} corridas", len(mapping))
    return mapping


def load_timeform_top3() -> List[dict]:
    tf_dir = settings.PROCESSED_TIMEFORM_TOP3_DIR
    parquet_paths = sorted(tf_dir.glob("timeform_top3_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        csv_dir = settings.RAW_TIMEFORM_TOP3_DIR
        sources = sorted(csv_dir.glob("timeform_top3_*.csv"))
        use_parquet = False

    rows: List[dict] = []

    for path in sources:
        try:
            if use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", path.name, exc)
            continue

        expected_cols = [
            "track_name",
            "race_time_iso",
            "TimeformTop1",
            "TimeformTop2",
            "TimeformTop3",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

        for _, row in df.iterrows():
            track = normalize_track_name(str(row.get("track_name", "")))
            race_iso = str(row.get("race_time_iso", ""))
            names = [
                clean_greyhound_name(str(row.get(col, "")))
                for col in ["TimeformTop1", "TimeformTop2", "TimeformTop3"]
            ]
            if not track or not race_iso or not any(names):
                continue
            rows.append(
                {
                    "track_key": track,
                    "race_iso": race_iso,
                    "top_names": names,
                    "raw": row.to_dict(),
                }
            )

    logger.info("Timeform Top3 carregado: {} corridas", len(rows))
    return rows


def load_timeform_forecast_top3() -> List[dict]:
    tf_dir = settings.PROCESSED_TIMEFORM_FORECAST_DIR
    parquet_paths = sorted(tf_dir.glob("TimeformForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        csv_dir = settings.RAW_TIMEFORM_FORECAST_DIR
        sources = sorted(csv_dir.glob("TimeformForecast_*.csv"))
        use_parquet = False

    rows: List[dict] = []
    for path in sources:
        try:
            if use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", path.name, exc)
            continue

        for col in ["track_name", "race_time_iso", "TimeformForecast"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, row in df.iterrows():
            track = normalize_track_name(str(row.get("track_name", "")))
            race_iso = str(row.get("race_time_iso", ""))
            names = _parse_forecast_top3(str(row.get("TimeformForecast", "")))
            if not track or not race_iso or not names:
                continue
            while len(names) < 3:
                names.append("")
            raw_row = {
                "track_name": track,
                "race_time_iso": race_iso,
                "TimeformTop1": names[0],
                "TimeformTop2": names[1],
                "TimeformTop3": names[2],
            }
            rows.append(
                {
                    "track_key": track,
                    "race_iso": race_iso,
                    "top_names": names[:3],
                    "raw": raw_row,
                }
            )

    logger.info("Timeform Forecast(Top3) carregado: {} corridas", len(rows))
    return rows


def load_timeform_forecast_all() -> pd.DataFrame:
    """Carrega TimeformForecast (parquet preferencial, csv fallback), parseia o campo textual
    TimeformForecast e retorna um DataFrame com track_key, race_iso e forecast_items (list de
    dicts por corrida: forecast_rank, forecast_odds, forecast_name_clean, forecast_name_raw).
    """
    tf_dir = settings.PROCESSED_TIMEFORM_FORECAST_DIR
    parquet_paths = sorted(tf_dir.glob("TimeformForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        csv_dir = settings.RAW_TIMEFORM_FORECAST_DIR
        sources = sorted(csv_dir.glob("TimeformForecast_*.csv"))
        use_parquet = False

    rows: List[dict] = []
    for path in sources:
        try:
            if use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
        except Exception as exc:
            logger.error("Falha ao ler {}: {}", path.name, exc)
            continue

        for col in ["track_name", "race_time_iso", "TimeformForecast"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, row in df.iterrows():
            track = normalize_track_name(str(row.get("track_name", "")))
            race_iso = str(row.get("race_time_iso", ""))
            if not track or not race_iso:
                continue
            forecast_items = _parse_forecast_all(str(row.get("TimeformForecast", "")))
            if not forecast_items:
                continue # evita carregar/iterar corrida sem itens
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "forecast_items": forecast_items,
            })

    logger.info("Timeform Forecast (all) carregado: {} corridas", len(rows))
    return pd.DataFrame(rows)


def _forecast_all_df_to_tf_rows(df: pd.DataFrame) -> List[dict]:
    """Converte o DataFrame de load_timeform_forecast_all() para lista de dicts no formato
    esperado por _calc_signals_forecast_odds_for_race: track_key, race_iso, forecast_items, raw.
    """
    if df.empty:
        return []
    rows: List[dict] = []
    for _, row in df.iterrows():
        track_key = row.get("track_key", "")
        race_iso = row.get("race_iso", "")
        forecast_items = row.get("forecast_items") or []
        raw = {
            "track_name": track_key,
            "race_time_iso": race_iso,
        }
        rows.append({
            "track_key": track_key,
            "race_iso": race_iso,
            "forecast_items": forecast_items,
            "raw": raw,
        })
    return rows


def _signal_race_selection_key(row: dict) -> Tuple[str, str, str]:
    """Chave (race_id, selection_id) para um sinal: race_time_iso, track_name, nome do galgo."""
    race_iso = row.get("race_time_iso") or ""
    track = row.get("track_name") or ""
    sel = (
        row.get("forecast_name_clean")
        or row.get("back_target_name")
        or row.get("lay_target_name")
        or ""
    )
    return (race_iso, track, sel)


def _dedupe_forecast_odds_signals_by_race_selection(
    result: List[dict], track_key: str, race_iso: str
) -> List[dict]:
    """Garante no maximo um sinal por (race_id, selection_id). Duplicatas: mantem o primeiro, log warning."""
    if not result:
        return result
    seen: Dict[Tuple[str, str, str], bool] = {}
    unique: List[dict] = []
    for row in result:
        key = _signal_race_selection_key(row)
        if key in seen:
            logger.warning(
                "forecast_odds: duplicata ignorada (nao deveria haver back e lay para o mesmo galgo/corrida): race_iso={!r} track={!r} selection={!r} entry_type={!r}",
                race_iso,
                track_key,
                key[2],
                row.get("entry_type"),
            )
            continue
        seen[key] = True
        unique.append(row)
    return unique


def _assert_forecast_odds_unique_race_selection(df: pd.DataFrame) -> None:
    """Valida que forecast_odds nao tem duas linhas para o mesmo (race_id, selection_id). Log error se tiver."""
    if df.empty or "race_time_iso" not in df.columns:
        return
    keys: List[Tuple[str, str, str]] = []
    for _, row in df.iterrows():
        keys.append(_signal_race_selection_key(row))
    if len(keys) != len(set(keys)):
        counts = Counter(keys)
        dupes = [k for k, c in counts.items() if c > 1]
        logger.error(
            "forecast_odds: violacao de unicidade (race_id, selection_id): existem {} chaves duplicadas; exemplos: {}",
            len(dupes),
            dupes[:5],
        )


def _calc_signals_forecast_odds_for_race(
    tf_row: dict,
    bf_win_index: Dict[Tuple[str, str], Dict[str, RunnerBF]],
    bf_place_index: Dict[Tuple[str, str], Dict[str, RunnerBF]] | None,
    market: str,
    rule: str = "forecast_odds",
    cat_index: Dict[Tuple[str, str], Dict[str, str]] | None = None,
) -> List[dict]:
    """Gera sinais por runner do forecast que casou com Betfair: no maximo 1 linha por galgo/corrida.
    value_ratio = back_target_bsp / forecast_odds. Se value_ratio >= FORECAST_ODDS_BACK_MIN_VALUE_RATIO
    -> back; se value_ratio <= FORECAST_ODDS_LAY_MAX_VALUE_RATIO -> lay; senao nao gera sinal (zona morta).
    PnL/ROI e schema iguais ao dashboard.
    """
    track_key = tf_row["track_key"]
    race_iso = tf_row["race_iso"]
    forecast_items = tf_row.get("forecast_items") or []
    raw = tf_row.get("raw") or {}

    if market == "place" and bf_place_index is not None:
        group = bf_place_index.get((track_key, race_iso))
    else:
        group = bf_win_index.get((track_key, race_iso))

    if not group:
        return []

    num_runners = len(group)
    stake_ref = 1.0
    liability_ref = 1.0
    legacy_scale = 10.0
    commission_rate = 0.02

    tf_top1 = forecast_items[0].get("forecast_name_clean") or "" if len(forecast_items) >= 1 else ""
    tf_top2 = forecast_items[1].get("forecast_name_clean") or "" if len(forecast_items) >= 2 else ""
    tf_top3 = forecast_items[2].get("forecast_name_clean") or "" if len(forecast_items) >= 3 else ""

    _cat = (cat_index or {}).get((track_key, race_iso), {}) or {}
    category = _cat.get("letter", "") or ""
    category_token = _cat.get("token", "") or ""

    base_neutral = {
        "date": race_iso.split("T")[0] if race_iso else "",
        "track_name": raw.get("track_name", track_key),
        "race_time_iso": race_iso,
        "tf_top1": tf_top1,
        "tf_top2": tf_top2,
        "tf_top3": tf_top3,
        "vol_top1": 0.0,
        "vol_top2": 0.0,
        "vol_top3": 0.0,
        "second_name_by_volume": "",
        "third_name_by_volume": "",
        "ratio_second_over_third": 0.0,
        "pct_diff_second_vs_third": 0.0,
        "leader_name_by_volume": "",
        "leader_volume_share_pct": 0.0,
        "num_runners": int(num_runners),
        "category": category,
        "category_token": category_token,
        "market": market,
        "rule": rule,
        "rule_label": RULE_LABELS.get(rule, rule),
        "total_matched_volume": 0.0,
    }

    result: List[dict] = []
    for item in forecast_items:
        name_clean = item.get("forecast_name_clean")
        if not name_clean:
            continue
        forecast_odds_val = item.get("forecast_odds")
        if forecast_odds_val is None or (isinstance(forecast_odds_val, float) and (forecast_odds_val <= 0 or pd.isna(forecast_odds_val))):
            continue
        forecast_odds_val = float(forecast_odds_val)
        forecast_rank = item.get("forecast_rank")
        if forecast_rank is None:
            continue
        forecast_rank = int(forecast_rank)

        runner = group.get(name_clean) if isinstance(group, dict) else None
        if not runner or pd.isna(runner.bsp):
            continue

        odd = float(runner.bsp)
        target_win_lose = int(runner.win_lose)
        trap_number = runner.trap_number

        value_ratio = odd / forecast_odds_val if forecast_odds_val > 0 else float("nan")
        value_log = math.log(value_ratio) if value_ratio > 0 else float("nan")

        # P&L por linha (nao assume back+lay para o mesmo galgo). Back: stake fixa; lucro vitoria =
        # (odds-1)*stake*(1-comissao), perda derrota = -stake. Lay: liability fixa; stake = liability/(odds-1);
        # perda na vitoria do cavalo = -liability; lucro na derrota = stake*(1-comissao).
        back_is_green = target_win_lose == 1
        if back_is_green:
            back_pnl_stake_ref = stake_ref * max(0.0, odd - 1.0) * (1.0 - commission_rate)
        else:
            back_pnl_stake_ref = -stake_ref

        liability_from_stake_ref = stake_ref * max(0.0, odd - 1.0)
        stake_from_liab_ref = liability_ref / max(0.001, odd - 1.0)
        if target_win_lose == 1:
            lay_pnl_stake_ref = -liability_from_stake_ref
            lay_pnl_liab_ref = -liability_ref
            lay_is_green = False
        else:
            lay_pnl_stake_ref = stake_ref * (1.0 - commission_rate)
            lay_pnl_liab_ref = stake_from_liab_ref * (1.0 - commission_rate)
            lay_is_green = True

        stake_fix10 = stake_ref * legacy_scale
        liability_fix10 = liability_ref * legacy_scale
        liability_from_stake10 = liability_from_stake_ref * legacy_scale
        stake_from_liab10 = stake_from_liab_ref * legacy_scale
        back_pnl_stake10 = back_pnl_stake_ref * legacy_scale
        lay_pnl_stake10 = lay_pnl_stake_ref * legacy_scale
        lay_pnl_liab10 = lay_pnl_liab_ref * legacy_scale

        extra = {
            "forecast_rank": forecast_rank,
            "forecast_odds": forecast_odds_val,
            "forecast_name_clean": name_clean,
            "value_ratio": value_ratio,
            "value_log": value_log,
        }

        out_back = {
            **base_neutral,
            **extra,
            "entry_type": "back",
            "back_target_name": name_clean,
            "back_target_bsp": round(odd, 2),
            "trap_number": trap_number if trap_number is not None else pd.NA,
            "lay_target_name": "",
            "lay_target_bsp": float("nan"),
            "stake_ref": round(stake_ref, 2),
            "liability_from_stake_ref": 0.0,
            "stake_for_liability_ref": 0.0,
            "liability_ref": 0.0,
            "pnl_stake_ref": round(back_pnl_stake_ref, 4),
            "pnl_liability_ref": 0.0,
            "roi_row_stake_ref": round(back_pnl_stake_ref / stake_ref if stake_ref > 0 else 0.0, 4),
            "roi_row_liability_ref": 0.0,
            "roi_row_exposure_ref": 0.0,
            "stake_fixed_10": round(stake_fix10, 2),
            "liability_from_stake_fixed_10": 0.0,
            "stake_for_liability_10": 0.0,
            "liability_fixed_10": 0.0,
            "win_lose": target_win_lose,
            "is_green": back_is_green,
            "pnl_stake_fixed_10": round(back_pnl_stake10, 2),
            "pnl_liability_fixed_10": 0.0,
            "roi_row_stake_fixed_10": round(back_pnl_stake10 / stake_fix10 if stake_fix10 > 0 else 0.0, 4),
            "roi_row_liability_fixed_10": 0.0,
            "roi_row_exposure_fixed_10": 0.0,
        }

        out_lay = {
            **base_neutral,
            **extra,
            "entry_type": "lay",
            "back_target_name": "",
            "back_target_bsp": float("nan"),
            "lay_target_name": name_clean,
            "lay_target_bsp": round(odd, 2),
            "trap_number": trap_number if trap_number is not None else pd.NA,
            "stake_ref": round(stake_ref, 2),
            "liability_from_stake_ref": round(liability_from_stake_ref, 4),
            "stake_for_liability_ref": round(stake_from_liab_ref, 4),
            "liability_ref": round(liability_ref, 2),
            "pnl_stake_ref": round(lay_pnl_stake_ref, 4),
            "pnl_liability_ref": round(lay_pnl_liab_ref, 4),
            "roi_row_stake_ref": round(lay_pnl_stake_ref / stake_ref if stake_ref > 0 else 0.0, 4),
            "roi_row_liability_ref": round(lay_pnl_liab_ref / liability_ref if liability_ref > 0 else 0.0, 4),
            "roi_row_exposure_ref": round(
                lay_pnl_stake_ref / liability_from_stake_ref if liability_from_stake_ref > 0 else 0.0,
                4,
            ),
            "stake_fixed_10": round(stake_fix10, 2),
            "liability_from_stake_fixed_10": round(liability_from_stake10, 2),
            "stake_for_liability_10": round(stake_from_liab10, 2),
            "liability_fixed_10": round(liability_fix10, 2),
            "win_lose": target_win_lose,
            "is_green": lay_is_green,
            "pnl_stake_fixed_10": round(lay_pnl_stake10, 2),
            "pnl_liability_fixed_10": round(lay_pnl_liab10, 2),
            "roi_row_stake_fixed_10": round(lay_pnl_stake10 / stake_fix10 if stake_fix10 > 0 else 0.0, 4),
            "roi_row_liability_fixed_10": round(lay_pnl_liab10 / liability_fix10 if liability_fix10 > 0 else 0.0, 4),
            "roi_row_exposure_fixed_10": round(
                lay_pnl_stake10 / liability_from_stake10 if liability_from_stake10 > 0 else 0.0,
                4,
            ),
        }

        # Direcao por value_ratio (back_target_bsp / forecast_odds): acima do minimo -> back;
        # abaixo do maximo -> lay; entre os dois -> sem sinal
        if pd.isna(value_ratio):
            continue
        if value_ratio >= settings.FORECAST_ODDS_BACK_MIN_VALUE_RATIO:
            result.append(out_back)
        elif value_ratio <= settings.FORECAST_ODDS_LAY_MAX_VALUE_RATIO:
            result.append(out_lay)
        # else: zona morta (entre lay_max e back_min), nao gera sinal

    # Garantia: no maximo um sinal por (race_id, selection_id) para forecast_odds
    result = _dedupe_forecast_odds_signals_by_race_selection(result, track_key, race_iso)
    return result


def _build_betfair_direct_rows(
    bf_win_index: Dict[Tuple[str, str], Dict[str, RunnerBF]]
) -> List[dict]:
    rows: List[dict] = []
    for (track_key, race_iso), runners in bf_win_index.items():
        if not runners:
            continue
        sorted_runners = sorted(
            (
                runner
                for runner in runners.values()
                if isinstance(runner.selection_name_clean, str) and runner.selection_name_clean
            ),
            key=lambda r: float(r.pptradedvol) if pd.notna(r.pptradedvol) else 0.0,
            reverse=True,
        )
        if len(sorted_runners) < 3:
            continue
        top_names = [runner.selection_name_clean for runner in sorted_runners[:3]]
        raw_row = {
            "track_name": track_key,
            "race_time_iso": race_iso,
            "TimeformTop1": top_names[0] if len(top_names) > 0 else "",
            "TimeformTop2": top_names[1] if len(top_names) > 1 else "",
            "TimeformTop3": top_names[2] if len(top_names) > 2 else "",
        }
        rows.append(
            {
                "track_key": track_key,
                "race_iso": race_iso,
                "top_names": top_names,
                "raw": raw_row,
            }
        )
    return rows


def _calc_signals_for_race(
    tf_row: dict,
    bf_win_index: Dict[Tuple[str, str], Dict[str, RunnerBF]],
    bf_place_index: Dict[Tuple[str, str], Dict[str, RunnerBF]] | None = None,
    market: str = "win",
    rule: str = "terceiro_queda50",
    leader_share_min: float = 0.5,
    cat_index: Dict[Tuple[str, str], Dict[str, str]] | None = None,
) -> List[dict]:
    track_key = tf_row["track_key"]
    race_iso = tf_row["race_iso"]
    top_names = [name for name in tf_row["top_names"] if isinstance(name, str) and name]
    group = bf_win_index.get((track_key, race_iso))
    if not group:
        return []

    num_runners = len(group)
    total_vol_race = 0.0
    triples: List[Tuple[str, float, float]] = []
    for name in top_names:
        runner = group.get(name)
        if not runner or pd.isna(runner.bsp):
            return []
        triples.append((name, max(0.0, float(runner.pptradedvol)), float(runner.bsp)))
    for runner in group.values():
        try:
            total_vol_race += max(0.0, float(runner.pptradedvol))
        except (TypeError, ValueError):
            continue

    if len(triples) < 3:
        return []

    triples_sorted = sorted(triples, key=lambda item: item[1], reverse=True)
    first, second, third = triples_sorted[0], triples_sorted[1], triples_sorted[2]

    vol2, vol3 = second[1], third[1]
    pct_diff = (vol2 - vol3) / vol2 if vol2 > 0 else float("inf")
    ratio = (vol2 / vol3) if vol3 > 0 else float("inf")

    target_name_clean: str | None = None
    target_bsp_win: float | None = None
    leader_share = 0.0

    if rule == "terceiro_queda50":
        if vol3 <= 0 or pct_diff <= 0.5:
            return []
        target_name_clean = third[0]
        target_bsp_win = third[2]
    else:
        leader_share = (first[1] / total_vol_race) if total_vol_race > 0 else 0.0
        if leader_share < float(leader_share_min):
            return []
        target_name_clean = first[0]
        target_bsp_win = first[2]

    if not target_name_clean:
        return []

    if market == "place" and bf_place_index is not None:
        target_runner = bf_place_index.get((track_key, race_iso), {}).get(target_name_clean)
    else:
        target_runner = bf_win_index.get((track_key, race_iso), {}).get(target_name_clean)
    if not target_runner:
        return []

    target_win_lose = int(target_runner.win_lose)
    trap_number = target_runner.trap_number
    odd = (
        float(target_runner.bsp)
        if market == "place" and target_runner is not None
        else float(target_bsp_win or 0.0)
    )

    # Nova referência: 1 unidade (mantemos colunas _fixed_10 por compatibilidade)
    stake_ref = 1.0
    liability_ref = 1.0
    legacy_scale = 10.0  # TODO: descontinuar colunas *_fixed_10 após migração completa
    commission_rate = 0.02

    # BACK - referência 1
    back_is_green = target_win_lose == 1
    if back_is_green:
        back_profit_gross_ref = stake_ref * max(0.0, odd - 1.0)
        back_pnl_stake_ref = back_profit_gross_ref * (1.0 - commission_rate)
    else:
        back_pnl_stake_ref = -stake_ref

    # LAY - referência 1
    liability_from_stake_ref = stake_ref * max(0.0, odd - 1.0)
    stake_from_liab_ref = liability_ref / max(0.001, odd - 1.0)

    if target_win_lose == 1:
        lay_pnl_stake_ref = -liability_from_stake_ref
        lay_pnl_liab_ref = -liability_ref
        lay_is_green = False
    else:
        lay_pnl_stake_ref = stake_ref * (1.0 - commission_rate)
        lay_pnl_liab_ref = stake_from_liab_ref * (1.0 - commission_rate)
        lay_is_green = True

    # Colunas legado (x10) preservadas
    stake_fix10 = stake_ref * legacy_scale
    liability_fix10 = liability_ref * legacy_scale
    liability_from_stake10 = liability_from_stake_ref * legacy_scale
    stake_from_liab10 = stake_from_liab_ref * legacy_scale
    back_pnl_stake10 = back_pnl_stake_ref * legacy_scale
    lay_pnl_stake10 = lay_pnl_stake_ref * legacy_scale
    lay_pnl_liab10 = lay_pnl_liab_ref * legacy_scale

    raw = tf_row["raw"]

    _cat = (cat_index or {}).get((track_key, race_iso), {}) or {}
    category = _cat.get("letter", "") or ""
    category_token = _cat.get("token", "") or ""

    def _vol_for(name_raw: object) -> float:
        name = clean_greyhound_name(str(name_raw)) if isinstance(name_raw, str) else ""
        return next((vol for runner_name, vol, _ in triples if runner_name == name), 0.0)

    base = {
        "date": race_iso.split("T")[0],
        "track_name": raw.get("track_name", ""),
        "race_time_iso": race_iso,
        "tf_top1": raw.get("TimeformTop1", ""),
        "tf_top2": raw.get("TimeformTop2", ""),
        "tf_top3": raw.get("TimeformTop3", ""),
        "vol_top1": _vol_for(raw.get("TimeformTop1")),
        "vol_top2": _vol_for(raw.get("TimeformTop2")),
        "vol_top3": _vol_for(raw.get("TimeformTop3")),
        "second_name_by_volume": second[0],
        "third_name_by_volume": third[0],
        "ratio_second_over_third": round(ratio, 2),
        "pct_diff_second_vs_third": round(pct_diff * 100.0, 2),
        "leader_name_by_volume": first[0],
        "leader_volume_share_pct": round(leader_share * 100.0, 2),
        "num_runners": int(num_runners),
        "category": category,
        "category_token": category_token,
        "market": market,
        "rule": rule,
        "rule_label": RULE_LABELS.get(rule, rule),
        "total_matched_volume": round(total_vol_race, 2),
    }

    out_back = {
        **base,
        "entry_type": "back",
        "back_target_name": target_name_clean,
        "back_target_bsp": round(odd, 2),
        "trap_number": trap_number if trap_number is not None else pd.NA,
        "lay_target_name": "",
        "lay_target_bsp": float("nan"),
        # Referência 1
        "stake_ref": round(stake_ref, 2),
        "liability_from_stake_ref": 0.0,
        "stake_for_liability_ref": 0.0,
        "liability_ref": 0.0,
        "pnl_stake_ref": round(back_pnl_stake_ref, 4),
        "pnl_liability_ref": 0.0,
        "roi_row_stake_ref": round(back_pnl_stake_ref / stake_ref if stake_ref > 0 else 0.0, 4),
        "roi_row_liability_ref": 0.0,
        "roi_row_exposure_ref": 0.0,
        # Legado (x10) - TODO: remover *_fixed_10 após migração
        "stake_fixed_10": round(stake_fix10, 2),
        "liability_from_stake_fixed_10": 0.0,
        "stake_for_liability_10": 0.0,
        "liability_fixed_10": 0.0,
        "win_lose": target_win_lose,
        "is_green": back_is_green,
        "pnl_stake_fixed_10": round(back_pnl_stake10, 2),
        "pnl_liability_fixed_10": 0.0,
        "roi_row_stake_fixed_10": round(back_pnl_stake10 / stake_fix10 if stake_fix10 > 0 else 0.0, 4),
        "roi_row_liability_fixed_10": 0.0,
        "roi_row_exposure_fixed_10": 0.0,
    }

    out_lay = {
        **base,
        "entry_type": "lay",
        "back_target_name": "",
        "back_target_bsp": float("nan"),
        "lay_target_name": target_name_clean,
        "lay_target_bsp": round(odd, 2),
        "trap_number": trap_number if trap_number is not None else pd.NA,
        # Referência 1
        "stake_ref": round(stake_ref, 2),
        "liability_from_stake_ref": round(liability_from_stake_ref, 4),
        "stake_for_liability_ref": round(stake_from_liab_ref, 4),
        "liability_ref": round(liability_ref, 2),
        "pnl_stake_ref": round(lay_pnl_stake_ref, 4),
        "pnl_liability_ref": round(lay_pnl_liab_ref, 4),
        "roi_row_stake_ref": round(
            lay_pnl_stake_ref / stake_ref if stake_ref > 0 else 0.0,
            4,
        ),
        "roi_row_liability_ref": round(
            lay_pnl_liab_ref / liability_ref if liability_ref > 0 else 0.0,
            4,
        ),
        "roi_row_exposure_ref": round(
            lay_pnl_stake_ref / liability_from_stake_ref if liability_from_stake_ref > 0 else 0.0,
            4,
        ),
        # Legado (x10) - TODO: remover *_fixed_10 após migração
        "stake_fixed_10": round(stake_fix10, 2),
        "liability_from_stake_fixed_10": round(liability_from_stake10, 2),
        "stake_for_liability_10": round(stake_from_liab10, 2),
        "liability_fixed_10": round(liability_fix10, 2),
        "win_lose": target_win_lose,
        "is_green": lay_is_green,
        "pnl_stake_fixed_10": round(lay_pnl_stake10, 2),
        "pnl_liability_fixed_10": round(lay_pnl_liab10, 2),
        "roi_row_stake_fixed_10": round(
            lay_pnl_stake10 / stake_fix10 if stake_fix10 > 0 else 0.0,
            4,
        ),
        "roi_row_liability_fixed_10": round(
            lay_pnl_liab10 / liability_fix10 if liability_fix10 > 0 else 0.0,
            4,
        ),
        "roi_row_exposure_fixed_10": round(
            lay_pnl_stake10 / liability_from_stake10 if liability_from_stake10 > 0 else 0.0,
            4,
        ),
    }

    return [out_back, out_lay]


def generate_signals(
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
    leader_share_min: float = 0.5,
    entry_type: str = "both",
) -> pd.DataFrame:
    bf_win_index = load_betfair_win()
    bf_place_index = load_betfair_place() if market == "place" else None
    cat_index = build_category_index_from_results()

    if source == "forecast" and rule == "forecast_odds":
        df_forecast = load_timeform_forecast_all()
        tf_rows = _forecast_all_df_to_tf_rows(df_forecast)
        use_forecast_odds = True
    elif source == "forecast":
        tf_rows = load_timeform_forecast_top3()
        use_forecast_odds = False
    elif source == "betfair_resultado":
        tf_rows = _build_betfair_direct_rows(bf_win_index)
        use_forecast_odds = False
    else:
        tf_rows = load_timeform_top3()
        use_forecast_odds = False

    signals_rows: List[dict] = []
    for row in tf_rows:
        if use_forecast_odds:
            results = _calc_signals_forecast_odds_for_race(
                row, bf_win_index, bf_place_index, market=market, rule=rule, cat_index=cat_index
            )
        else:
            results = _calc_signals_for_race(
                row,
                bf_win_index,
                bf_place_index,
                market=market,
                rule=rule,
                leader_share_min=leader_share_min,
                cat_index=cat_index,
            )
        for result in results:
            if entry_type in ("both", result.get("entry_type")):
                signals_rows.append(result)

    df = pd.DataFrame(signals_rows)
    if rule == "forecast_odds" and not df.empty:
        _assert_forecast_odds_unique_race_selection(df)
    logger.info(
        "Sinais encontrados (source={}, market={}, rule={}, leader_share_min={}, entry_type={}): {}",
        source,
        market,
        rule,
        leader_share_min,
        entry_type,
        len(df),
    )
    return df


def write_signals_csv(
    df: pd.DataFrame,
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
) -> Path:
    raw_dir = settings.RAW_SIGNALS_DIR
    processed_dir = settings.PROCESSED_SIGNALS_DIR
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)
    raw_path = raw_dir / f"signals_{source}_{market}_{rule}.csv"
    parquet_path = processed_dir / f"signals_{source}_{market}_{rule}.parquet"

    df = df.copy()
    df["source"] = source
    df["market"] = market
    df["rule"] = rule
    df["rule_label"] = RULE_LABELS.get(rule, rule)

    if df.empty:
        df_sorted = pd.DataFrame(
            [],
            columns=[
                "date",
                "track_name",
                "race_time_iso",
                "tf_top1",
                "tf_top2",
                "tf_top3",
                "vol_top1",
                "vol_top2",
                "vol_top3",
                "second_name_by_volume",
                "third_name_by_volume",
                "ratio_second_over_third",
                "pct_diff_second_vs_third",
                "num_runners",
                "category",
                "category_token",
                "trap_number",
                "lay_target_name",
                "lay_target_bsp",
                "back_target_name",
                "back_target_bsp",
                "leader_name_by_volume",
                "leader_volume_share_pct",
                "total_matched_volume",
                "stake_ref",
                "liability_from_stake_ref",
                "stake_for_liability_ref",
                "liability_ref",
                "pnl_stake_ref",
                "pnl_liability_ref",
                "roi_row_stake_ref",
                "roi_row_liability_ref",
                "roi_row_exposure_ref",
                "stake_fixed_10",
                "liability_from_stake_fixed_10",
                "stake_for_liability_10",
                "liability_fixed_10",
                "win_lose",
                "is_green",
                "pnl_stake_fixed_10",
                "pnl_liability_fixed_10",
                "roi_row_stake_fixed_10",
                "roi_row_liability_fixed_10",
                "roi_row_exposure_fixed_10",
                "forecast_rank",
                "forecast_odds",
                "forecast_name_clean",
                "value_ratio",
                "value_log",
                "source",
                "market",
                "rule",
                "rule_label",
                "entry_type",
            ],
        )
    else:
        if "category" not in df.columns:
            df["category"] = ""
        if "category_token" not in df.columns:
            df["category_token"] = ""
        df_sorted = df.sort_values(
            ["date", "track_name", "race_time_iso", "entry_type"]
        ).reset_index(drop=True)

    if "num_runners" not in df_sorted.columns:
        df_sorted["num_runners"] = pd.Series(dtype="Int64") if df_sorted.empty else pd.array([pd.NA] * len(df_sorted), dtype="Int64")
    if "category" not in df_sorted.columns:
        df_sorted["category"] = ""
    if "category_token" not in df_sorted.columns:
        df_sorted["category_token"] = ""

    write_dataframe_snapshots(df_sorted, raw_path=raw_path, parquet_path=parquet_path)
    logger.info(
        "Sinais gerados. CSV bruto: {} | Parquet: {} ({} linhas)",
        raw_path,
        parquet_path,
        len(df_sorted),
    )
    return parquet_path


__all__ = [
    "RunnerBF",
    "_signal_race_selection_key",
    "build_category_index_from_results",
    "generate_signals",
    "load_betfair_place",
    "load_betfair_win",
    "load_timeform_forecast_all",
    "load_timeform_forecast_top3",
    "load_timeform_top3",
    "write_signals_csv",
]


def _debug_load_timeform_forecast_all() -> None:
    """Teste rapido: chama load_timeform_forecast_all(), imprime quantidade de corridas,
    distribuicao de itens por corrida e 2 exemplos de forecast_items.
    Executar: python -m src.greyhounds.analysis.signals
    """
    df = load_timeform_forecast_all()
    n = len(df)
    print(f"Corridas carregadas: {n}")
    if n == 0:
        return
    counts = df["forecast_items"].map(len)
    print(
        f"Distribuicao de itens por corrida: min={counts.min()}, "
        f"median={counts.median():.0f}, max={counts.max()}"
    )
    for i, row in df.head(2).iterrows():
        print(f"\nExemplo corrida {i + 1}: track_key={row['track_key']!r}, race_iso={row['race_iso']!r}")
        print("forecast_items:", row["forecast_items"])


def _debug_forecast_odds_small_sample() -> None:
    """Valida geracao de sinais forecast_odds: total de linhas, entry_type, value_ratio, 5 exemplos.
    forecast_odds gera no maximo 1 linha por galgo/corrida (back ou lay); value_counts pode ser desigual.
    Controla tamanho via env:
    - GREYHOUNDS_DEBUG_MAX_ROWS: limita linhas usadas nos prints (default 50000)
    """
    df = generate_signals(source="forecast", market="win", rule="forecast_odds", entry_type="both")
    print(f"Total de linhas (gerado): {len(df)}")
    if len(df) == 0:
        return

    max_rows = int(os.getenv("GREYHOUNDS_DEBUG_MAX_ROWS", "50000"))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Amostrando {max_rows} linhas para estatísticas/prints.")
    print("\ndf['entry_type'].value_counts():")
    print(df["entry_type"].value_counts())
    print("\ndf['value_ratio'].describe():")
    print(df["value_ratio"].describe())
    print("\nContagem NaN: forecast_odds={}, back_target_bsp={}".format(
        df["forecast_odds"].isna().sum(), df["back_target_bsp"].isna().sum()
    ))
    cols = [
        "track_name", "race_time_iso", "entry_type", "forecast_rank", "forecast_odds",
        "back_target_bsp", "lay_target_bsp", "value_ratio", "win_lose", "pnl_stake_ref",
    ]
    available = [c for c in cols if c in df.columns]
    print("\n5 linhas (colunas: {}):".format(available))
    print(df[available].head(5).to_string())


if __name__ == "__main__":
    # Debug leve sempre ok
    _debug_load_timeform_forecast_all()

    # Debug pesado (gera sinais) só se habilitar explicitamente:
    #   GREYHOUNDS_DEBUG_FORECAST_ODDS=1 python -m src.greyhounds.analysis.signals
    if os.getenv("GREYHOUNDS_DEBUG_FORECAST_ODDS", "").strip() in ("1", "true", "yes", "y"):
        print("\n--- forecast_odds ---")
        _debug_forecast_odds_small_sample()

