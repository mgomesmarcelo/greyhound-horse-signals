from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from dateutil import parser as date_parser
from loguru import logger

from src.greyhounds.config import RULE_LABELS, settings
from src.greyhounds.utils.text import clean_horse_name, normalize_track_name
from src.greyhounds.utils.files import write_dataframe_snapshots

_TRAP_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _strip_trap_prefix(name: str) -> str:
    return _TRAP_PREFIX_RE.sub("", name or "").strip()


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    return normalize_track_name(str(menu_hint or ""))


def _to_iso_yyyy_mm_dd_thh_mm(value: str) -> str:
    try:
        dt = date_parser.parse(value, dayfirst=True)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


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
        cleaned = clean_horse_name(candidate)
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
            df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        )
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
            df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        )
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
                )

    logger.info("Betfair PLACE index criado: {} corridas", len(index))
    return index


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
                clean_horse_name(str(row.get(col, "")))
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
    odd = (
        float(target_runner.bsp)
        if market == "place" and target_runner is not None
        else float(target_bsp_win or 0.0)
    )

    # Nova referência: 1 unidade (mantemos colunas _fixed_10 por compatibilidade)
    stake_ref = 1.0
    liability_ref = 1.0
    legacy_scale = 10.0  # TODO: descontinuar colunas *_fixed_10 após migração completa
    commission_rate = 0.065

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

    def _vol_for(name_raw: object) -> float:
        name = clean_horse_name(str(name_raw)) if isinstance(name_raw, str) else ""
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

    if source == "forecast":
        tf_rows = load_timeform_forecast_top3()
    elif source == "betfair_resultado":
        tf_rows = _build_betfair_direct_rows(bf_win_index)
    else:
        tf_rows = load_timeform_top3()

    signals_rows: List[dict] = []
    for row in tf_rows:
        results = _calc_signals_for_race(
            row,
            bf_win_index,
            bf_place_index,
            market=market,
            rule=rule,
            leader_share_min=leader_share_min,
        )
        for result in results:
            if entry_type in ("both", result.get("entry_type")):
                signals_rows.append(result)

    df = pd.DataFrame(signals_rows)
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
                "source",
                "market",
                "rule",
                "rule_label",
                "entry_type",
            ],
        )
    else:
        df_sorted = df.sort_values(
            ["date", "track_name", "race_time_iso", "entry_type"]
        ).reset_index(drop=True)

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
    "generate_signals",
    "load_betfair_place",
    "load_betfair_win",
    "load_timeform_forecast_top3",
    "load_timeform_top3",
    "write_signals_csv",
]

