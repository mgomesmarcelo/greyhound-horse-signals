from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import ast
import pandas as pd
from dateutil import parser as date_parser
from loguru import logger

from ..config import RULE_LABELS, settings
from ..utils.files import write_dataframe_snapshots
from ..utils.text import clean_horse_name, normalize_track_name


_TRAP_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _strip_trap_prefix(name: str) -> str:
    return _TRAP_PREFIX_RE.sub("", name or "").strip()


def _signals_snapshot_path(source: str, market: str, rule: str, provider: str) -> Path:
    """Path do parquet: processed/signals/<provider>/signals_<source>_<market>_<rule>.parquet."""
    filename = f"signals_{source}_{market}_{rule}.parquet"
    if provider == "timeform":
        return settings.PROCESSED_SIGNALS_TIMEFORM_DIR / filename
    return settings.PROCESSED_SIGNALS_SPORTINGLIFE_DIR / filename


def _signals_raw_path(source: str, market: str, rule: str, provider: str) -> Path:
    filename = f"signals_{source}_{market}_{rule}.csv"
    if provider == "timeform":
        return settings.RAW_SIGNALS_TIMEFORM_DIR / filename
    return settings.RAW_SIGNALS_SPORTINGLIFE_DIR / filename


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    # Normaliza prefixos regionais e separadores (ex.: "IRE / Punchestown 31st Dec" -> "Punchestown 31st Dec")
    text = (menu_hint or "").strip()
    text = re.sub(r"^\s*(?:UK|IRE|IRL|GB|UK\s*&\s*IRE)\s*/\s*", "", text, flags=re.IGNORECASE)
    # Troca separadores incomuns por espaco
    text = re.sub(r"[/|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Extrai apenas letras e espacos ate o primeiro digito (ex.: "Punchestown 31st Dec" -> "Punchestown")
    m = re.match(r"^([A-Za-z][A-Za-z\s]+?)(?:\s*\d|$)", text)
    base = m.group(1) if m else text
    return normalize_track_name(base)


def _to_iso_yyyy_mm_dd_thh_mm(value: str) -> str:
    # event_dt exemplo: "19-09-2025 20:01"
    try:
        dt = date_parser.parse(value, dayfirst=True)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _to_iso_series(values: pd.Series) -> pd.Series:
    """Converte uma Series de datas variadas para ISO (YYYY-MM-DDTHH:MM) de forma vetorizada.

    Usa pandas.to_datetime com dayfirst=True e errors='coerce' para robustez e performance,
    evitando chamadas por linha ao dateutil (que sao lentas e podem travar em strings ruins).
    """
    try:
        s = values.astype(str).str.strip()
        # Normaliza separador
        s = s.str.replace("/", "-", regex=False)
        # Tentativa 1: dd-mm-YYYY HH:MM (mais comum nos arquivos)
        dt = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
        mask = dt.isna()
        if mask.any():
            # Tentativa 2: YYYY-mm-dd HH:MM
            dt2 = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M", errors="coerce")
            dt.loc[mask] = dt2
            mask = dt.isna()
        if mask.any():
            # Fallback pontual usando dateutil apenas nos remanescentes
            def _safe_parse(v: str) -> pd.Timestamp | None:
                try:
                    return pd.Timestamp(date_parser.parse(v, dayfirst=True))
                except Exception:
                    return pd.NaT
            dt3 = s[mask].map(_safe_parse)
            dt.loc[mask] = pd.to_datetime(dt3, errors="coerce")
        out = dt.dt.strftime("%Y-%m-%dT%H:%M")
        return out.fillna("")
    except Exception:
        # Fallback final: funcao escalar
        return values.astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)


def _iter_result_paths(pattern: str) -> list[Path]:
    """Retorna arquivos de resultado: preferencialmente parquets em processed/Result, senao parquet/csv em Result."""
    parquet_in_processed = sorted(settings.PROCESSED_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_in_processed:
        return parquet_in_processed
    parquet_in_raw = sorted(settings.RAW_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_in_raw:
        return parquet_in_raw
    return sorted(settings.RAW_RESULT_DIR.glob(f"{pattern}.csv"))


def _parse_forecast_top3(text: str) -> List[str]:
    """Extrai apenas os 3 primeiros nomes previstos da string de forecast (Timeform ou Sporting Life).

    Suporta formatos como:
    - "TimeformForecast : 2.50 Nome A, 3.50 Nome B, 4.50 Nome C, ..."
    - "Forecast: 2.50 Nome A, 3.50 Nome B, ..."
    - "Nome A (2/1), Nome B (5/2), Nome C (4/1), ..."
    - "2.50 Nome A, 3.50 Nome B, 4.50 Nome C"
    """
    if not isinstance(text, str):
        return []
    s = re.sub(r"(?i)(?:timeformforecast|(?:betting\s+)?forecast)\s*:\s*", "", text.strip())
    parts = [p.strip() for p in s.split(",") if p and isinstance(p, str)]
    names: List[str] = []
    for p in parts:
        # remove odds no inicio (ex.: 2.50 Nome A)
        m1 = re.match(r"^\s*\d+(?:\.\d+)?\s+(.+)$", p)
        if m1:
            candidate = m1.group(1).strip()
        else:
            # remove odds entre parenteses no final (ex.: Nome A (2/1))
            candidate = re.sub(r"\s*\([^\)]*\)\s*$", "", p).strip()
        candidate = _strip_trap_prefix(candidate)
        cleaned = clean_horse_name(candidate)
        if cleaned and cleaned not in names:
            names.append(cleaned)
        if len(names) >= 3:
            break
    return names


_FORECAST_ITEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(.+)$", re.IGNORECASE)


def _parse_forecast_all(text: str) -> List[Dict[str, Any]]:
    """Parseia o texto completo do forecast (Timeform ou Sporting Life) e retorna lista de itens com odds e rank.

    Formato esperado: "TimeformForecast : 2.88 Horse A, 5.00 Horse B, ..." ou "Forecast: 2.50 Horse A, ..."
    Itens malformados sao ignorados.
    """
    if not isinstance(text, str):
        return []
    stripped = re.sub(r"(?i)(?:timeformforecast|(?:betting\s+)?forecast)\s*:\s*", "", text.strip())
    parts = [p.strip() for p in stripped.split(",") if isinstance(p, str) and p.strip()]
    items: List[Dict[str, Any]] = []
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
        name_clean = clean_horse_name(name_raw)
        if not name_clean:
            continue
        items.append({
            "forecast_rank": rank,
            "forecast_odds": odds_val,
            "forecast_name_clean": name_clean,
            "forecast_name_raw": name_raw,
        })
    return items


@dataclass
class RunnerBF:
    selection_name_raw: str
    selection_name_clean: str
    pptradedvol: float
    bsp: float
    win_lose: int


def load_betfair_win() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    """Carrega resultados WIN (UK/IRE): parquet em processed/Result preferencial, senao CSV em Result.

    Indexa por (track_key, race_iso). Compativel com padroes dwbfprices*win* (parquet ou csv).
    """
    all_files = _iter_result_paths("dwbfprices*win*")
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for result_path in all_files:
        try:
            if result_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(result_path)
            else:
                df = pd.read_csv(result_path, encoding=settings.CSV_ENCODING)
            # Normaliza cabecalhos para minusculas (compativel com arquivos antigos em CAIXA ALTA)
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception as e:
            logger.error("Falha ao ler {}: {}", result_path.name, e)
            continue

        # Garante colunas
        for col in ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]:
            if col not in df.columns:
                df[col] = ""

        # Limpeza e normalizacao
        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = _to_iso_series(df["event_dt"].astype(str))
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), grp in df.groupby(["track_key", "race_iso" ], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, r in grp.iterrows():
                name_clean = r["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                new_pp = float(r["pptradedvol"])
                new_bsp = float(r["bsp"]) if pd.notna(r["bsp"]) else float("nan")
                new_win = int(r["win_lose"])
                existing = runners.get(name_clean)
                if existing:
                    merged_pp = existing.pptradedvol + new_pp
                    merged_bsp = new_bsp if pd.notna(new_bsp) else existing.bsp
                    merged_win = 1 if existing.win_lose == 1 or new_win == 1 else existing.win_lose
                    runners[name_clean] = RunnerBF(
                        selection_name_raw=existing.selection_name_raw,
                        selection_name_clean=name_clean,
                        pptradedvol=merged_pp,
                        bsp=merged_bsp,
                        win_lose=merged_win,
                    )
                else:
                    runners[name_clean] = RunnerBF(
                        selection_name_raw=r["selection_name_raw"],
                        selection_name_clean=name_clean,
                        pptradedvol=new_pp,
                        bsp=new_bsp,
                        win_lose=new_win,
                    )

    logger.info("Betfair WIN index criado: {} corridas", len(index))
    return index


def load_betfair_place() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    """Carrega resultados PLACE (UK/IRE): parquet em processed/Result preferencial, senao CSV em Result.

    Indexa por (track_key, race_iso). Compativel com padroes dwbfprices*place* (parquet ou csv).
    """
    all_files = _iter_result_paths("dwbfprices*place*")
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for result_path in all_files:
        try:
            if result_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(result_path)
            else:
                df = pd.read_csv(result_path, encoding=settings.CSV_ENCODING)
            # Normaliza cabecalhos para minusculas (compativel com arquivos antigos em CAIXA ALTA)
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception as e:
            logger.error("Falha ao ler {}: {}", result_path.name, e)
            continue

        # Garante colunas
        for col in ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]:
            if col not in df.columns:
                df[col] = ""

        # Limpeza e normalizacao
        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = _to_iso_series(df["event_dt"].astype(str))
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), grp in df.groupby(["track_key", "race_iso" ], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, r in grp.iterrows():
                name_clean = r["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                new_pp = float(r["pptradedvol"])
                new_bsp = float(r["bsp"]) if pd.notna(r["bsp"]) else float("nan")
                new_win = int(r["win_lose"])
                existing = runners.get(name_clean)
                if existing:
                    merged_pp = existing.pptradedvol + new_pp
                    merged_bsp = new_bsp if pd.notna(new_bsp) else existing.bsp
                    merged_win = 1 if existing.win_lose == 1 or new_win == 1 else existing.win_lose
                    runners[name_clean] = RunnerBF(
                        selection_name_raw=existing.selection_name_raw,
                        selection_name_clean=name_clean,
                        pptradedvol=merged_pp,
                        bsp=merged_bsp,
                        win_lose=merged_win,
                    )
                else:
                    runners[name_clean] = RunnerBF(
                        selection_name_raw=r["selection_name_raw"],
                        selection_name_clean=name_clean,
                        pptradedvol=new_pp,
                        bsp=new_bsp,
                        win_lose=new_win,
                    )

    logger.info("Betfair PLACE index criado: {} corridas", len(index))
    return index

def load_timeform_top3() -> List[dict]:
    """Carrega timeform_top3 (parquet em processed preferencial, csv em timeform_top3 como fallback).

    Suporta dois esquemas:
    - Esquema antigo: colunas TimeformTop1/2/3
    - Esquema novo: colunas TimeformPrev_list (string representando lista) ou TimeformPrev (string com nomes separados por ';')
    """
    processed_dir = settings.PROCESSED_TIMEFORM_TOP3_DIR
    raw_dir = settings.RAW_TIMEFORM_TOP3_DIR
    parquet_paths = sorted(processed_dir.glob("timeform_top3_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("timeform_top3_*.csv"))
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
        except Exception as e:
            logger.error("Falha ao ler {}: {}", path.name, e)
            continue

        # Normaliza colunas basicas
        for col in ["track_name", "race_time_iso"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(r.get("race_time_iso", "")))
            names: List[str] = []
            if {"TimeformTop1", "TimeformTop2", "TimeformTop3"}.issubset(set(df.columns)):
                names = [clean_horse_name(str(r.get(c, ""))) for c in ["TimeformTop1", "TimeformTop2", "TimeformTop3"]]
            else:
                # Tenta esquema novo
                raw_list = r.get("TimeformPrev_list")
                raw_text = r.get("TimeformPrev")
                parsed: List[str] = []
                if isinstance(raw_list, str) and raw_list.strip().startswith("["):
                    try:
                        tmp = ast.literal_eval(raw_list)
                        if isinstance(tmp, list):
                            parsed = [str(x) for x in tmp]
                    except Exception:
                        parsed = []
                if not parsed and isinstance(raw_text, str):
                    # separa por ';'
                    parsed = [p.strip() for p in raw_text.split(";") if isinstance(p, str) and p.strip()]
                names = [clean_horse_name(x) for x in parsed if isinstance(x, str) and x]
                # garante tamanho 3
                while len(names) < 3:
                    names.append("")
                names = names[:3]
            if not track or not race_iso or not any(names):
                continue
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": {
                    "track_name": track,
                    "race_time_iso": race_iso,
                    "TimeformTop1": names[0] if len(names) > 0 else "",
                    "TimeformTop2": names[1] if len(names) > 1 else "",
                    "TimeformTop3": names[2] if len(names) > 2 else "",
                },
            })
    logger.info("Timeform Top3 carregado: {} corridas", len(rows))
    return rows


def load_timeform_forecast_top3() -> List[dict]:
    """Carrega TimeformForecast (parquet em processed preferencial, csv em TimeformForecast como fallback).

    Retorna linhas com os 3 primeiros previstos; mesmo formato de saida de load_timeform_top3.
    """
    processed_dir = settings.PROCESSED_TIMEFORM_FORECAST_DIR
    raw_dir = settings.RAW_TIMEFORM_FORECAST_DIR
    parquet_paths = sorted(processed_dir.glob("TimeformForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("TimeformForecast_*.csv"))
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
        except Exception as e:
            logger.error("Falha ao ler {}: {}", path.name, e)
            continue

        for col in ["track_name", "race_time_iso", "TimeformForecast"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(r.get("race_time_iso", "")))
            names = _parse_forecast_top3(str(r.get("TimeformForecast", "")))
            if not track or not race_iso or not names:
                continue
            # garante 3 posicoes
            while len(names) < 3:
                names.append("")
            raw_like = {
                "track_name": track,
                "race_time_iso": race_iso,
                "TimeformTop1": names[0],
                "TimeformTop2": names[1],
                "TimeformTop3": names[2],
            }
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": raw_like,
            })

    logger.info("Timeform Forecast(Top3) carregado: {} corridas", len(rows))
    return rows


def load_timeform_forecast_all() -> pd.DataFrame:
    """Carrega TimeformForecast (parquet em processed preferencial, csv em TimeformForecast como fallback).

    Retorna DataFrame com colunas track_key, race_iso, forecast_items (lista de dicts:
    forecast_rank, forecast_odds, forecast_name_clean, forecast_name_raw).
    """
    processed_dir = settings.PROCESSED_TIMEFORM_FORECAST_DIR
    raw_dir = settings.RAW_TIMEFORM_FORECAST_DIR
    parquet_paths = sorted(processed_dir.glob("TimeformForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("TimeformForecast_*.csv"))
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
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(row.get("race_time_iso", "")).strip())
            if not track or not race_iso:
                continue
            forecast_items = _parse_forecast_all(str(row.get("TimeformForecast", "")))
            if not forecast_items:
                continue
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
    out: List[dict] = []
    for _, row in df.iterrows():
        track_key = row.get("track_key", "")
        race_iso = row.get("race_iso", "")
        forecast_items = row.get("forecast_items") or []
        raw = {"track_name": track_key, "race_time_iso": race_iso}
        out.append({
            "track_key": track_key,
            "race_iso": race_iso,
            "forecast_items": forecast_items,
            "raw": raw,
        })
    return out


def _signal_race_selection_key(row: dict) -> Tuple[str, str, str]:
    """Chave (race_id, selection_id): race_time_iso, track_name, nome do cavalo."""
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
    """No maximo um sinal por (race_id, selection_id). Duplicatas: mantem o primeiro."""
    if not result:
        return result
    seen: Dict[Tuple[str, str, str], bool] = {}
    unique: List[dict] = []
    for row in result:
        key = _signal_race_selection_key(row)
        if key in seen:
            logger.warning(
                "forecast_odds: duplicata ignorada: race_iso={!r} track={!r} selection={!r} entry_type={!r}",
                race_iso, track_key, key[2], row.get("entry_type"),
            )
            continue
        seen[key] = True
        unique.append(row)
    return unique


def _assert_forecast_odds_unique_race_selection(df: pd.DataFrame) -> None:
    """Valida que forecast_odds nao tem duas linhas para o mesmo (race_id, selection_id)."""
    if df.empty or "race_time_iso" not in df.columns:
        return
    from collections import Counter
    keys: List[Tuple[str, str, str]] = []
    for _, row in df.iterrows():
        keys.append(_signal_race_selection_key(row.to_dict()))
    if len(keys) != len(set(keys)):
        counts = Counter(keys)
        dupes = [k for k, c in counts.items() if c > 1]
        logger.error(
            "forecast_odds: violacao de unicidade (race_id, selection_id): {} duplicatas; exemplos: {}",
            len(dupes), dupes[:5],
        )


def _calc_signals_forecast_odds_for_race(
    tf_row: dict,
    bf_win_index: Dict[Tuple[str, str], Dict[str, RunnerBF]],
    bf_place_index: Dict[Tuple[str, str], Dict[str, RunnerBF]] | None,
    market: str,
    rule: str = "forecast_odds",
) -> List[dict]:
    """Gera sinais por runner do forecast que casou com Betfair: no maximo 1 linha por cavalo/corrida.
    value_ratio = back_target_bsp / forecast_odds. value_ratio >= BACK_MIN -> back;
    value_ratio <= LAY_MAX -> lay; senao zona morta.
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
    stake_fix10 = 1.0
    commission_rate = 0.065

    tf_top1 = forecast_items[0].get("forecast_name_clean") or "" if len(forecast_items) >= 1 else ""
    tf_top2 = forecast_items[1].get("forecast_name_clean") or "" if len(forecast_items) >= 2 else ""
    tf_top3 = forecast_items[2].get("forecast_name_clean") or "" if len(forecast_items) >= 3 else ""

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
        "total_matched_volume": 0.0,
        "market": market,
        "rule": rule,
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

        value_ratio = odd / forecast_odds_val if forecast_odds_val > 0 else float("nan")

        back_is_green = target_win_lose == 1
        if back_is_green:
            back_pnl_stake10 = stake_fix10 * max(0.0, odd - 1.0) * (1.0 - commission_rate)
        else:
            back_pnl_stake10 = -stake_fix10

        liability_from_stake10 = stake_fix10 * max(0.0, odd - 1.0)
        stake_from_liab10 = 1.0 / max(0.001, odd - 1.0)
        if target_win_lose == 1:
            lay_pnl_stake10 = -liability_from_stake10
            lay_pnl_liab10 = -1.0
            lay_is_green = False
        else:
            lay_pnl_stake10 = stake_fix10 * (1.0 - commission_rate)
            lay_pnl_liab10 = stake_from_liab10 * (1.0 - commission_rate)
            lay_is_green = True

        extra = {
            "forecast_rank": forecast_rank,
            "forecast_odds": forecast_odds_val,
            "value_ratio": value_ratio,
        }

        out_back = {
            **base_neutral,
            **extra,
            "entry_type": "back",
            "back_target_name": name_clean,
            "back_target_bsp": round(odd, 2),
            "lay_target_name": "",
            "lay_target_bsp": float("nan"),
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
        }

        out_lay = {
            **base_neutral,
            **extra,
            "entry_type": "lay",
            "back_target_name": "",
            "back_target_bsp": float("nan"),
            "lay_target_name": name_clean,
            "lay_target_bsp": round(odd, 2),
            "stake_fixed_10": round(stake_fix10, 2),
            "liability_from_stake_fixed_10": round(liability_from_stake10, 2),
            "stake_for_liability_10": round(stake_from_liab10, 2),
            "liability_fixed_10": 1.0,
            "win_lose": target_win_lose,
            "is_green": lay_is_green,
            "pnl_stake_fixed_10": round(lay_pnl_stake10, 2),
            "pnl_liability_fixed_10": round(lay_pnl_liab10, 2),
            "roi_row_stake_fixed_10": round(
                lay_pnl_stake10 / stake_fix10 if stake_fix10 > 0 else 0.0, 4,
            ),
            "roi_row_liability_fixed_10": round(
                lay_pnl_liab10 / 1.0 if 1.0 > 0 else 0.0, 4,
            ),
        }

        if pd.isna(value_ratio):
            continue
        if value_ratio >= settings.FORECAST_ODDS_BACK_MIN_VALUE_RATIO:
            result.append(out_back)
        elif value_ratio <= settings.FORECAST_ODDS_LAY_MAX_VALUE_RATIO:
            result.append(out_lay)

    result = _dedupe_forecast_odds_signals_by_race_selection(result, track_key, race_iso)
    return result


def load_sportinglife_top3() -> List[dict]:
    """Carrega sportinglife_top3 (parquet em processed preferencial, csv em sportinglife_top3 como fallback).

    Colunas esperadas: track_name, race_time_iso, TimeformTop1/2/3.
    """
    processed_dir = settings.PROCESSED_SPORTINGLIFE_TOP3_DIR
    raw_dir = settings.RAW_SPORTINGLIFE_TOP3_DIR
    parquet_paths = sorted(processed_dir.glob("sportinglife_top3_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("sportinglife_top3_*.csv"))
        use_parquet = False

    rows: List[dict] = []
    for path in sources:
        try:
            if not use_parquet:
                try:
                    if path.stat().st_size == 0:
                        logger.warning("Arquivo vazio ignorado: {}", path.name)
                        continue
                except Exception:
                    pass
            if use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
            if df is None or (hasattr(df, "empty") and df.empty and len(df.columns) == 0):
                logger.warning("Sem colunas/linhas em {}  ignorado", path.name)
                continue
        except Exception as e:
            if "No columns to parse from file" in str(e):
                logger.warning("Sem colunas em {}  ignorado", path.name)
                continue
            logger.error("Falha ao ler {}: {}", path.name, e)
            continue

        for col in ["track_name", "race_time_iso", "TimeformTop1", "TimeformTop2", "TimeformTop3"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(r.get("race_time_iso", "")))
            names = [clean_horse_name(str(r.get(c, ""))) for c in ["TimeformTop1", "TimeformTop2", "TimeformTop3"]]
            if not track or not race_iso or not any(names):
                continue
            while len(names) < 3:
                names.append("")
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": {
                    "track_name": track,
                    "race_time_iso": race_iso,
                    "TimeformTop1": names[0],
                    "TimeformTop2": names[1],
                    "TimeformTop3": names[2],
                },
            })
    logger.info("Sporting Life Top3 carregado: {} corridas", len(rows))
    return rows


def load_sportinglife_forecast_top3() -> List[dict]:
    """Carrega SportingLifeForecast (parquet em processed preferencial, csv em SportingLifeForecast como fallback).

    Retorna os 3 primeiros do Forecast; mesmo formato de saida dos loaders de Top3.
    """
    processed_dir = settings.PROCESSED_SPORTINGLIFE_FORECAST_DIR
    raw_dir = settings.RAW_SPORTINGLIFE_FORECAST_DIR
    parquet_paths = sorted(processed_dir.glob("SportingLifeForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("SportingLifeForecast_*.csv"))
        use_parquet = False

    rows: List[dict] = []
    for path in sources:
        try:
            if not use_parquet:
                try:
                    if path.stat().st_size == 0:
                        logger.warning("Arquivo vazio ignorado: {}", path.name)
                        continue
                except Exception:
                    pass
            if use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    encoding=settings.CSV_ENCODING,
                    engine="python",
                    on_bad_lines="skip",
                )
            if df is None or (hasattr(df, "empty") and df.empty and len(df.columns) == 0):
                logger.warning("Sem colunas/linhas em {}  ignorado", path.name)
                continue
        except Exception as e:
            if "No columns to parse from file" in str(e):
                logger.warning("Sem colunas em {}  ignorado", path.name)
                continue
            logger.error("Falha ao ler {}: {}", path.name, e)
            continue

        forecast_col = "SportingLifeForecast" if "SportingLifeForecast" in df.columns else "TimeformForecast"
        for col in ["track_name", "race_time_iso", forecast_col]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(r.get("race_time_iso", "")))
            names = _parse_forecast_top3(str(r.get(forecast_col, "")))
            if not track or not race_iso or not names:
                continue
            while len(names) < 3:
                names.append("")
            raw_like = {
                "track_name": track,
                "race_time_iso": race_iso,
                "TimeformTop1": names[0],
                "TimeformTop2": names[1],
                "TimeformTop3": names[2],
            }
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": raw_like,
            })
    logger.info("Sporting Life Forecast(Top3) carregado: {} corridas", len(rows))
    return rows


def load_sportinglife_forecast_all() -> pd.DataFrame:
    """Carrega SportingLifeForecast (parquet em processed preferencial, csv em SportingLifeForecast como fallback).

    Retorna DataFrame com colunas track_key, race_iso, forecast_items (lista de dicts:
    forecast_rank, forecast_odds, forecast_name_clean, forecast_name_raw).
    """
    processed_dir = settings.PROCESSED_SPORTINGLIFE_FORECAST_DIR
    raw_dir = settings.RAW_SPORTINGLIFE_FORECAST_DIR
    parquet_paths = sorted(processed_dir.glob("SportingLifeForecast_*.parquet"))
    if parquet_paths:
        sources = parquet_paths
        use_parquet = True
    else:
        sources = sorted(raw_dir.glob("SportingLifeForecast_*.csv"))
        use_parquet = False

    rows: List[dict] = []
    forecast_col = "SportingLifeForecast"
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

        if forecast_col not in df.columns:
            forecast_col_use = "TimeformForecast" if "TimeformForecast" in df.columns else forecast_col
        else:
            forecast_col_use = forecast_col
        for col in ["track_name", "race_time_iso", forecast_col_use]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, row in df.iterrows():
            track = normalize_track_name(str(row.get("track_name", "")))
            race_iso = _to_iso_yyyy_mm_dd_thh_mm(str(row.get("race_time_iso", "")))
            if not track or not race_iso:
                continue
            forecast_items = _parse_forecast_all(str(row.get(forecast_col_use, "")))
            if not forecast_items:
                continue
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "forecast_items": forecast_items,
            })

    logger.info("Sporting Life Forecast (all) carregado: {} corridas", len(rows))
    return pd.DataFrame(rows)


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
    top_names = [n for n in tf_row["top_names"] if isinstance(n, str) and n]
    # Selecao por volume sempre no mercado WIN
    group = bf_win_index.get((track_key, race_iso))
    if not group:
        return []

    num_runners = len(group)
    total_vol_race = 0.0
    for runner in group.values():
        try:
            vol = float(runner.pptradedvol)
        except (TypeError, ValueError):
            vol = 0.0
        if vol > 0:
            total_vol_race += vol

    # Coleta volumes e BSP para os tres de referencia
    triples: List[Tuple[str, float, float]] = []  # (name_clean, vol, bsp)
    for name in top_names:
        r = group.get(name)
        if not r or pd.isna(r.bsp):
            return []
        triples.append((name, max(0.0, float(r.pptradedvol)), float(r.bsp)))

    if len(triples) < 3:
        return []

    # Ordena por volume desc entre os Top3 de referencia
    triples_sorted = sorted(triples, key=lambda t: t[1], reverse=True)
    first, second, third = triples_sorted[0], triples_sorted[1], triples_sorted[2]

    # Metricas auxiliares entre 2o e 3o
    vol2, vol3 = second[1], third[1]
    pct_diff = (vol2 - vol3) / vol2 if vol2 > 0 else float("inf")
    ratio = (vol2 / vol3) if vol3 > 0 else float("inf")
    leader_share = (first[1] / total_vol_race) if total_vol_race > 0 else 0.0

    target_name_clean: str | None = None
    target_bsp_win: float | None = None
    if rule == "terceiro_queda50":
        if vol3 <= 0 or pct_diff <= 0.5:
            return []
        target_name_clean = third[0]
        target_bsp_win = third[2]
    else:
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
    odd = float(target_runner.bsp) if market == "place" else float(target_bsp_win or 0.0)

    stake_fix10 = 1.0
    commission_rate = 0.065

    back_is_green = target_win_lose == 1
    if back_is_green:
        back_profit_gross = stake_fix10 * max(0.0, odd - 1.0)
        back_pnl_stake10 = back_profit_gross * (1.0 - commission_rate)
    else:
        back_pnl_stake10 = -stake_fix10

    liability_from_stake10 = stake_fix10 * max(0.0, odd - 1.0)
    liability_fix10 = 1.0
    stake_from_liab10 = liability_fix10 / max(0.001, odd - 1.0)

    if target_win_lose == 1:
        lay_pnl_stake10 = -liability_from_stake10
        lay_pnl_liab10 = -liability_fix10
        lay_is_green = False
    else:
        lay_pnl_stake10 = stake_fix10 * (1.0 - commission_rate)
        lay_pnl_liab10 = stake_from_liab10 * (1.0 - commission_rate)
        lay_is_green = True

    raw = tf_row["raw"]

    def _vol_for(name_raw: object) -> float:
        name = clean_horse_name(str(name_raw)) if isinstance(name_raw, str) else ""
        return next((v for runner_name, v, _ in triples if runner_name == name), 0.0)

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
        "num_runners": int(num_runners),
        "total_matched_volume": round(total_vol_race, 2),
        "leader_name_by_volume": first[0],
        "leader_volume_share_pct": round(leader_share * 100.0, 2),
        "market": market,
        "rule": rule,
    }

    out_back = {
        **base,
        "entry_type": "back",
        "back_target_name": target_name_clean,
        "back_target_bsp": round(odd, 2),
        "lay_target_name": "",
        "lay_target_bsp": float("nan"),
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
    }

    out_lay = {
        **base,
        "entry_type": "lay",
        "back_target_name": "",
        "back_target_bsp": float("nan"),
        "lay_target_name": target_name_clean,
        "lay_target_bsp": round(odd, 2),
        "stake_fixed_10": round(stake_fix10, 2),
        "liability_from_stake_fixed_10": round(liability_from_stake10, 2),
        "stake_for_liability_10": round(stake_from_liab10, 2),
        "liability_fixed_10": round(liability_fix10, 2),
        "win_lose": target_win_lose,
        "is_green": lay_is_green,
        "pnl_stake_fixed_10": round(lay_pnl_stake10, 2),
        "pnl_liability_fixed_10": round(lay_pnl_liab10, 2),
        "roi_row_stake_fixed_10": round(
            lay_pnl_stake10 / liability_from_stake10 if liability_from_stake10 > 0 else 0.0,
            4,
        ),
        "roi_row_liability_fixed_10": round(
            lay_pnl_liab10 / liability_fix10 if liability_fix10 > 0 else 0.0,
            4,
        ),
    }

    return [out_back, out_lay]


def generate_signals(
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
    entry_type: str = "both",
    provider: str = "timeform",
    leader_share_min: float = 0.5,
    strategy: str | None = None,
) -> pd.DataFrame:
    if strategy:
        entry_type = strategy
    bf_win_index = load_betfair_win()
    bf_place_index = load_betfair_place() if market == "place" else None

    use_forecast_odds = (
        provider in ("timeform", "sportinglife")
        and source == "forecast"
        and rule == "forecast_odds"
    )
    total_forecast_races = 0
    total_forecast_runners = 0
    if use_forecast_odds:
        if provider == "sportinglife":
            df_forecast = load_sportinglife_forecast_all()
        else:
            df_forecast = load_timeform_forecast_all()
        tf_rows = _forecast_all_df_to_tf_rows(df_forecast)
        total_forecast_races = len(tf_rows)
        total_forecast_runners = sum(len(r.get("forecast_items") or []) for r in tf_rows)
    elif provider == "sportinglife":
        if source == "forecast":
            tf_rows = load_sportinglife_forecast_top3()
        else:
            tf_rows = load_sportinglife_top3()
    else:
        if source == "forecast":
            tf_rows = load_timeform_forecast_top3()
        else:
            tf_rows = load_timeform_top3()

    signals: List[dict] = []
    for row in tf_rows:
        if use_forecast_odds:
            results = _calc_signals_forecast_odds_for_race(
                row, bf_win_index, bf_place_index, market=market, rule=rule
            )
        else:
            results = _calc_signals_for_race(
                row,
                bf_win_index,
                bf_place_index,
                market=market,
                rule=rule,
                leader_share_min=leader_share_min,
            )
        for res in results:
            if entry_type == "both" or res.get("entry_type") == entry_type:
                signals.append(res)

    df = pd.DataFrame(signals)
    if use_forecast_odds and total_forecast_races > 0:
        match_rate = (len(signals) / total_forecast_runners * 100.0) if total_forecast_runners > 0 else 0.0
        logger.info(
            "Forecast odds: corridas carregadas={}, runners no forecast={}, sinais gerados={}, taxa match forecast vs betfair={:.1f}%",
            total_forecast_races, total_forecast_runners, len(df), match_rate,
        )
    logger.info(
        "Sinais encontrados (provider={}, source={}, market={}, rule={}, entry_type={}, leader_share_min={}): {}",
        provider,
        source,
        market,
        rule,
        entry_type,
        leader_share_min,
        len(df),
    )
    if use_forecast_odds and not df.empty:
        _assert_forecast_odds_unique_race_selection(df)
    try:
        parquet_path = _signals_snapshot_path(
            source=source,
            market=market,
            rule=rule,
            provider=provider,
        )
        _ensure_dir(parquet_path.parent)
        df.to_parquet(parquet_path, index=False)
        logger.info("Parquet de sinais salvo: {} ({} linhas)", parquet_path, len(df))
    except Exception as e:
        logger.error("Falha ao salvar Parquet de sinais para horses: {}", e)
    return df


def write_signals_csv(
    df: pd.DataFrame,
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
    provider: str = "timeform",
) -> Path:
    out_path = _signals_raw_path(source=source, market=market, rule=rule, provider=provider)
    _ensure_dir(out_path.parent)
    df = df.copy()
    df["source"] = source
    df["market"] = market
    df["rule"] = rule
    df["rule_label"] = RULE_LABELS.get(rule, rule)
    df["provider"] = provider
    for col in ["num_runners", "total_matched_volume"]:
        if col not in df.columns:
            df[col] = pd.NA
    if df.empty:
        # cria CSV vazio com cabecalhos padrao
        df_sorted = pd.DataFrame([], columns=[
            "date","track_name","race_time_iso",
            "tf_top1","tf_top2","tf_top3",
            "vol_top1","vol_top2","vol_top3",
            "second_name_by_volume","third_name_by_volume",
            "ratio_second_over_third","pct_diff_second_vs_third",
            "num_runners","total_matched_volume",
            "lay_target_name","lay_target_bsp",
            "back_target_name","back_target_bsp",
            "leader_name_by_volume","leader_volume_share_pct",
            "stake_fixed_10","liability_from_stake_fixed_10",
            "stake_for_liability_10","liability_fixed_10",
            "win_lose","is_green","pnl_stake_fixed_10","pnl_liability_fixed_10",
            "roi_row_stake_fixed_10","roi_row_liability_fixed_10",
            "source","market","rule","rule_label","provider","entry_type",
        ])
    else:
        df_sorted = df.sort_values(["date", "track_name", "race_time_iso", "entry_type"]).reset_index(drop=True)
    parquet_path = _signals_snapshot_path(
        source=source,
        market=market,
        rule=rule,
        provider=provider,
    )
    try:
        write_dataframe_snapshots(df_sorted, raw_path=out_path, parquet_path=parquet_path)
        logger.info("Sinais gravados: CSV {} e Parquet {} ({} linhas)", out_path, parquet_path, len(df_sorted))
    except Exception as e:
        logger.error("Falha ao gravar sinais (CSV/Parquet) para horses: {}", e)
    return out_path


