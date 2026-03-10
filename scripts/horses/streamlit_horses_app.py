import sys
import io
import json
import hashlib
import math
import pickle
import datetime
import calendar
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
import re
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.horses.config import settings
from src.horses.utils.text import normalize_track_name
from src.horses.utils.strategy_name import format_strategy_name
from src.horses.analysis.signals import (
    _extract_track_from_menu_hint,
    _signals_raw_path,
    _signals_snapshot_path,
    _to_iso_series,
)

# Flag para validacao manual do filtro de datas (mostra caption com date_mode e range efetivo).
DEBUG_DATES = False

# --- PASSO 1 (diff guiado): Blocos que existem em greyhounds e foram portados ou nao se aplicam a horses ---
# GREYHOUNDS tem / HORSES:
# - init/sanitize session_state (consolidated_strategies, MF, pending_import, etc.): horses nao tem modo consolidado/MF; portado apenas init de filtros (weekdays_ms, hour_bucket_ms) e CORE_STATE_KEYS.
# - CORE_STATE_KEYS, RULE_EXTRA_KEYS: portado para horses (sem trap_ms; horses usa num_runners_bucket_ms; RULE_EXTRA_KEYS com leader_min para lider_volume_total).
# - Indices cacheados (category_index, num_runners_index): horses ja tinha; greyhounds usa drop_duplicates e mask no category_index (portado otimizacao opcional no cached).
# - Filtros: date range (horses ja tinha), tracks (ja tinha), categories/subcategories (ja tinha), num_runners (ja tinha como bucket), BSP (ja tinha), min_total_volume (ja tinha), leader_min (ja tinha). FALTANDO: weekdays_ms, hour_bucket_ms -> portado.
# - Filtro trap: greyhounds tem trap_number; horses nao tem trap -> nao portado (omitido).
# - forecast_rank, value_ratio, only_value_bets: regra forecast_odds em greyhounds; horses nao tem essa regra -> nao portado.
# - Graficos/metricas: horses ja tem ROI/PnL, drawdown, por pista, mensal, evolucao; portado _compute_hour_bucket_series e filtro por hora para alinhar UX.
# - load_signals_enriched: horses ja tinha; portado enriquecimento vetorizado (keys_series.map) em vez de apply.
# - Provider: horses ja passa provider (timeform/sportinglife) em paths e selectbox.
#
# Export/Import estrategia (CSV): colunas exportadas (exemplo):
# strategy_name,created_at,rule_select_label,source_select_label,provider,market,entry_type,
# date_start_input,date_end_input,tracks_ms,cats_ms,subcats_ms,weekdays_ms,hour_bucket_ms,
# num_runners_bucket_ms,bsp_low,bsp_high,min_total_volume,leader_min,forecast_rank_ms,
# value_ratio_min,value_ratio_max,only_value_bets

# Regras disponiveis para cavalos
HORSE_RULE_LABELS: dict[str, str] = {
    "terceiro_queda50": "Regra 1 – 3º com queda ≥ 50% vs 2º",
    "lider_volume_total": "Regra 2 – Líder com volume dominante",
    "forecast_odds": "Forecast Odds (Timeform)",
}
HORSE_RULE_LABELS_INV = {v: k for k, v in HORSE_RULE_LABELS.items()}

# Tipo de entrada (Back/Lay)
ENTRY_TYPE_LABELS: dict[str, str] = {
    "back": "Back",
    "lay": "Lay",
}
ENTRY_TYPE_LABELS_INV = {v: k for k, v in ENTRY_TYPE_LABELS.items()}

_RACE_TIME_ISO_FORMAT = "%Y-%m-%dT%H:%M"

CORE_STATE_KEYS = [
    "tracks_ms", "cats_ms", "subcats_ms",
    "weekdays_ms", "hour_bucket_ms",
    "num_runners_bucket_ms",
    "bsp_low", "bsp_high",
    "min_total_volume",
]
RULE_EXTRA_KEYS: dict[str, List[str]] = {
    "lider_volume_total": ["leader_min"],
    "forecast_odds": ["forecast_rank_ms", "value_ratio_min", "value_ratio_max", "only_value_bets"],
}

DATE_KEYS_FOR_STRATEGY = ("date_start_input", "date_end_input")


def _rule_label_to_slug(rule_label: str) -> str:
    """Converte label da regra para slug (ex.: Forecast Odds (Timeform) -> forecast_odds)."""
    if not rule_label:
        return ""
    slug = HORSE_RULE_LABELS_INV.get(rule_label, rule_label)
    if slug == rule_label and "Forecast" in str(rule_label):
        return "forecast_odds"
    return slug


def get_current_strategy_snapshot(
    rule_select_label: str,
    source_select_label: str,
    market: str,
    entry_type: str,
    provider: str,
    pnl: Optional[float] = None,
    roi: Optional[float] = None,
) -> dict:
    """
    Captura os parametros atuais do app em um dict serializavel (core + extras da regra ativa).
    PNL/ROI nao sao incluidos no CSV.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rule_slug = _rule_label_to_slug(rule_select_label)
    source_slug = "top3" if (source_select_label or "").strip() == "Top 3" else "forecast"
    tracks_ms = st.session_state.get("tracks_ms")
    cats_ms = st.session_state.get("cats_ms")
    bsp_low = st.session_state.get("bsp_low")
    bsp_high = st.session_state.get("bsp_high")
    strategy_name = format_strategy_name(
        rule_slug, source_slug, market, entry_type,
        tracks_ms=tracks_ms, cats_ms=cats_ms,
        bsp_low=bsp_low, bsp_high=bsp_high,
    )
    out: dict[str, Any] = {
        "strategy_name": strategy_name,
        "created_at": ts,
        "rule_select_label": rule_select_label,
        "source_select_label": source_select_label,
        "provider": provider,
        "market": market,
        "entry_type": entry_type,
    }
    filter_keys = list(CORE_STATE_KEYS) + list(RULE_EXTRA_KEYS.get(rule_slug, []))
    if rule_slug == "forecast_odds":
        filter_keys = [k for k in filter_keys if k != "min_total_volume"]
    for key in list(DATE_KEYS_FOR_STRATEGY) + filter_keys:
        if key not in st.session_state:
            continue
        val = st.session_state[key]
        if isinstance(val, (datetime.date, datetime.datetime)):
            out[key] = val.isoformat() if hasattr(val, "isoformat") else str(val)
        elif isinstance(val, (tuple, list)) and val and isinstance(val[0], (datetime.date, datetime.datetime)):
            out[key] = json.dumps([v.isoformat() if hasattr(v, "isoformat") else str(v) for v in val])
        elif isinstance(val, (list, tuple)):
            out[key] = json.dumps(list(val))
        elif isinstance(val, bool) and key == "only_value_bets":
            out[key] = val
        else:
            out[key] = val
    return out


def strategy_to_csv_bytes(strategy_dict: dict) -> bytes:
    """Exporta estrategia para CSV com header e 1 linha; listas em JSON, datas em ISO."""
    row: dict[str, Any] = {}
    for k, v in strategy_dict.items():
        if isinstance(v, (list, tuple)):
            row[k] = json.dumps(v)
        elif hasattr(v, "isoformat"):
            row[k] = v.isoformat()
        else:
            row[k] = v
    df = pd.DataFrame([row])
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def _is_likely_statement_csv(raw: bytes) -> bool:
    """Guard: se parecer statement (Back, Market P/L etc.), nao tratar como estrategia."""
    if not raw or len(raw) < 50:
        return False
    try:
        first = raw[:200].decode("utf-8", errors="ignore")
    except Exception:
        return False
    return "Market P/L" in first or (first.strip().startswith("Back") and "stake" in first.lower())


def parse_strategies_csv(uploaded_file: Any) -> List[dict]:
    """
    Le CSV e devolve lista de dicts; colunas que parecem JSON sao parseadas.
    Exemplo de CSV exportado (colunas): strategy_name,created_at,rule_select_label,source_select_label,
    provider,market,entry_type,date_start_input,date_end_input,tracks_ms,cats_ms,subcats_ms,weekdays_ms,
    hour_bucket_ms,num_runners_bucket_ms,bsp_low,bsp_high,min_total_volume,leader_min,forecast_rank_ms,
    value_ratio_min,value_ratio_max,only_value_bets
    """
    if uploaded_file is None:
        return []
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else b""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    if _is_likely_statement_csv(raw):
        return []
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except Exception:
            return []
    if df.empty:
        return []
    first_row = df.columns.tolist() if len(df.columns) > 0 else []
    strategy_like = any(
        str(c).strip() in ("rule_select_label", "source_select_label", "market", "entry_type", "strategy_name", "provider")
        for c in first_row
    )
    if not strategy_like and len(first_row) >= 1:
        first_cell = str(first_row[0]).strip() if first_row else ""
        if first_cell in ("Back", "Market P/L"):
            return []
    out: List[dict] = []
    for _, row in df.iterrows():
        d: dict[str, Any] = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip().startswith(("[", "{")):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            d[col] = val
        if "only_value_bets" in d and isinstance(d["only_value_bets"], str):
            d["only_value_bets"] = d["only_value_bets"].strip().lower() in ("true", "1", "yes")
        if isinstance(d.get("strategy_name"), str):
            s = d["strategy_name"].strip()
            s2 = re.split(r"\s*[\u2022*]\s*PNL:\s*", s, maxsplit=1)[0]
            d["strategy_name"] = s2.rstrip(" \u2022*").strip()
        out.append(d)
    return out


def _apply_strategy_to_state(strategy_dict: dict) -> None:
    """Aplica ao session_state apenas chaves conhecidas; chamar ANTES de qualquer widget."""
    imported_rule_label = strategy_dict.get("rule_select_label", "")
    imported_rule_slug = _rule_label_to_slug(imported_rule_label)
    if imported_rule_slug == "forecast_odds":
        strategy_dict = dict(strategy_dict)
        strategy_dict["provider"] = "timeform"
        strategy_dict["source_select_label"] = "Forecast"
    allowed = (
        {"rule_select_label", "source_select_label", "provider", "market", "entry_type"}
        | set(CORE_STATE_KEYS)
        | set(RULE_EXTRA_KEYS.get(imported_rule_slug, []))
        | set(DATE_KEYS_FOR_STRATEGY)
    )
    for key, value in strategy_dict.items():
        if key not in allowed:
            continue
        if key == "only_value_bets" and isinstance(value, str):
            value = value.strip().lower() in ("true", "1", "yes")
        if key in DATE_KEYS_FOR_STRATEGY and isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                value = parsed.date() if hasattr(parsed, "date") else value
        if key == "rule_select_label":
            st.session_state["horse_rule_select_label"] = value
        elif key == "source_select_label":
            st.session_state["horse_source_select_label"] = value
        elif key == "provider":
            st.session_state["provider"] = value
        else:
            st.session_state[key] = value


def _hash_uploaded_file(uploaded_file: Any) -> str:
    """Retorna hash SHA256 em hex do conteudo do arquivo."""
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return hashlib.sha256(data).hexdigest()


# MarketFeeder statement CSV (sem header): type, env, placed_dt, sport, event_path, selection, stake, price, profit, balance, trigger
_MF_STATEMENT_COLS = [
    "type", "env", "placed_dt", "sport", "event_path", "selection", "stake", "price", "profit", "balance", "trigger"
]


def _parse_event_path_mf(event_path: str) -> dict:
    """Extrai provider, track_raw, off_time, grade_distance de event_path (formato Greyhound Racing / ...)."""
    out = {"provider": "", "track_raw": "", "off_time": "", "grade_distance": ""}
    s = str(event_path or "").strip()
    if not s:
        return out
    m_prov = re.search(r"Greyhound Racing\s*/\s*([^/]+?)(?:\s*/\s*|$)", s)
    if m_prov:
        out["provider"] = m_prov.group(1).strip()
    m_time = re.search(r"/\s*(\d{1,2}:\d{2})\s+([A-Za-z\s]+?)\s+\d{1,2}(?:st|nd|rd|th)\s+\w+\s+-", s)
    if m_time:
        out["off_time"] = m_time.group(1).strip()
        out["track_raw"] = m_time.group(2).strip()
    if not out["track_raw"]:
        m_track = re.search(r"\d{1,2}:\d{2}\s+([A-Za-z\s]+?)\s+\d{1,2}(?:st|nd|rd|th)", s)
        if m_track:
            out["track_raw"] = m_track.group(1).strip()
        m_off = re.search(r"(\d{1,2}:\d{2})\s+[A-Za-z]", s)
        if m_off:
            out["off_time"] = m_off.group(1).strip()
    m_grade = re.search(r"\s-\s+([A-Z]\d+\s+\d+m)\s*-", s)
    if m_grade:
        out["grade_distance"] = m_grade.group(1).strip()
    if not out["grade_distance"]:
        m_grade = re.search(r"([A-Z]\d+\s+\d+m)", s, re.IGNORECASE)
        if m_grade:
            out["grade_distance"] = m_grade.group(1).strip()
    return out


def parse_marketfeeder_statement(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Le CSV do MarketFeeder (sem header). Retorna (df_bets, df_market_pl, total_commission)."""
    import csv
    buf = io.BytesIO(file_bytes)
    reader = csv.reader(io.TextIOWrapper(buf, encoding="utf-8", errors="replace"), quoting=csv.QUOTE_MINIMAL)
    rows = list(reader)
    if not rows:
        return (pd.DataFrame(columns=_MF_STATEMENT_COLS), pd.DataFrame(columns=_MF_STATEMENT_COLS), 0.0)
    n_cols = len(_MF_STATEMENT_COLS)
    data = []
    for row in rows:
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        else:
            row = row[:n_cols]
        data.append(row)
    df = pd.DataFrame(data, columns=_MF_STATEMENT_COLS)
    df["placed_dt"] = pd.to_datetime(df["placed_dt"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for col in ("stake", "price", "profit", "balance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df_bets = df[df["type"].astype(str).str.strip().isin(("Back", "Lay"))].copy()
    df_market_pl = df[df["type"].astype(str).str.strip() == "Market P/L"].copy()
    df_commission = df[df["type"].astype(str).str.strip() == "Commission"]
    total_commission = float(pd.to_numeric(df_commission["profit"], errors="coerce").fillna(0).sum())
    if df_bets.empty:
        return (df_bets, df_market_pl, total_commission)
    df_bets["entry_type"] = df_bets["type"].astype(str).str.strip().str.lower()
    trap_series = df_bets["selection"].astype(str).str.extract(r"^\s*(\d+)\.\s*", expand=False)
    df_bets["selection_trap"] = pd.to_numeric(trap_series, errors="coerce").astype("Int64")
    df_bets["selection_name"] = df_bets["selection"].astype(str).str.replace(r"^\s*\d+\.\s*", "", regex=True).str.strip()
    parsed = df_bets["event_path"].astype(str).map(_parse_event_path_mf)
    df_bets["provider"] = parsed.map(lambda x: x["provider"])
    df_bets["track_raw"] = parsed.map(lambda x: x["track_raw"])
    df_bets["off_time"] = parsed.map(lambda x: x["off_time"])
    df_bets["grade_distance"] = parsed.map(lambda x: x["grade_distance"])
    df_bets["track_clean"] = df_bets["track_raw"].astype(str).map(normalize_track_name)

    def _combine_race_time(row: pd.Series) -> str:
        dt = row.get("placed_dt")
        off = str(row.get("off_time") or "").strip()
        if pd.isna(dt) or not off:
            return ""
        try:
            d = dt.date() if hasattr(dt, "date") else dt
            parts = off.split(":")
            h = int(parts[0]) if len(parts) > 0 else 0
            m = int(parts[1]) if len(parts) > 1 else 0
            t = datetime.time(h, m, 0)
            combined = datetime.datetime.combine(d, t)
            return combined.strftime("%Y-%m-%dT%H:%M")
        except Exception:
            return ""

    df_bets["race_time_iso"] = df_bets.apply(_combine_race_time, axis=1)
    race_date_series = pd.to_datetime(df_bets["race_time_iso"], format="%Y-%m-%dT%H:%M", errors="coerce").dt.strftime("%Y-%m-%d")
    df_bets["race_date"] = race_date_series.fillna("")
    sel_name_clean = df_bets["selection_name"].fillna("").astype(str).str.strip()
    df_bets["join_key_stmt"] = (
        df_bets["race_date"].astype(str) + "|" +
        df_bets["track_clean"].fillna("").astype(str) + "|" +
        df_bets["off_time"].fillna("").astype(str) + "|" +
        sel_name_clean + "|" +
        df_bets["entry_type"].astype(str)
    )
    return (df_bets, df_market_pl, total_commission)


def _is_likely_marketfeeder_statement_csv(file_bytes: bytes) -> bool:
    """True se a primeira celula for 'Back' ou 'Market P/L' (CSV sem header MarketFeeder)."""
    import csv
    buf = io.BytesIO(file_bytes)
    try:
        reader = csv.reader(io.TextIOWrapper(buf, encoding="utf-8", errors="replace"), quoting=csv.QUOTE_MINIMAL)
        first_row = next(reader, None)
    except Exception:
        return False
    if not first_row:
        return False
    first_cell = str(first_row[0]).strip() if first_row else ""
    return first_cell in ("Back", "Market P/L")


def _calc_drawdown_series(series: pd.Series) -> float:
    """Drawdown maximo (peak-to-trough) sobre serie cumulativa de profit."""
    if series.empty:
        return 0.0
    cum = series.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


DATE_KEYS_IGNORED_ON_IMPORT = ("date_mode", "date_start_input", "date_end_input", "date_range_slider")

_EXPORT_BASIC_ORDER = (
    ["strategy_name", "created_at", "import_filename", "rule_select_label", "source_select_label", "provider", "market", "entry_type"]
    + list(CORE_STATE_KEYS)
    + ["leader_min", "forecast_rank_ms", "value_ratio_min", "value_ratio_max", "only_value_bets"]
)


def strategies_list_to_csv_bytes(strategies: List[dict]) -> bytes:
    """Exporta lista de estrategias para CSV (uma linha por estrategia)."""
    if not strategies:
        return b""
    all_keys = set()
    for s in strategies:
        all_keys.update(s.keys())
    all_keys.discard("pnl")
    all_keys.discard("roi")
    for k in DATE_KEYS_IGNORED_ON_IMPORT:
        all_keys.discard(k)
    ordered_basic = [c for c in _EXPORT_BASIC_ORDER if c in all_keys]
    rest = sorted(all_keys - set(ordered_basic))
    columns = ordered_basic + rest
    rows = []
    for s in strategies:
        row = {}
        for k in columns:
            if k == "strategy_name":
                v = _regen_strategy_name_from_dict(s)
            else:
                v = s.get(k)
            if isinstance(v, (list, tuple)):
                row[k] = json.dumps(v)
            elif hasattr(v, "isoformat"):
                row[k] = v.isoformat()
            else:
                row[k] = v if v is not None else ""
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def _regen_strategy_name_from_dict(s: dict) -> str:
    """Regenera strategy_name a partir dos campos do dict (sem PNL/ROI)."""
    rule_label = s.get("rule_select_label", "")
    source_label = s.get("source_select_label", "")
    market = s.get("market", "")
    entry_type = s.get("entry_type", "")
    if not rule_label and not source_label:
        existing = (s.get("strategy_name") or "Sem nome").strip()
        idx = existing.find("PNL:")
        if idx != -1:
            existing = existing[:idx].rstrip(" \u2022*").strip()
        return existing or "Sem nome"
    rule_slug = _rule_label_to_slug(rule_label)
    if not rule_slug:
        rule_slug = rule_label
    source_slug = "top3" if (source_label or "").strip() == "Top 3" else "forecast"
    tracks_ms = s.get("tracks_ms")
    cats_ms = s.get("cats_ms")
    if isinstance(tracks_ms, str) and tracks_ms.strip().startswith("["):
        try:
            tracks_ms = json.loads(tracks_ms)
        except (json.JSONDecodeError, TypeError):
            tracks_ms = None
    if isinstance(cats_ms, str) and cats_ms.strip().startswith("["):
        try:
            cats_ms = json.loads(cats_ms)
        except (json.JSONDecodeError, TypeError):
            cats_ms = None
    bsp_low = s.get("bsp_low")
    bsp_high = s.get("bsp_high")
    return format_strategy_name(
        rule_slug, source_slug, market, entry_type,
        tracks_ms=tracks_ms, cats_ms=cats_ms,
        bsp_low=bsp_low, bsp_high=bsp_high,
    )


def _strategy_list(d: dict, key: str):
    """Extrai lista do strategy dict (list ou JSON string)."""
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, list):
        return v if v else None
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def get_group_key(strategy_dict: dict) -> Tuple[str, str, str, str]:
    """Extrai (source_slug, market, rule_slug, provider) de um strategy_dict."""
    source_val = strategy_dict.get("source_select_label", "Top 3")
    market_val = strategy_dict.get("market", "win")
    rule_val = strategy_dict.get("rule_select_label", "")
    rule_slug = _rule_label_to_slug(rule_val) if rule_val else "terceiro_queda50"
    source_slug = "top3" if (str(source_val or "").strip() == "Top 3") else "forecast"
    provider = strategy_dict.get("provider", "timeform")
    if rule_slug == "forecast_odds":
        provider = "timeform"
    return (source_slug, market_val, rule_slug, provider)


def _shorten_filename(name: Optional[str], max_len: int = 55) -> str:
    """Encurta nome de arquivo preservando extensao."""
    if not name or not (name := name.strip()):
        return ""
    if len(name) <= max_len:
        return name
    if "." in name:
        base, ext = name.rsplit(".", 1)
        ext = "." + ext
    else:
        base, ext = name, ""
    available = max_len - len(ext) - 3
    if available <= 0:
        return name[:max_len]
    prefix_len = available // 2
    suffix_len = available - prefix_len
    return f"{base[:prefix_len]}...{base[-suffix_len:]}{ext}"


def _format_strategy_line(strategy_dict: dict, tail: str = "") -> str:
    """Formata uma linha em markdown: [import_filename] -> strategy_name."""
    fname = (strategy_dict.get("import_filename") or "").strip()
    sname = (strategy_dict.get("strategy_name") or "Sem nome").strip()
    if fname:
        fname = _shorten_filename(fname)
    prefix = f"[{fname}] -> " if fname else ""
    return f"{prefix}**{sname}**"


def _visualize_label(strategy_dict: dict) -> str:
    """Label da opcao 'Visualizar' para uma estrategia consolidada."""
    name = (strategy_dict.get("strategy_name") or "Sem nome").strip()
    fname = (strategy_dict.get("import_filename") or "").strip()
    if fname:
        fname = _shorten_filename(fname)
        return f"Estrategia: [{fname}] -> {name}"
    return f"Estrategia: {name}"


def _compute_hour_bucket_series(series: pd.Series) -> pd.Series:
    """Retorna serie de buckets horarios (08:00-12:00, etc.) a partir de race_time_iso."""
    ts = pd.to_datetime(series, format=_RACE_TIME_ISO_FORMAT, errors="coerce")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    bucket = pd.Series(index=series.index, dtype="object")
    bucket[(minutes >= 8 * 60) & (minutes <= 12 * 60)] = "08:00-12:00"
    bucket[(minutes >= 12 * 60 + 1) & (minutes <= 16 * 60)] = "12:01-16:00"
    bucket[(minutes >= 16 * 60 + 1) & (minutes <= 20 * 60)] = "16:01-20:00"
    bucket[(minutes >= 20 * 60 + 1)] = "20:01-23:59"
    return bucket


def _iter_result_paths(pattern: str) -> List[Path]:
    """Preferencialmente parquets em processed/Result, senao Result (parquet ou csv)."""
    parquet_processed = sorted(settings.PROCESSED_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_processed:
        return parquet_processed
    parquet_raw = sorted(settings.RAW_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_raw:
        return parquet_raw
    return sorted(settings.RAW_RESULT_DIR.glob(f"{pattern}.csv"))


def _stat_signature(paths: List[Path]) -> Tuple[Tuple[str, float, str], ...]:
    signature: List[Tuple[str, float, str]] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path), stat.st_mtime, path.suffix.lstrip(".").lower()))
    return tuple(signature)


def _cache_dir() -> Path:
    """Diretorio para cache em disco (indices category/num_runners)."""
    d = Path(settings.PROCESSED_DIR) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_result_signature_cached() -> Tuple[Tuple[str, float, str], ...]:
    """Assinatura dos ficheiros de resultado (uma vez por run, reutilizada)."""
    key = "_result_signature"
    if key not in st.session_state:
        paths = _iter_result_paths("dwbfprices*win*")
        st.session_state[key] = _stat_signature(paths)
    return st.session_state[key]


@st.cache_data(show_spinner=False)
def _read_dataframe_cached(
    path_str: str,
    mtime: float,
    file_type: str,
    columns: Tuple[str, ...] | None = None,
) -> pd.DataFrame:
    del mtime  # usado apenas para chave de cache
    if file_type == "parquet":
        if columns is not None:
            return pd.read_parquet(path_str, columns=list(columns))
        return pd.read_parquet(path_str)

    read_kwargs: dict = {
        "encoding": settings.CSV_ENCODING,
        "engine": "python",
        "on_bad_lines": "skip",
    }
    if columns is not None:
        read_kwargs["usecols"] = list(columns)
    return pd.read_csv(path_str, **read_kwargs)


@st.cache_data(show_spinner=False)
def _cached_category_index(
    files_signature: Tuple[Tuple[str, float, str], ...],
) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    columns = ("menu_hint", "event_dt", "event_name")
    for path_str, mtime, file_type in files_signature:
        try:
            df_r = _read_dataframe_cached(path_str, mtime, file_type, columns)
        except Exception:
            continue
        df_r = df_r.dropna(subset=list(columns), how="any")
        if df_r.empty:
            continue
        df_r["track_key"] = df_r["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df_r["race_iso"] = _to_iso_series(df_r["event_dt"].astype(str))
        df_r["cat_letter"] = df_r["event_name"].astype(str).map(_extract_category_letter)
        df_r["cat_token"] = df_r["event_name"].astype(str).map(_extract_category_token)
        for _, r in df_r.iterrows():
            key = (str(r["track_key"]), str(r["race_iso"]))
            if not key[0] or not key[1]:
                continue
            if key not in mapping:
                mapping[key] = {
                    "letter": str(r.get("cat_letter", "")),
                    "token": str(r.get("cat_token", "")),
                }
    return mapping


@st.cache_data(show_spinner=False)
def _cached_num_runners_index(
    files_signature: Tuple[Tuple[str, float, str], ...],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    columns = ("menu_hint", "event_dt")
    for path_str, mtime, file_type in files_signature:
        try:
            df_r = _read_dataframe_cached(path_str, mtime, file_type, columns)
        except Exception:
            continue
        if df_r.empty:
            continue
        df_r["track_key"] = df_r["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df_r["race_iso"] = _to_iso_series(df_r["event_dt"].astype(str))
        grp = df_r.groupby(["track_key", "race_iso"], dropna=False).size()
        for (tk, ri), n in grp.items():
            if isinstance(tk, str) and tk and isinstance(ri, str) and ri:
                counts[(tk, ri)] = int(n)
    return counts


def _build_category_index() -> dict[tuple[str, str], dict[str, str]]:
    """Indice (track_key, race_iso) -> {letter, token}. Usa cache em disco por assinatura dos ficheiros."""
    signature = _get_result_signature_cached()
    cache_key = hashlib.md5(str(signature).encode()).hexdigest()
    cache_path = _cache_dir() / f"horses_category_index_{cache_key}.pkl"
    try:
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                stored = pickle.load(f)
            if stored.get("signature") == signature:
                return stored.get("data") or {}
    except Exception:
        pass
    result = _cached_category_index(signature)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"signature": signature, "data": result}, f)
    except Exception:
        pass
    return result


def _build_num_runners_index() -> dict[tuple[str, str], int]:
    """Indice (track_key, race_iso) -> num_runners. Usa cache em disco por assinatura dos ficheiros."""
    signature = _get_result_signature_cached()
    cache_key = hashlib.md5(str(signature).encode()).hexdigest()
    cache_path = _cache_dir() / f"horses_num_runners_index_{cache_key}.pkl"
    try:
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                stored = pickle.load(f)
            if stored.get("signature") == signature:
                return stored.get("data") or {}
    except Exception:
        pass
    result = _cached_num_runners_index(signature)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"signature": signature, "data": result}, f)
    except Exception:
        pass
    return result


NR_BUCKET_LABELS = ["1-4", "5-8", "9-13", "14-19", "20+"]


def _bucket_num_runners(series: pd.Series) -> pd.Series:
    """Converte numero de corredores em faixas padronizadas."""
    if series is None:
        return pd.Series(dtype="string")
    numeric = pd.to_numeric(series, errors="coerce")
    buckets = pd.cut(
        numeric,
        bins=[0, 4, 8, 13, 19, float("inf")],
        labels=NR_BUCKET_LABELS,
        right=True,
        include_lowest=True,
    )
    return buckets.astype("string")


def _make_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Chave por linha para dedup: race_time_iso|track_name|market|entry_type|target_name."""
    if df.empty:
        return pd.Series(dtype=str)
    race = df.get("race_time_iso", pd.Series("", index=df.index)).astype(str).fillna("")
    track = df.get("track_name", pd.Series("", index=df.index)).astype(str).fillna("")
    mkt = df.get("market", pd.Series("win", index=df.index))
    if isinstance(mkt, pd.Series):
        mkt = mkt.astype(str).fillna("")
    else:
        mkt = pd.Series(str(mkt), index=df.index)
    entry = df.get("entry_type", pd.Series("", index=df.index)).astype(str).fillna("")
    back_t = df.get("back_target_name", pd.Series("", index=df.index)).astype(str).fillna("")
    lay_t = df.get("lay_target_name", pd.Series("", index=df.index)).astype(str).fillna("")
    target = pd.Series("", index=df.index)
    target[entry == "back"] = back_t[entry == "back"]
    target[entry == "lay"] = lay_t[entry == "lay"]
    other = (entry != "back") & (entry != "lay")
    target[other] = back_t[other].where(back_t[other] != "", lay_t[other])
    return race + "|" + track + "|" + mkt + "|" + entry + "|" + target


def _build_mask_from_strategy(df: pd.DataFrame, strategy_dict: dict) -> pd.Series:
    """Mascara booleana: True onde a linha atende aos filtros do strategy_dict (cavalos: sem trap, com num_runners_bucket e leader_min)."""
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)
    work = df.copy()
    rule_slug = _rule_label_to_slug(strategy_dict.get("rule_select_label", ""))
    entry_raw = strategy_dict.get("entry_type")
    entry = str(entry_raw).strip().lower() if entry_raw is not None else ""
    if entry in ("back", "lay") and "entry_type" in df.columns:
        entry_series = df["entry_type"].astype(str).str.strip().str.lower()
        mask &= (entry_series == entry).fillna(False)

    if rule_slug != "forecast_odds":
        min_vol = strategy_dict.get("min_total_volume")
        if min_vol is not None and "total_matched_volume" in df.columns:
            vol = pd.to_numeric(df["total_matched_volume"], errors="coerce").fillna(0.0)
            try:
                mask &= vol >= float(min_vol)
            except (TypeError, ValueError):
                pass

    tracks = _strategy_list(strategy_dict, "tracks_ms")
    if tracks and "track_name" in df.columns:
        mask &= df["track_name"].astype(str).isin([str(t) for t in tracks])

    if ("category" not in work.columns or "category_token" not in work.columns) and "track_name" in work.columns and "race_time_iso" in work.columns:
        cat_index = _build_category_index()
        work["_key_track"] = work["track_name"].astype(str).map(normalize_track_name)
        work["_key_race"] = work["race_time_iso"].astype(str)
        keys_series = pd.Series(list(zip(work["_key_track"], work["_key_race"])), index=work.index)
        work["category"] = keys_series.map(lambda k: (cat_index.get(k, {}) or {}).get("letter", "")).fillna("").astype(str)
        work["category_token"] = keys_series.map(lambda k: (cat_index.get(k, {}) or {}).get("token", "")).fillna("").astype(str)

    weekdays_ms = _strategy_list(strategy_dict, "weekdays_ms")
    if weekdays_ms and "race_time_iso" in work.columns:
        wd_names = {"Seg": 0, "Ter": 1, "Qua": 2, "Qui": 3, "Sex": 4, "Sab": 5, "Dom": 6}
        sel_nums = [wd_names.get(str(w), w) for w in weekdays_ms if isinstance(w, (int, str)) and (isinstance(w, int) or str(w) in wd_names)]
        if sel_nums:
            ts_filt = pd.to_datetime(work["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            mask &= ts_filt.dt.weekday.isin(sel_nums).fillna(False)

    hb = _strategy_list(strategy_dict, "hour_bucket_ms")
    if hb and "race_time_iso" in work.columns:
        bucket_series = _compute_hour_bucket_series(work["race_time_iso"])
        mask &= bucket_series.isin(hb).fillna(False)

    nr_bucket = _strategy_list(strategy_dict, "num_runners_bucket_ms")
    if nr_bucket is not None and "num_runners" in work.columns:
        work = work.copy()
        work["num_runners_bucket"] = _bucket_num_runners(work["num_runners"])
        if nr_bucket:
            mask &= work["num_runners_bucket"].astype(str).isin([str(x) for x in nr_bucket]).fillna(False)
        else:
            mask &= pd.Series(False, index=df.index)

    bsp_low = strategy_dict.get("bsp_low")
    bsp_high = strategy_dict.get("bsp_high")
    back_col = "back_target_bsp"
    lay_col = "lay_target_bsp"
    if bsp_low is not None and bsp_high is not None:
        try:
            low, high = float(bsp_low), float(bsp_high)
            has_back = back_col in work.columns
            has_lay = lay_col in work.columns
            if entry == "back" and has_back:
                bsp_ser = pd.to_numeric(work[back_col], errors="coerce")
                mask &= ((bsp_ser >= low) & (bsp_ser <= high)).fillna(False)
            elif entry == "lay" and has_lay:
                bsp_ser = pd.to_numeric(work[lay_col], errors="coerce")
                mask &= ((bsp_ser >= low) & (bsp_ser <= high)).fillna(False)
            elif entry in ("both", "ambos", ""):
                part_back = (pd.to_numeric(work[back_col], errors="coerce") >= low) & (pd.to_numeric(work[back_col], errors="coerce") <= high) if has_back else pd.Series(False, index=df.index)
                part_lay = (pd.to_numeric(work[lay_col], errors="coerce") >= low) & (pd.to_numeric(work[lay_col], errors="coerce") <= high) if has_lay else pd.Series(False, index=df.index)
                if has_back and has_lay:
                    mask &= (part_back | part_lay).fillna(False)
                elif has_back:
                    mask &= part_back.fillna(False)
                elif has_lay:
                    mask &= part_lay.fillna(False)
        except (TypeError, ValueError):
            pass

    cats = _strategy_list(strategy_dict, "cats_ms")
    if cats and "category" in work.columns:
        mask &= work["category"].astype(str).isin([str(c) for c in cats])

    subcats = _strategy_list(strategy_dict, "subcats_ms")
    if subcats and "category_token" in work.columns:
        mask &= work["category_token"].astype(str).isin([str(s) for s in subcats])

    if rule_slug == "lider_volume_total":
        leader_min = strategy_dict.get("leader_min")
        if leader_min is not None and "leader_volume_share_pct" in df.columns:
            try:
                mask &= (pd.to_numeric(df["leader_volume_share_pct"], errors="coerce").fillna(0) >= float(leader_min))
            except (TypeError, ValueError):
                pass

    if rule_slug == "forecast_odds":
        ranks = _strategy_list(strategy_dict, "forecast_rank_ms")
        vmin = strategy_dict.get("value_ratio_min")
        vmax = strategy_dict.get("value_ratio_max")
        only_value = strategy_dict.get("only_value_bets")
        if isinstance(only_value, str):
            only_value = only_value.strip().lower() in ("true", "1", "yes")
        if ranks and "forecast_rank" in df.columns:
            try:
                rank_series = pd.to_numeric(df["forecast_rank"], errors="coerce")
                rank_ints = [int(x) for x in ranks if x is not None]
                if rank_ints:
                    mask &= rank_series.isin(rank_ints).fillna(False)
            except (TypeError, ValueError):
                pass
        if vmin is not None and vmax is not None and "value_ratio" in df.columns:
            try:
                vr = pd.to_numeric(df["value_ratio"], errors="coerce").fillna(float("nan"))
                mask &= vr.between(float(vmin), float(vmax)).fillna(False)
            except (TypeError, ValueError):
                pass
        if only_value and "value_ratio" in df.columns:
            try:
                vr_fill = pd.to_numeric(df["value_ratio"], errors="coerce").fillna(0.0)
                mask &= (vr_fill >= 1.0).fillna(False)
            except (TypeError, ValueError):
                pass

    if rule_slug == "terceiro_queda50" and "pct_diff_second_vs_third" in df.columns:
        mask &= (pd.to_numeric(df["pct_diff_second_vs_third"], errors="coerce").fillna(0) > 50.0)

    return mask.reindex(df.index, fill_value=False).fillna(False)


def _build_consolidated_df_by_groups(strategies: List[dict]) -> Tuple[pd.DataFrame, int, int, int]:
    """Agrupa estrategias por (source, market, rule, provider), carrega signals, aplica OR das mascaras, dedup. Retorna (df_union, total_before, total_after, overlap)."""
    grouped: dict[Tuple[str, str, str, str], List[dict]] = {}
    for s in strategies:
        gk = get_group_key(s)
        grouped.setdefault(gk, []).append(s)

    frames: List[pd.DataFrame] = []
    total_before_sum = 0
    for (source, market, rule, provider), strategies_in_group in grouped.items():
        try:
            signals_mtime = _get_signals_mtime(source, market, rule, provider)
            df_group, _ = load_signals_enriched(source=source, market=market, rule=rule, provider=provider, signals_mtime=signals_mtime)
        except Exception:
            continue
        if df_group.empty:
            continue
        mask_group = None
        for strat in strategies_in_group:
            m = _build_mask_from_strategy(df_group, strat)
            mask_group = m if mask_group is None else (mask_group | m)
        df_group_sel = df_group[mask_group].copy() if mask_group is not None else df_group.iloc[0:0].copy()
        df_group_sel["dedup_key"] = _make_dedup_key(df_group_sel)
        before = len(df_group_sel)
        df_group_sel = df_group_sel.drop_duplicates(subset=["dedup_key"])
        total_before_sum += before
        frames.append(df_group_sel)

    if not frames:
        return (pd.DataFrame(), 0, 0, 0)
    df_union = pd.concat(frames, ignore_index=True)
    df_union = df_union.drop_duplicates(subset=["dedup_key"])
    total_after = len(df_union)
    overlap = total_before_sum - total_after
    return (df_union, total_before_sum, total_after, overlap)


def _get_dates_in_range_from_state(df: pd.DataFrame, session_state: dict) -> List[str]:
    """
    Calcula dates_in_range a partir de session_state (sem renderizar widgets).
    Respeita date_mode: Calendario usa apenas date_start_input/date_end_input;
    Barra usa apenas date_range_slider.
    """
    raw_date_values = df["date"].dropna().astype(str).unique().tolist()
    parsed_dates = []
    for date_str in raw_date_values:
        parsed = pd.to_datetime(date_str, errors="coerce")
        if pd.notna(parsed):
            parsed_dates.append((date_str, parsed.date()))
    parsed_dates.sort(key=lambda item: item[1])
    if not parsed_dates:
        return sorted(raw_date_values) if raw_date_values else []
    min_date = parsed_dates[0][1]
    max_date = parsed_dates[-1][1]

    def _norm_dt(value: object, fallback: datetime.date) -> datetime.date:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        if isinstance(value, str):
            p = pd.to_datetime(value, errors="coerce")
            if pd.notna(p):
                return p.date()
        return fallback

    date_mode = session_state.get("date_mode", "Calendário")
    date_start_key = "date_start_input"
    date_end_key = "date_end_input"
    slider_key = "date_range_slider"

    if date_mode == "Calendário":
        stored_start = _norm_dt(session_state.get(date_start_key, min_date), min_date)
        stored_end = _norm_dt(session_state.get(date_end_key, max_date), max_date)
        range_start = max(min_date, min(stored_start, max_date))
        range_end = max(min_date, min(stored_end, max_date))
        if range_start > range_end:
            range_end = range_start
    else:
        current = session_state.get(slider_key, None)
        if isinstance(current, (tuple, list)) and len(current) == 2:
            start_norm = current[0]
            end_norm = current[1]
            if isinstance(start_norm, pd.Timestamp):
                start_norm = start_norm.to_pydatetime()
            if isinstance(end_norm, pd.Timestamp):
                end_norm = end_norm.to_pydatetime()
            if isinstance(start_norm, datetime.date) and not isinstance(start_norm, datetime.datetime):
                start_norm = datetime.datetime.combine(start_norm, datetime.time.min)
            if isinstance(end_norm, datetime.date) and not isinstance(end_norm, datetime.datetime):
                end_norm = datetime.datetime.combine(end_norm, datetime.time.min)
            slider_min_dt = datetime.datetime.combine(min_date, datetime.time.min)
            slider_max_dt = datetime.datetime.combine(max_date, datetime.time.min)
            start_norm = max(slider_min_dt, min(start_norm, slider_max_dt))
            end_norm = max(slider_min_dt, min(end_norm, slider_max_dt))
            if start_norm > end_norm:
                end_norm = start_norm
            range_start = start_norm.date()
            range_end = end_norm.date()
        else:
            range_start, range_end = min_date, max_date
        range_start = max(min_date, range_start)
        range_end = min(max_date, range_end)
        if range_start > range_end:
            range_end = range_start

    if DEBUG_DATES:
        session_state["_debug_dates_effective"] = {
            "date_mode": date_mode,
            "start": range_start,
            "end": range_end,
        }

    return [d for d, parsed in parsed_dates if range_start <= parsed <= range_end]


def _apply_filters_to_df_filtered(
    df_filtered: pd.DataFrame,
    session_state: dict,
    rule: str,
    entry_type: str,
) -> pd.DataFrame:
    """
    Aplica os mesmos filtros que o dashboard (sem widgets). Retorna filt.
    Cavalos: num_runners_bucket, leader_min; sem trap.

    Ordem de aplicacao (fonte unica para Fase 2):
    1. Enriquecimento num_runners (se faltar)
    2. Volume (min_total_volume), exceto regra forecast_odds
    3. Enriquecimento category/category_token (para opcoes)
    4. Weekday (weekdays_ms)
    5. Hour bucket (hour_bucket_ms)
    6. Regra lider_volume_total: leader_min
    7. Regra forecast_odds: forecast_rank_ms, value_ratio, only_value_bets
    8. Pistas (tracks_ms)
    9. Regra terceiro_queda50: pct_diff_second_vs_third > 50
    10. BSP (bsp_low, bsp_high)
    11. Categorias (cats_ms)
    12. Subcategorias (subcats_ms)
    13. Numero de corredores bucket (num_runners_bucket_ms)
    14. Entry type (back/lay)
    """
    filt = df_filtered.copy()
    if "num_runners" not in filt.columns and not filt.empty:
        num_index = _build_num_runners_index()
        filt["_key_track"] = filt["track_name"].astype(str).map(normalize_track_name)
        filt["_key_race"] = filt["race_time_iso"].astype(str)
        filt["num_runners"] = filt.apply(
            lambda r: num_index.get((str(r["_key_track"]), str(r["_key_race"])), pd.NA),
            axis=1,
        )
    if rule != "forecast_odds":
        volume_key = "min_total_volume"
        min_total_volume = float(session_state.get(volume_key, 0.0))
        volume_series = pd.to_numeric(filt.get("total_matched_volume", pd.Series(dtype=float)), errors="coerce")
        filt = filt[volume_series.fillna(0.0) >= min_total_volume]

    if ("category" not in filt.columns or "category_token" not in filt.columns) and not filt.empty:
        cat_index = _build_category_index()
        filt = filt.copy()
        filt["_key_track"] = filt["track_name"].astype(str).map(normalize_track_name)
        filt["_key_race"] = filt["race_time_iso"].astype(str)
        filt["category"] = filt.apply(
            lambda r: (cat_index.get((str(r["_key_track"]), str(r["_key_race"])), {}) or {}).get("letter", ""),
            axis=1,
        )
        filt["category_token"] = filt.apply(
            lambda r: (cat_index.get((str(r["_key_track"]), str(r["_key_race"])), {}) or {}).get("token", ""),
            axis=1,
        )
    cat_letters = sorted([c for c in filt["category"].dropna().unique().tolist() if isinstance(c, str) and c]) if not filt.empty and "category" in filt.columns else []
    sub_tokens = []
    if "category_token" in filt.columns and not filt.empty:
        sel_cats_state = session_state.get("sel_cats") or session_state.get("cats_ms")
        token_source = filt[filt["category"].isin(sel_cats_state)] if sel_cats_state and "category" in filt.columns else filt
        raw_tokens = [t for t in token_source["category_token"].dropna().astype(str).unique().tolist() if isinstance(t, str) and t]
        def _sub_sort(tok: str) -> tuple:
            m = re.match(r"^([A-Z]+)(\d+)$", str(tok))
            if m:
                return (m.group(1), int(m.group(2)))
            m2 = re.match(r"^([A-Z]+)", str(tok))
            return ((m2.group(1) if m2 else str(tok)), 0)
        sub_tokens = sorted(raw_tokens, key=_sub_sort)

    wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
    sel_weekdays_nums = [num for num, label in wd_names.items() if label in session_state.get("weekdays_ms", [])]
    if sel_weekdays_nums is not None and sel_weekdays_nums:
        if "race_time_iso" in filt.columns and not filt.empty:
            ts_filt = pd.to_datetime(filt["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            filt = filt[ts_filt.dt.weekday.isin(sel_weekdays_nums)]
        else:
            filt = filt.iloc[0:0]

    sel_hour_buckets = session_state.get("hour_bucket_ms", None)
    if sel_hour_buckets:
        if "race_time_iso" in filt.columns and not filt.empty:
            hb_filt = _compute_hour_bucket_series(filt["race_time_iso"])
            filt = filt[hb_filt.isin(sel_hour_buckets)]
        else:
            filt = filt.iloc[0:0]

    if rule == "lider_volume_total":
        leader_min = float(session_state.get("leader_min", 50.0))
        if "leader_volume_share_pct" in filt.columns:
            filt = filt[filt["leader_volume_share_pct"].fillna(0) >= leader_min]

    if rule == "forecast_odds":
        if "forecast_rank" in filt.columns:
            filt["forecast_rank"] = pd.to_numeric(filt["forecast_rank"], errors="coerce")
        if "value_ratio" in filt.columns:
            filt["value_ratio"] = pd.to_numeric(filt["value_ratio"], errors="coerce")
        sel_ranks = session_state.get("forecast_rank_ms", None)
        if sel_ranks and "forecast_rank" in filt.columns:
            filt = filt[filt["forecast_rank"].isin(sel_ranks)]
        vr_min = float(session_state.get("value_ratio_min", 0.0))
        vr_max = float(session_state.get("value_ratio_max", 1.0))
        if "value_ratio" in filt.columns:
            filt = filt[filt["value_ratio"].fillna(float("nan")).between(vr_min, vr_max)]
        if session_state.get("only_value_bets", False) and "value_ratio" in filt.columns:
            filt = filt[filt["value_ratio"].fillna(0.0) >= 1.0]

    sel_tracks = session_state.get("tracks_ms", None)
    if sel_tracks and "track_name" in filt.columns:
        filt = filt[filt["track_name"].isin(sel_tracks)]
    if rule == "terceiro_queda50" and "pct_diff_second_vs_third" in filt.columns:
        filt = filt[filt["pct_diff_second_vs_third"].fillna(0) > 50.0]

    bsp_low = float(session_state.get("bsp_low", 1.01))
    bsp_high = float(session_state.get("bsp_high", 100.0))
    if entry_type == "both":
        filt = filt[
            ((filt["entry_type"] == "lay") & (filt["lay_target_bsp"].between(bsp_low, bsp_high)))
            | ((filt["entry_type"] == "back") & (filt["back_target_bsp"].between(bsp_low, bsp_high)))
        ]
    else:
        bsp_col = "lay_target_bsp" if entry_type == "lay" else "back_target_bsp"
        if bsp_col in filt.columns:
            filt = filt[(filt[bsp_col] >= bsp_low) & (filt[bsp_col] <= bsp_high)]

    sel_cats = session_state.get("sel_cats") or session_state.get("cats_ms")
    if cat_letters and sel_cats is not None:
        if sel_cats and "category" in filt.columns:
            filt = filt[filt["category"].isin(sel_cats)]
        else:
            filt = filt.iloc[0:0]

    sel_subcats = session_state.get("sel_subcats") or session_state.get("subcats_ms")
    if sub_tokens and sel_subcats is not None:
        if sel_subcats and "category_token" in filt.columns:
            filt = filt[filt["category_token"].isin(sel_subcats)]
        else:
            filt = filt.iloc[0:0]

    sel_nr_bucket = session_state.get("sel_num_runners_bucket") or session_state.get("num_runners_bucket_ms")
    if sel_nr_bucket is not None:
        filt = filt.copy()
        filt["num_runners_bucket"] = _bucket_num_runners(filt.get("num_runners"))
        if sel_nr_bucket and "num_runners_bucket" in filt.columns:
            filt = filt[filt["num_runners_bucket"].astype(str).isin([str(x) for x in sel_nr_bucket])]
        else:
            filt = filt.iloc[0:0]

    if entry_type != "both" and "entry_type" in filt.columns:
        filt = filt[filt["entry_type"] == entry_type]

    return filt


def compute_export_pnl_roi(
    filt_df: pd.DataFrame,
    entry_type: str,
    base_amount: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Calcula PNL e ROI para o export. Retorna (export_pnl, export_roi_pct). ROI em percentual (ex.: 6.8)."""
    if filt_df is None or filt_df.empty:
        return (None, None)
    if entry_type == "back":
        summary = _compute_summary_metrics(filt_df, "back", base_amount)
        return (summary["pnl_stake"], (summary["roi_stake"] * 100.0) if summary["roi_stake"] is not None else None)
    if entry_type == "lay":
        summary = _compute_summary_metrics(filt_df, "lay", base_amount)
        return (summary["pnl_stake"], (summary["roi_stake"] * 100.0) if summary["roi_stake"] is not None else None)
    back_df = filt_df[filt_df["entry_type"] == "back"] if "entry_type" in filt_df.columns else filt_df.iloc[0:0]
    lay_df = filt_df[filt_df["entry_type"] == "lay"] if "entry_type" in filt_df.columns else filt_df.iloc[0:0]
    s_back = _compute_summary_metrics(back_df, "back", base_amount)
    s_lay = _compute_summary_metrics(lay_df, "lay", base_amount)
    export_pnl = s_back["pnl_stake"] + s_lay["pnl_stake"]
    total_base = s_back["base_stake"] + s_lay["base_stake"]
    export_roi_pct = (export_pnl / total_base * 100.0) if total_base > 0 else 0.0
    return (export_pnl, export_roi_pct)


def _calc_drawdown(series: pd.Series) -> float:
    """Retorna o drawdown máximo (maior perda acumulada) em valor negativo."""
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = series - running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _compute_summary_metrics(df_block: pd.DataFrame, entry_kind: str, base_amount: float) -> dict[str, float]:
    """Calcula métricas agregadas usadas nos cabeçalhos, mensais e curvas."""
    metrics: dict[str, float] = {
        "tracks": 0,
        "signals": 0,
        "greens": 0,
        "reds": 0,
        "avg_bsp": 0.0,
        "accuracy": 0.0,
        "base_stake": 0.0,
        "pnl_stake": 0.0,
        "roi_stake": 0.0,
        "min_pnl_stake": 0.0,
        "drawdown_stake": 0.0,
    }
    if df_block is None or df_block.empty:
        return metrics

    scale_factor = base_amount
    metrics["tracks"] = int(df_block["track_name"].nunique())
    metrics["signals"] = int(len(df_block))
    metrics["greens"] = int((df_block["is_green"] == True).sum())
    metrics["reds"] = int(metrics["signals"] - metrics["greens"])

    if entry_kind == "lay":
        metrics["avg_bsp"] = float(df_block["lay_target_bsp"].mean())
    else:
        metrics["avg_bsp"] = float(df_block["back_target_bsp"].mean())
    metrics["accuracy"] = (metrics["greens"] / metrics["signals"]) if metrics["signals"] > 0 else 0.0

    if entry_kind == "lay":
        total_base_stake10 = float(df_block["liability_from_stake_fixed_10"].sum())
    else:
        total_base_stake10 = 1.0 * metrics["signals"]
    total_pnl_stake10 = float(df_block["pnl_stake_fixed_10"].sum())
    metrics["base_stake"] = total_base_stake10 * scale_factor
    metrics["pnl_stake"] = total_pnl_stake10 * scale_factor
    metrics["roi_stake"] = (metrics["pnl_stake"] / metrics["base_stake"]) if metrics["base_stake"] > 0 else 0.0

    sort_col = "race_ts" if "race_ts" in df_block.columns else ("race_time_iso" if "race_time_iso" in df_block.columns else None)
    ordered = df_block.sort_values(sort_col) if sort_col else df_block
    cumulative_stake = (ordered["pnl_stake_fixed_10"] * scale_factor).cumsum()
    metrics["min_pnl_stake"] = float(cumulative_stake.min()) if not cumulative_stake.empty else 0.0
    metrics["drawdown_stake"] = _calc_drawdown(cumulative_stake)

    if entry_kind == "lay":
        metrics["stake_liab"] = float(df_block["stake_for_liability_10"].sum()) * scale_factor
        total_pnl_liab10 = float(df_block["pnl_liability_fixed_10"].sum())
        metrics["pnl_liab"] = total_pnl_liab10 * scale_factor
        metrics["roi_liab"] = (metrics["pnl_liab"] / (base_amount * metrics["signals"])) if metrics["signals"] > 0 and base_amount > 0 else 0.0

        cumulative_liab = (ordered["pnl_liability_fixed_10"] * scale_factor).cumsum()
        metrics["min_pnl_liab"] = float(cumulative_liab.min()) if not cumulative_liab.empty else 0.0
        metrics["drawdown_liab"] = _calc_drawdown(cumulative_liab)
    return metrics


def _format_month_label(ts: pd.Timestamp) -> str:
    """Retorna rótulo no formato Jan/2024 (abreviação em inglês)."""
    if pd.isna(ts):
        return ""
    import calendar

    month_abbr = calendar.month_abbr[ts.month] if ts.month in range(1, 13) else ""
    return f"{month_abbr}/{ts.year}"


def _render_base_amount_input(default: float = 1.0) -> float:
    """Campo único para definir a base de Stake/Liab, usado em todos os modos."""
    if "base_amount" not in st.session_state:
        st.session_state["base_amount"] = float(default)
    col_base, _ = st.columns([1, 6])
    with col_base:
        val = st.number_input(
            "Valor base (Stake e Liability)",
            min_value=0.01,
            max_value=100000.0,
            value=float(st.session_state["base_amount"]),
            step=0.10,
            format="%.2f",
            key="base_amount",
            label_visibility="collapsed",
            help="Padrão: 1 unidade. Ajuste para reescalar PnL/ROI/drawdown.",
        )
        st.caption("Stake/Liab base (padrão: 1 unidade)")
    return float(val)


def _classify_category_from_event_name(event_name: str) -> tuple[str, str]:
    """
    Classificacao especifica para corridas de cavalos.
    Retorna (letter, token) onde:
    - token e uma categoria canonica detalhada (ex.: 'G1','LISTED','HCP_CHS','NOV_HRD','MDN','NURSERY','COND','STKS','NHF','CHS','HRD','OTHER')
    - letter e um rotulo compacto para UI (ex.: 'G','L','H','N','M','Y','C','S','F','C','R','O')
    """
    txt = str(event_name or "").strip()
    # remove distancia inicial (1m6f, 7f, 2m, 2m4f, etc.)
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

    # 1) Group races
    m = re.search(r"\bgroup\s*(1|2|3)\b", s)
    if m:
        return _result(f"G{m.group(1)}")
    m = re.search(r"\bg([123])\b", s)
    if m:
        return _result(f"G{m.group(1)}")

    # 2) Listed
    if re.search(r"\blisted\b", s):
        return _result("LISTED")

    # 3) National Hunt Flat (Bumper)
    if re.search(r"\b(nhf|inhf|bumper)\b", s):
        return _result("NHF")

    # 4) Jumps indicators
    has_chs = bool(re.search(r"\b(chase|chases|chs)\b", s))
    has_hrd = bool(re.search(r"\b(hurdle|hurdles|hrd|hdl)\b", s))

    # 5) Handicap
    if re.search(r"\b(handicap|hcap|hcp)\b", s):
        if has_chs:
            return _result("HCP_CHS")
        if has_hrd:
            return _result("HCP_HRD")
        return _result("HCP")

    # 6) Novice
    if re.search(r"\bnov(?:ice)?\b", s):
        if has_chs:
            return _result("NOV_CHS")
        if has_hrd:
            return _result("NOV_HRD")
        return _result("NOV")

    # 7) Maiden
    if re.search(r"\b(maiden|mdn)\b", s):
        return _result("MDN")

    # 8) Nursery (2yo Handicap)
    if re.search(r"\bnursery\b", s):
        return _result("NURSERY")

    # 9) Conditions
    if re.search(r"\b(conditions|cond)\b", s):
        return _result("COND")

    # 10) Stakes (se nao capturado por Listed/Group)
    if re.search(r"\b(stakes|stks)\b", s):
        return _result("STKS")

    # 11) Puro indicador de jumps sem outras classes
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


def _get_signals_mtime(source: str, market: str, rule: str, provider: str) -> float:
    parquet_path = _signals_snapshot_path(source=source, market=market, rule=rule, provider=provider)
    if parquet_path.exists():
        try:
            return parquet_path.stat().st_mtime
        except OSError:
            return 0.0
    csv_path = _signals_raw_path(source=source, market=market, rule=rule, provider=provider)
    if csv_path.exists():
        try:
            return csv_path.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


@st.cache_data(show_spinner=False)
def load_signals(
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
    provider: str = "timeform",
    signals_mtime: float = 0.0,
) -> pd.DataFrame:
    """
    Carrega sinais (parquet preferencial, CSV legado como fallback) com cache dependente de mtime.
    """
    del signals_mtime  # usado apenas na chave de cache
    parquet_path = _signals_snapshot_path(source=source, market=market, rule=rule, provider=provider)
    if parquet_path.exists():
        try:
            stat = parquet_path.stat()
            return _read_dataframe_cached(
                str(parquet_path),
                stat.st_mtime,
                "parquet",
                None,
            )
        except Exception:
            pass

    csv_path = _signals_raw_path(source=source, market=market, rule=rule, provider=provider)
    if csv_path.exists():
        try:
            stat = csv_path.stat()
            return _read_dataframe_cached(
                str(csv_path),
                stat.st_mtime,
                "csv",
                None,
            )
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _signals_enriched_prebuilt_path(source: str, market: str, rule: str, provider: str) -> Path:
    """Path do parquet pre-enriquecido: data/horses/processed/signals_enriched/<provider>/signals_<source>_<market>_<rule>.parquet."""
    filename = f"signals_{source}_{market}_{rule}.parquet"
    return settings.DATA_DIR / "processed" / "signals_enriched" / provider / filename


@st.cache_data(show_spinner=False)
def _try_load_signals_enriched_prebuilt(
    source: str,
    market: str,
    rule: str,
    provider: str,
    prebuilt_mtime: float = 0.0,
) -> tuple[pd.DataFrame, bool]:
    """
    Tenta carregar o parquet pre-enriquecido. Retorna (df, True) se existir e for valido, senao (vazio, False).
    """
    del prebuilt_mtime
    path = _signals_enriched_prebuilt_path(source=source, market=market, rule=rule, provider=provider)
    if not path.exists():
        return pd.DataFrame(), False
    try:
        stat = path.stat()
        df = _read_dataframe_cached(str(path), stat.st_mtime, "parquet", None)
        if df is None or df.empty:
            return pd.DataFrame(), False
        df = df.copy()
        if "rule" not in df.columns:
            df["rule"] = rule
        if "track_key" in df.columns and "_key_track" not in df.columns:
            df["_key_track"] = df["track_key"].astype(str)
        if "race_iso" in df.columns and "_key_race" not in df.columns:
            df["_key_race"] = df["race_iso"].astype(str)
        return df, True
    except Exception:
        return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def load_signals_enriched(
    source: str,
    market: str,
    rule: str,
    provider: str,
    signals_mtime: float = 0.0,
) -> pd.DataFrame:
    """
    Carrega um único arquivo de sinais (por provider/source/market/rule) e aplica
    enriquecimento básico. Tenta primeiro signals_enriched; se nao existir, usa load_signals + enriquecimento em runtime.
    Filtragem por entry_type deve ser feita no front.
    """
    prebuilt_path = _signals_enriched_prebuilt_path(source=source, market=market, rule=rule, provider=provider)
    prebuilt_mtime = prebuilt_path.stat().st_mtime if prebuilt_path.exists() else 0.0
    df_prebuilt, from_prebuilt = _try_load_signals_enriched_prebuilt(
        source=source, market=market, rule=rule, provider=provider, prebuilt_mtime=prebuilt_mtime
    )
    if from_prebuilt and not df_prebuilt.empty:
        return df_prebuilt, True

    part_mtime = _get_signals_mtime(source, market, rule, provider)
    df = load_signals(
        source=source,
        market=market,
        rule=rule,
        provider=provider,
        signals_mtime=part_mtime or signals_mtime,
    )
    if df is None or df.empty:
        return pd.DataFrame(), False

    df = df.copy()
    df["rule"] = rule
    if "date" in df.columns and "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "race_time_iso" in df.columns and "race_ts" not in df.columns:
        df["race_ts"] = pd.to_datetime(df["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
    if "track_name" in df.columns and "race_time_iso" in df.columns:
        df["_key_track"] = df["track_name"].astype(str).map(normalize_track_name)
        df["_key_race"] = df["race_time_iso"].astype(str)

    if "_key_track" in df.columns and "_key_race" in df.columns:
        keys = list(zip(df["_key_track"].astype(str), df["_key_race"].astype(str)))
        keys_series = pd.Series(keys, index=df.index)

        if "num_runners" not in df.columns:
            num_index = _build_num_runners_index()
            df["num_runners"] = keys_series.map(lambda k: num_index.get(k, pd.NA))
            if df["num_runners"].dtype == "object":
                df["num_runners"] = pd.to_numeric(df["num_runners"], errors="coerce")

        if ("category" not in df.columns) or ("category_token" not in df.columns):
            cat_index = _build_category_index()
            cat_letter = {k: (v or {}).get("letter", "") for k, v in cat_index.items()}
            cat_token = {k: (v or {}).get("token", "") for k, v in cat_index.items()}
            df["category"] = keys_series.map(lambda k: cat_letter.get(k, "")).fillna("").astype(str)
            df["category_token"] = keys_series.map(lambda k: cat_token.get(k, "")).fillna("").astype(str)

    return df, False


def main() -> None:
    st.set_page_config(page_title="Sinais LAY/BACK - Cavalos", layout="wide")
    st.title("Sinais LAY/BACK - Estratégias Cavalos")

    small_width = 360
    small_height = 180

    # CSS global: limita altura dos multiselects (chips e lista) com rolagem
    st.markdown(
        """
        <style>
        /* Altura máx. do componente inteiro */
        div[data-testid="stMultiSelect"] {
            max-height: 200px;
            overflow-y: auto;
        }
        /* Altura máx. da área de chips selecionados */
        div[data-baseweb="tag-list"] {
            max-height: 90px;
            overflow-y: auto;
        }
        /* Altura máx. da lista suspensa de opções */
        div[data-baseweb="popover"] ul {
            max-height: 180px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Seletores principais (regra, fonte, mercado, entrada, provedor)
    # Ordem pedida: Regra -> Provedor -> Fonte -> Mercado -> Entrada
    col_rule, col_prov, col_src, col_mkt, col_entry = st.columns([1.6, 1.0, 1.2, 1.0, 1.2])

    rule_label_pairs = [(k, v) for k, v in HORSE_RULE_LABELS.items()]
    rule_labels = [label for _, label in rule_label_pairs]

    if "pending_strategy_import" not in st.session_state:
        st.session_state["pending_strategy_import"] = None
    if "import_feedback" not in st.session_state:
        st.session_state["import_feedback"] = None
    if "import_uploader_nonce" not in st.session_state:
        st.session_state["import_uploader_nonce"] = 0
    if "last_import_hash" not in st.session_state:
        st.session_state["last_import_hash"] = None
    if "consolidated_strategies" not in st.session_state:
        st.session_state["consolidated_strategies"] = []
    if "visualize_mode" not in st.session_state:
        st.session_state["visualize_mode"] = "Consolidado (Tudo)"
    if "last_consolidated_import_hash" not in st.session_state:
        st.session_state["last_consolidated_import_hash"] = None
    if "consolidated_import_hashes" not in st.session_state:
        st.session_state["consolidated_import_hashes"] = set()
    if "pending_clear_consolidated" not in st.session_state:
        st.session_state["pending_clear_consolidated"] = False
    if "pending_restore_filters" not in st.session_state:
        st.session_state["pending_restore_filters"] = False
    if "consolidated_uploader_nonce" not in st.session_state:
        st.session_state["consolidated_uploader_nonce"] = 0
    if "applied_strategy" not in st.session_state:
        st.session_state["applied_strategy"] = None
    if "pending_remove_consolidated" not in st.session_state:
        st.session_state["pending_remove_consolidated"] = None
    if "mf_bets_df" not in st.session_state:
        st.session_state["mf_bets_df"] = pd.DataFrame()
    if "mf_market_pl_df" not in st.session_state:
        st.session_state["mf_market_pl_df"] = pd.DataFrame()
    if "mf_loaded" not in st.session_state:
        st.session_state["mf_loaded"] = False
    if "last_mf_statement_hash" not in st.session_state:
        st.session_state["last_mf_statement_hash"] = None
    if "mf_commission_total" not in st.session_state:
        st.session_state["mf_commission_total"] = 0.0
    if "show_internal_signals_table" not in st.session_state:
        st.session_state["show_internal_signals_table"] = False
    if "import_in_progress" not in st.session_state:
        st.session_state["import_in_progress"] = False

    def _reset_rule_dependent_state() -> None:
        keys_to_clear = [
            "date_start_input",
            "date_end_input",
            "date_range_slider",
            "dates_range",
            "dates_ms",
            "tracks_ms",
            "num_runners_ms",
            "num_runners_bucket_ms",
            "cats_ms",
            "subcats_ms",
            "sel_num_runners",
            "sel_num_runners_bucket",
            "sel_cats",
            "sel_subcats",
            "bsp_low",
            "bsp_high",
            "bsp_slider",
            "weekdays_ms",
            "hour_bucket_ms",
            "forecast_rank_ms",
            "value_ratio_min",
            "value_ratio_max",
            "only_value_bets",
            "leader_min",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

    if st.session_state.get("pending_clear_consolidated"):
        st.session_state["consolidated_strategies"] = []
        st.session_state["visualize_mode"] = "Consolidado (Tudo)"
        st.session_state["applied_strategy"] = None
        st.session_state.pop("consolidated_union_result", None)
        _reset_rule_dependent_state()
        st.session_state["import_uploader_nonce"] = st.session_state.get("import_uploader_nonce", 0) + 1
        st.session_state["consolidated_uploader_nonce"] = st.session_state.get("consolidated_uploader_nonce", 0) + 1
        st.session_state["last_import_hash"] = None
        st.session_state["last_consolidated_import_hash"] = None
        st.session_state["consolidated_import_hashes"] = set()
        st.session_state["pending_strategy_import"] = None
        st.session_state["pending_clear_consolidated"] = False
        st.session_state["import_feedback"] = "Consolidadas limpas"

    if st.session_state.get("pending_restore_filters"):
        _reset_rule_dependent_state()
        st.session_state["applied_strategy"] = None
        st.session_state["import_uploader_nonce"] = st.session_state.get("import_uploader_nonce", 0) + 1
        st.session_state["last_import_hash"] = None
        st.session_state["pending_strategy_import"] = None
        st.session_state["pending_restore_filters"] = False
        st.session_state["import_feedback"] = "Filtros restaurados"

    remove_key = st.session_state.get("pending_remove_consolidated")
    if remove_key is not None:
        L = st.session_state.get("consolidated_strategies", [])
        idx = remove_key if isinstance(remove_key, int) else None
        if idx is not None and 0 <= idx < len(L):
            del st.session_state["consolidated_strategies"][idx]
        st.session_state.pop("consolidated_union_result", None)
        st.session_state["pending_remove_consolidated"] = None
        if not st.session_state.get("consolidated_strategies"):
            st.session_state["visualize_mode"] = "Consolidado (Tudo)"
            _reset_rule_dependent_state()
            st.session_state["consolidated_uploader_nonce"] = st.session_state.get("consolidated_uploader_nonce", 0) + 1
            st.session_state["last_consolidated_import_hash"] = None
            st.session_state["consolidated_import_hashes"] = set()
        st.rerun()

    pending_list = st.session_state.get("pending_strategy_import")
    apply_idx = st.session_state.pop("apply_strategy_index", None)
    if apply_idx is not None and isinstance(pending_list, list) and 0 <= apply_idx < len(pending_list):
        strategy_dict = pending_list[apply_idx]
        _apply_strategy_to_state(strategy_dict)
        keep_keys = frozenset(["strategy_name", "created_at", "import_filename", "rule_select_label", "source_select_label", "provider", "market", "entry_type"]) | frozenset(CORE_STATE_KEYS) | frozenset(RULE_EXTRA_KEYS.get(_rule_label_to_slug(strategy_dict.get("rule_select_label", "")), []))
        st.session_state["applied_strategy"] = {k: v for k, v in strategy_dict.items() if k in keep_keys}
        st.session_state["pending_strategy_import"] = None
        st.session_state["import_feedback"] = "Estrategia aplicada com sucesso"
        st.session_state["import_uploader_nonce"] = st.session_state.get("import_uploader_nonce", 0) + 1
        st.rerun()
    if isinstance(pending_list, list) and len(pending_list) == 1:
        strategy_dict = pending_list[0]
        _apply_strategy_to_state(strategy_dict)
        keep_keys = frozenset(["strategy_name", "created_at", "import_filename", "rule_select_label", "source_select_label", "provider", "market", "entry_type"]) | frozenset(CORE_STATE_KEYS) | frozenset(RULE_EXTRA_KEYS.get(_rule_label_to_slug(strategy_dict.get("rule_select_label", "")), []))
        st.session_state["applied_strategy"] = {k: v for k, v in strategy_dict.items() if k in keep_keys}
        st.session_state["pending_strategy_import"] = None
        st.session_state["import_feedback"] = "Estrategia aplicada com sucesso"
        st.session_state["import_uploader_nonce"] = st.session_state.get("import_uploader_nonce", 0) + 1
        st.rerun()

    with col_rule:
        if "horse_rule_select_label" not in st.session_state:
            st.session_state["horse_rule_select_label"] = rule_labels[0]

        def _on_rule_change() -> None:
            _reset_rule_dependent_state()

        selected_rule_label = st.selectbox(
            "Regra",
            rule_labels,
            key="horse_rule_select_label",
            on_change=_on_rule_change,
        )
        rule = HORSE_RULE_LABELS_INV.get(selected_rule_label, "terceiro_queda50")

    with col_prov:
        if rule == "forecast_odds":
            provider = "timeform"
            st.caption("timeform (fixo)")
        else:
            prov_options = ["timeform", "sportinglife"]
            idx = 1 if st.session_state.get("provider") == "sportinglife" else 0
            provider = st.selectbox(
                "Provedor",
                prov_options,
                index=idx,
                key="provider",
            )

    with col_src:
        source_options = ["top3", "forecast"]
        source_label_options = ["Top 3", "Forecast"]
        if rule == "forecast_odds":
            source = "forecast"
            source_label = "Forecast"
            st.caption("Forecast (fixo)")
        else:
            if "horse_source_select_label" not in st.session_state:
                st.session_state["horse_source_select_label"] = source_label_options[0]
            selected_source_label = st.selectbox(
                "Fonte de dados",
                source_label_options,
                key="horse_source_select_label",
            )
            source = "top3" if selected_source_label == "Top 3" else "forecast"
            source_label = selected_source_label

    with col_mkt:
        mkt_idx = 1 if st.session_state.get("market") == "place" else 0
        market = st.selectbox("Mercado", ["win", "place"], index=mkt_idx, key="market")

    with col_entry:
        entry_opt_labels = ["ambos", ENTRY_TYPE_LABELS["back"], ENTRY_TYPE_LABELS["lay"]]
        entry_type_from_state = st.session_state.get("entry_type", "both")
        entry_idx = {"both": 0, "back": 1, "lay": 2}.get(entry_type_from_state, 0)
        entry_label = st.selectbox("Tipo de entrada", entry_opt_labels, index=entry_idx)
        if entry_label == "ambos":
            entry_type = "both"
        else:
            entry_type = "back" if entry_label == ENTRY_TYPE_LABELS["back"] else "lay"
        st.session_state["entry_type"] = entry_type

    st.caption(f"Regra: {selected_rule_label} · Fonte: {source_label} · Mercado: {market} · Entrada: {entry_label} · Provedor: {provider}")

    signals_mtime = _get_signals_mtime(source, market, rule, provider)
    df_early, enriched_prebuilt = load_signals_enriched(
        source=source,
        market=market,
        rule=rule,
        provider=provider,
        signals_mtime=signals_mtime,
    )
    if df_early.empty and not st.session_state.get("mf_loaded"):
        st.session_state["_df"] = pd.DataFrame()
        st.session_state["_df_filtered"] = pd.DataFrame()
        st.session_state["_filt"] = pd.DataFrame()
        st.session_state["_enriched_prebuilt"] = False
        export_pnl, export_roi = None, None
    else:
        if not df_early.empty and "is_green" in df_early.columns:
            df_early["is_green"] = df_early["is_green"].fillna(False).astype(bool)
        if not df_early.empty and "date" in df_early.columns:
            raw_date_values = df_early["date"].dropna().astype(str).unique().tolist()
            parsed_dates = []
            for date_str in raw_date_values:
                parsed = pd.to_datetime(date_str, errors="coerce")
                if pd.notna(parsed):
                    parsed_dates.append((date_str, parsed.date()))
            parsed_dates.sort(key=lambda item: item[1])
            if parsed_dates:
                min_date, max_date = parsed_dates[0][1], parsed_dates[-1][1]
                if "date_start_input" not in st.session_state:
                    st.session_state["date_start_input"] = min_date
                if "date_end_input" not in st.session_state:
                    st.session_state["date_end_input"] = max_date
        dates_in_range = _get_dates_in_range_from_state(df_early, st.session_state) if not df_early.empty else []
        df_filtered_early = df_early[df_early["date"].isin(dates_in_range)].copy() if dates_in_range else (df_early.iloc[0:0].copy() if not df_early.empty else pd.DataFrame())
        if "total_matched_volume" not in df_filtered_early.columns:
            df_filtered_early["total_matched_volume"] = pd.NA
        df_filtered_early["total_matched_volume"] = pd.to_numeric(df_filtered_early["total_matched_volume"], errors="coerce")

        is_consolidated = len(st.session_state.get("consolidated_strategies", [])) > 0
        if is_consolidated:
            st.session_state["applied_strategy"] = None

        if is_consolidated and st.session_state.get("consolidated_strategies"):
            consolidated_strategies = st.session_state["consolidated_strategies"]
            _cached = st.session_state.get("consolidated_union_result")
            if _cached is not None and len(_cached) == 4:
                df_union_early, _tb, _ta, _ov = _cached
            else:
                df_union_early, _tb, _ta, _ov = _build_consolidated_df_by_groups(consolidated_strategies)
                st.session_state["consolidated_union_result"] = (df_union_early, _tb, _ta, _ov)
            viz_mode = st.session_state.get("visualize_mode", "Consolidado (Tudo)")
            if viz_mode == "Consolidado (So BACK)" and not df_union_early.empty and "entry_type" in df_union_early.columns:
                filt_early = df_union_early[df_union_early["entry_type"] == "back"].copy()
            elif viz_mode == "Consolidado (So LAY)" and not df_union_early.empty and "entry_type" in df_union_early.columns:
                filt_early = df_union_early[df_union_early["entry_type"] == "lay"].copy()
            elif isinstance(viz_mode, str) and viz_mode.startswith("Estrategia:"):
                chosen = next((s for s in consolidated_strategies if _visualize_label(s) == viz_mode), None)
                if chosen is not None:
                    try:
                        source_g, market_g, rule_g, provider_g = get_group_key(chosen)
                        signals_mtime_g = _get_signals_mtime(source_g, market_g, rule_g, provider_g)
                        df_group, _ = load_signals_enriched(source=source_g, market=market_g, rule=rule_g, provider=provider_g, signals_mtime=signals_mtime_g)
                        mask_i = _build_mask_from_strategy(df_group, chosen)
                        df_i = df_group[mask_i].copy()
                        df_i["dedup_key"] = _make_dedup_key(df_i)
                        df_i = df_i.drop_duplicates(subset=["dedup_key"])
                        filt_early = df_i
                    except Exception:
                        filt_early = df_union_early.copy()
                else:
                    filt_early = df_union_early.copy()
            else:
                filt_early = df_union_early.copy()
            df_filtered_early = df_union_early
        else:
            filt_early = _apply_filters_to_df_filtered(df_filtered_early, st.session_state, rule, entry_type)

        st.session_state["_df"] = df_early
        st.session_state["_df_filtered"] = df_filtered_early
        st.session_state["_filt"] = filt_early
        st.session_state["_enriched_prebuilt"] = enriched_prebuilt
        base_amount_early = float(st.session_state.get("base_amount", 1.0))
        export_pnl, export_roi = compute_export_pnl_roi(filt_early, entry_type, base_amount_early)
        if is_consolidated:
            export_pnl, export_roi = None, None

    import_feedback_msg = st.session_state.pop("import_feedback", None)
    is_consolidated = len(st.session_state.get("consolidated_strategies", [])) > 0

    col_restore, col_exp, col_imp, col_imp_cons, col_mf, col_clear = st.columns(6)
    with col_restore:
        if st.button("Restaurar filtros", key="restore_filters_btn", disabled=is_consolidated):
            st.session_state["pending_restore_filters"] = True
            st.rerun()
        if import_feedback_msg:
            st.caption(import_feedback_msg)
    with col_exp:
        snapshot = get_current_strategy_snapshot(
            selected_rule_label, source_label, market, entry_type, provider,
            pnl=export_pnl, roi=export_roi,
        )
        csv_bytes = strategy_to_csv_bytes(snapshot)
        st.download_button(
            "Exportar estrategia (CSV)",
            data=csv_bytes,
            file_name="estrategia_cavalos.csv",
            mime="text/csv",
            key="export_strategy_btn",
            disabled=is_consolidated,
        )
        if is_consolidated:
            st.caption("Consolidado ativo: use 'Exportar consolidadas (CSV)'.")
            consolidated_list = st.session_state.get("consolidated_strategies", [])
            csv_cons_bytes = strategies_list_to_csv_bytes(consolidated_list)
            st.download_button(
                "Exportar consolidadas (CSV)",
                data=csv_cons_bytes,
                file_name="consolidado_estrategias_cavalos.csv",
                mime="text/csv",
                key="export_consolidated_btn",
            )
    with col_imp:
        if is_consolidated:
            st.session_state["import_uploader_nonce"] = st.session_state.get("import_uploader_nonce", 0) + 1
            st.session_state["last_import_hash"] = None
            st.session_state["import_in_progress"] = False
            st.session_state["pending_strategy_import"] = None
            _strategy_uploader_key = f"import_strategy_csv_{st.session_state['import_uploader_nonce']}"
            st.file_uploader("Importar estrategia (CSV)", type=["csv"], key=_strategy_uploader_key, disabled=True)
            st.warning("Consolidado ativo: limpe as consolidadas para aplicar estrategia unica.")
        else:
            _strategy_uploader_key = f"import_strategy_csv_{st.session_state.get('import_uploader_nonce', 0)}"
            uploaded_single = st.file_uploader("Importar estrategia (CSV)", type=["csv"], key=_strategy_uploader_key)
            if uploaded_single is None:
                st.session_state["last_import_hash"] = None
                st.session_state["import_in_progress"] = False
            else:
                data = uploaded_single.getvalue()
                filename = uploaded_single.name or ""
                h = _hash_uploaded_file(io.BytesIO(data))
                last_h = st.session_state.get("last_import_hash")
                in_progress = st.session_state.get("import_in_progress", False)
                if last_h != h or not in_progress:
                    if _is_likely_marketfeeder_statement_csv(data):
                        st.session_state["import_feedback"] = "Arquivo parece statement MarketFeeder. Use o uploader de statement."
                    elif _is_likely_statement_csv(data):
                        st.session_state["import_feedback"] = "Arquivo parece statement; use apenas CSV de estrategia."
                    else:
                        strategies = parse_strategies_csv(io.BytesIO(data))
                        if strategies:
                            for d in strategies:
                                d["import_filename"] = filename
                            st.session_state["pending_strategy_import"] = strategies if len(strategies) > 1 else strategies[0]
                            st.session_state["last_import_hash"] = h
                            st.session_state["import_in_progress"] = True
                            st.rerun()
                        else:
                            st.session_state["import_feedback"] = "Nenhuma estrategia valida no CSV."
                st.session_state["last_import_hash"] = h
    with col_imp_cons:
        _consolidated_uploader_key = f"import_consolidated_csv_{st.session_state.get('consolidated_uploader_nonce', 0)}"
        uploaded_consolidated = st.file_uploader(
            "Importar consolidadas (CSV)", type=["csv"], key=_consolidated_uploader_key, accept_multiple_files=True
        )
        uploaded_files = uploaded_consolidated if isinstance(uploaded_consolidated, list) else ([uploaded_consolidated] if uploaded_consolidated else [])
        if uploaded_files:
            hashes = st.session_state.get("consolidated_import_hashes", set())
            any_new = False
            skipped_mf = False
            for uf in uploaded_files:
                data = uf.getvalue()
                if _is_likely_marketfeeder_statement_csv(data):
                    skipped_mf = True
                    continue
                h = _hash_uploaded_file(io.BytesIO(data))
                if h in hashes:
                    continue
                strategies = parse_strategies_csv(io.BytesIO(data))
                if strategies:
                    for d in strategies:
                        d["import_filename"] = uf.name
                    st.session_state["consolidated_strategies"] = st.session_state.get("consolidated_strategies", []) + strategies
                    hashes.add(h)
                    any_new = True
            st.session_state["consolidated_import_hashes"] = hashes
            if skipped_mf:
                st.session_state["import_feedback"] = "Um ou mais arquivos parecem statement MarketFeeder; use o uploader de statement."
            if any_new:
                st.session_state.pop("consolidated_union_result", None)
                st.rerun()
    with col_mf:
        mf_uploader_key = "import_marketfeeder_statement_csv"
        uploaded_mf = st.file_uploader("Importar statement MarketFeeder (CSV)", type=["csv"], key=mf_uploader_key)
        if uploaded_mf is not None:
            data_mf = uploaded_mf.getvalue()
            mf_hash = _hash_uploaded_file(io.BytesIO(data_mf))
            last_mf_hash = st.session_state.get("last_mf_statement_hash")
            if mf_hash != last_mf_hash:
                try:
                    df_mf_bets, df_mf_pl, total_comm = parse_marketfeeder_statement(data_mf)
                    st.session_state["mf_bets_df"] = df_mf_bets
                    st.session_state["mf_market_pl_df"] = df_mf_pl
                    st.session_state["mf_commission_total"] = total_comm
                    st.session_state["mf_loaded"] = True
                    st.session_state["last_mf_statement_hash"] = mf_hash
                    st.rerun()
                except Exception as e:
                    st.session_state["import_feedback"] = f"Erro ao ler statement: {e}"
    with col_clear:
        if st.button("Limpar consolidadas", key="clear_consolidated_btn"):
            st.session_state["pending_clear_consolidated"] = True
            st.rerun()

    pending_after_upload = st.session_state.get("pending_strategy_import")
    if isinstance(pending_after_upload, list) and len(pending_after_upload) > 1:
        st.caption("Varias estrategias no CSV: escolha qual aplicar.")
        options = [s.get("strategy_name") or f"Estrategia {i+1}" for i, s in enumerate(pending_after_upload)]
        sel_idx = st.selectbox("Estrategia para aplicar", range(len(options)), format_func=lambda i: options[i], key="import_strategy_select")
        if st.button("Aplicar selecionada", key="apply_import_btn"):
            st.session_state["apply_strategy_index"] = sel_idx
            st.rerun()

    if is_consolidated:
        n_strat = len(st.session_state.get("consolidated_strategies", []))
        consolidated_strategies_for_banner = st.session_state.get("consolidated_strategies", [])
        _union_result = _build_consolidated_df_by_groups(consolidated_strategies_for_banner)
        st.session_state["consolidated_union_result"] = _union_result
        _tb, _ta, _ov = _union_result[1], _union_result[2], _union_result[3]
        st.info(
            f"Consolidado ativo ({n_strat} estrategias). Para editar filtros ou aplicar uma estrategia unica, limpe as consolidadas. "
            f"Sinais: {_tb} | Unicos: {_ta} | Sobreposicao: {_ov}"
        )
        visualize_options = ["Consolidado (Tudo)", "Consolidado (So BACK)", "Consolidado (So LAY)"]
        for s in st.session_state.get("consolidated_strategies", []):
            visualize_options.append(_visualize_label(s))
        current_viz = st.session_state.get("visualize_mode", "Consolidado (Tudo)")
        if current_viz not in visualize_options:
            st.session_state["visualize_mode"] = "Consolidado (Tudo)"
        st.selectbox("Visualizar", visualize_options, key="visualize_mode")
        consolidated = st.session_state.get("consolidated_strategies", [])
        n = len(consolidated)
        with st.expander(f"Estrategias consolidadas ({n})", expanded=(n > 0)):
            for i, s in enumerate(consolidated):
                gk = get_group_key(s)
                st.caption(f"source/market/rule/provider: {gk[0]} / {gk[1]} / {gk[2]} / {gk[3]}")
                col_resumo, col_btn = st.columns([0.85, 0.15])
                with col_resumo:
                    st.markdown(_format_strategy_line(s))
                with col_btn:
                    st.button(
                        "Remover",
                        key=f"remove_cons_{i}",
                        on_click=lambda idx=i: st.session_state.update({"pending_remove_consolidated": idx}),
                    )
    elif st.session_state.get("applied_strategy") is not None:
        ap = st.session_state["applied_strategy"]
        with st.expander("Estrategia aplicada", expanded=True):
            st.markdown(_format_strategy_line(ap))

    df = st.session_state.get("_df", pd.DataFrame())
    with st.expander("Debug (carregamento de sinais)", expanded=False):
        st.write("Selecionado:", {"source": source, "market": market, "rule": rule, "provider": provider})
        st.write("df.shape:", df.shape if df is not None else 0)
        if df is not None and not df.empty:
            st.write("df.columns(sample):", list(df.columns)[:80])

    if df.empty and not st.session_state.get("mf_loaded"):
        st.info(
            "Nenhum sinal encontrado para a selecao. "
            "Gere antes com: python scripts/horses/generate_horse_signals.py --source {src} --market {mkt} --rule {rule} --entry_type {et}".format(
                src=source, mkt=market, rule=rule, et=entry_type
            )
        )
        return

    if st.session_state.get("mf_loaded"):
        mf_bets = st.session_state.get("mf_bets_df", pd.DataFrame())
        mf_pl = st.session_state.get("mf_market_pl_df", pd.DataFrame())
        st.warning("Statement importado: a tabela abaixo vem do MarketFeeder. A tabela interna vem do dataset de sinais.")
        if not mf_bets.empty:
            sinais = len(mf_bets)
            profit_series = pd.to_numeric(mf_bets["profit"], errors="coerce").fillna(0)
            stake_series = pd.to_numeric(mf_bets["stake"], errors="coerce").fillna(0)
            greens = int((profit_series > 0).sum())
            reds = int((profit_series <= 0).sum())
            pnl_bruto = float(profit_series.sum())
            commission = float(st.session_state.get("mf_commission_total", 0.0))
            pnl = pnl_bruto + commission
            total_stake = float(stake_series.sum())
            roi = (pnl / total_stake) if total_stake > 0 else 0.0
            drawdown = _calc_drawdown_series(profit_series)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Sinais (statement)", sinais)
            with c2:
                st.metric("Greens", greens)
            with c3:
                st.metric("Reds", reds)
            with c4:
                st.metric("PnL (liquido)", f"{pnl:.2f}")
            with c5:
                st.metric("ROI (liquido)", f"{roi:.2%}")
            with c6:
                st.metric("Drawdown", f"{drawdown:.2f}")
            if commission != 0:
                st.caption(f"Comissao descontada: {commission:.2f}. PnL bruto apostas: {pnl_bruto:.2f}.")
            st.subheader("Resultados (statement)")
            disp_cols = ["placed_dt", "track_clean", "off_time", "grade_distance", "selection_trap", "selection_name", "stake", "price", "profit", "trigger"]
            avail = [c for c in disp_cols if c in mf_bets.columns]
            st.dataframe(mf_bets[avail].rename(columns={"track_clean": "track"}), use_container_width=True)
        if not mf_pl.empty:
            with st.expander("Market P/L (statement)", expanded=False):
                st.dataframe(mf_pl, use_container_width=True)
        st.checkbox(
            "Exibir tabela interna (sinais)",
            key="show_internal_signals_table",
            help="Quando ativo, exibe a tabela de resultados gerada a partir do dataset de sinais.",
        )
        if st.button("Limpar statement", key="clear_mf_statement_btn"):
            st.session_state["mf_bets_df"] = pd.DataFrame()
            st.session_state["mf_market_pl_df"] = pd.DataFrame()
            st.session_state["mf_commission_total"] = 0.0
            st.session_state["mf_loaded"] = False
            st.session_state["last_mf_statement_hash"] = None
            st.rerun()
        if df.empty:
            return

    # Check rapido enriched: categoria, race_ts, num_runners (remover apos validar port)
    def _pct_empty(ser: pd.Series) -> float:
        if ser is None or ser.empty:
            return 0.0
        if pd.api.types.is_datetime64_any_dtype(ser):
            return float(ser.isna().mean() * 100.0)
        s = ser.astype(str).str.strip()
        return float(((s == "") | s.isna()).mean() * 100.0)
    cat_pct = _pct_empty(df["category"]) if "category" in df.columns else float("nan")
    race_ts_pct = _pct_empty(df["race_ts"]) if "race_ts" in df.columns else float("nan")
    nr_pct = float(df["num_runners"].isna().mean() * 100.0) if "num_runners" in df.columns else float("nan")
    enriched_prebuilt = st.session_state.get("_enriched_prebuilt", False)
    enriched_source = "enriched prebuilt" if enriched_prebuilt else "enriched runtime"
    st.caption(
        f"[debug] Carregado: {enriched_source} | category vazio: {cat_pct:.1f}% | race_ts NaT: {race_ts_pct:.1f}% | num_runners vazio: {nr_pct:.1f}%"
    )

    df_filtered = st.session_state.get("_df_filtered", pd.DataFrame()).copy()
    if not df_filtered.empty and entry_type != "both":
        df_filtered = df_filtered[df_filtered["entry_type"] == entry_type]
    if "total_matched_volume" not in df_filtered.columns:
        df_filtered["total_matched_volume"] = pd.NA
    df_filtered["total_matched_volume"] = pd.to_numeric(df_filtered["total_matched_volume"], errors="coerce")

    # Filtros principais: datas, pistas, BSP
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        raw_date_values = df["date"].dropna().astype(str).unique().tolist()
        parsed_dates = []
        for date_str in raw_date_values:
            parsed = pd.to_datetime(date_str, errors="coerce")
            if pd.notna(parsed):
                parsed_dates.append((date_str, parsed.date()))
        parsed_dates.sort(key=lambda item: item[1])

        date_start_key = "date_start_input"
        date_end_key = "date_end_input"

        if parsed_dates:
            min_date = parsed_dates[0][1]
            max_date = parsed_dates[-1][1]

            def _normalize_date_value(value: object, fallback: datetime.date) -> datetime.date:
                if isinstance(value, pd.Timestamp):
                    return value.to_pydatetime().date()
                if isinstance(value, datetime.datetime):
                    return value.date()
                if isinstance(value, datetime.date):
                    return value
                if isinstance(value, str):
                    parsed_value = pd.to_datetime(value, errors="coerce")
                    if pd.notna(parsed_value):
                        return parsed_value.date()
                return fallback

            stored_start = _normalize_date_value(st.session_state.get(date_start_key, min_date), min_date)
            stored_end = _normalize_date_value(st.session_state.get(date_end_key, max_date), max_date)

            sanitized_start = max(min_date, min(stored_start, max_date))
            sanitized_end = max(min_date, min(stored_end, max_date))
            if sanitized_start > sanitized_end:
                sanitized_end = sanitized_start

            if st.session_state.get(date_start_key) != sanitized_start:
                st.session_state[date_start_key] = sanitized_start
            if st.session_state.get(date_end_key) != sanitized_end:
                st.session_state[date_end_key] = sanitized_end

            mode_options = ["Calendário", "Barra"]
            default_mode = st.session_state.get("date_mode", "Calendário")
            active_mode = st.radio(
                "Modo de seleção de datas",
                mode_options,
                horizontal=True,
                index=0 if default_mode != "Barra" else 1,
                key="date_mode_selector",
                disabled=is_consolidated,
            )
            st.session_state["date_mode"] = active_mode

            start_col, end_col = st.columns(2)
            with start_col:
                start_selected = st.date_input(
                    "Data inicial",
                    min_value=min_date,
                    max_value=max_date,
                    key=date_start_key,
                    disabled=(active_mode == "Barra") or is_consolidated,
                )
            with end_col:
                end_selected = st.date_input(
                    "Data final",
                    min_value=min_date,
                    max_value=max_date,
                    key=date_end_key,
                    disabled=(active_mode == "Barra") or is_consolidated,
                )

            cal_range_start = min(start_selected, end_selected)
            cal_range_end = max(start_selected, end_selected)

            slider_key = "date_range_slider"
            slider_min_dt = datetime.datetime.combine(min_date, datetime.time.min)
            slider_max_dt = datetime.datetime.combine(max_date, datetime.time.min)
            default_slider_value = (
                datetime.datetime.combine(sanitized_start, datetime.time.min),
                datetime.datetime.combine(sanitized_end, datetime.time.min),
            )

            current_slider_value = st.session_state.get(slider_key, default_slider_value)
            if not isinstance(current_slider_value, (tuple, list)) or len(current_slider_value) != 2:
                current_slider_value = default_slider_value
            else:
                start_norm = current_slider_value[0]
                end_norm = current_slider_value[1]
                if isinstance(start_norm, pd.Timestamp):
                    start_norm = start_norm.to_pydatetime()
                if isinstance(end_norm, pd.Timestamp):
                    end_norm = end_norm.to_pydatetime()
                if isinstance(start_norm, datetime.date) and not isinstance(start_norm, datetime.datetime):
                    start_norm = datetime.datetime.combine(start_norm, datetime.time.min)
                if isinstance(end_norm, datetime.date) and not isinstance(end_norm, datetime.datetime):
                    end_norm = datetime.datetime.combine(end_norm, datetime.time.min)
                if not isinstance(start_norm, datetime.datetime):
                    start_norm = default_slider_value[0]
                if not isinstance(end_norm, datetime.datetime):
                    end_norm = default_slider_value[1]
                start_norm = max(slider_min_dt, min(start_norm, slider_max_dt))
                end_norm = max(slider_min_dt, min(end_norm, slider_max_dt))
                if start_norm > end_norm:
                    end_norm = start_norm
                current_slider_value = (start_norm, end_norm)
            st.session_state[slider_key] = current_slider_value

            slider_start_dt, slider_end_dt = st.slider(
                "Intervalo de datas (barra)",
                min_value=slider_min_dt,
                max_value=slider_max_dt,
                format="YYYY-MM-DD",
                key=slider_key,
                disabled=(active_mode == "Calendário") or is_consolidated,
            )
            slider_start_date = slider_start_dt.date()
            slider_end_date = slider_end_dt.date()

            bar_range_start = slider_start_date
            bar_range_end = slider_end_date

            if active_mode == "Barra":
                range_start, range_end = bar_range_start, bar_range_end
            else:
                range_start, range_end = cal_range_start, cal_range_end

            range_start = max(min_date, range_start)
            range_end = min(max_date, range_end)
            if range_start > range_end:
                range_end = range_start

            dates_in_range = [d for d, parsed in parsed_dates if range_start <= parsed <= range_end]

            if DEBUG_DATES:
                eff = st.session_state.get("_debug_dates_effective", {})
                ui_start = st.session_state.get("date_start_input", "")
                ui_end = st.session_state.get("date_end_input", "")
                ui_slider = st.session_state.get("date_range_slider", "")
                st.caption(
                    f"[debug datas] Filtro early: date_mode={eff.get('date_mode', '?')} start={eff.get('start', '?')} end={eff.get('end', '?')} | "
                    f"UI: date_start={ui_start} date_end={ui_end} date_mode={active_mode} slider={ui_slider}"
                )
        else:
            range_start = range_end = None
            dates_in_range = sorted(raw_date_values)

        df_filtered = df[df["date"].isin(dates_in_range)].copy() if dates_in_range else df.iloc[0:0].copy()

    with col_f2:
        tracks = sorted(df_filtered["track_name"].dropna().unique().tolist())
        tb1, tb2, _ = st.columns([1, 1, 2])
        with tb1:
            st.button(
                "Todos",
                key="tracks_all",
                on_click=lambda: st.session_state.update({"tracks_ms": list(tracks)}),
            )
        with tb2:
            st.button(
                "Limpar",
                key="tracks_none",
                on_click=lambda: st.session_state.update({"tracks_ms": []}),
            )
        existing_tracks = st.session_state.get("tracks_ms")
        if existing_tracks is None:
            sanitized_tracks = list(tracks)
        else:
            sanitized_tracks = [t for t in existing_tracks if t in tracks]
            if not sanitized_tracks and existing_tracks:
                sanitized_tracks = list(tracks)
        st.session_state["tracks_ms"] = sanitized_tracks
        # Limita a altura da multiselect de pistas e adiciona rolagem
        st.markdown(
            """
            <style>
            /* Limita altura e rolagem apenas no multiselect de Pistas */
            #tracks-ms-container [data-testid="stMultiSelect"] > div {
                max-height: 260px;
                overflow-y: auto;
            }
            #tracks-ms-container [data-baseweb="tag-list"] {
                max-height: 220px;
                overflow-y: auto;
            }
            #tracks-ms-container [role="listbox"] {
                max-height: 220px;
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container():
            st.markdown('<div id="tracks-ms-container">', unsafe_allow_html=True)
            sel_tracks = st.multiselect("Pistas", tracks, default=sanitized_tracks, key="tracks_ms")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_f3:
        if entry_type == "lay":
            bsp_col = "lay_target_bsp"
            base_df_for_bsp = df_filtered[df_filtered["entry_type"] == "lay"]
        elif entry_type == "back":
            bsp_col = "back_target_bsp"
            base_df_for_bsp = df_filtered[df_filtered["entry_type"] == "back"]
        else:
            bsp_col = None
            base_df_for_bsp = df_filtered
        if entry_type == "both":
            if base_df_for_bsp.empty:
                bsp_min = 1.01
                bsp_max = 100.0
            else:
                combined_bsp = base_df_for_bsp[["lay_target_bsp", "back_target_bsp"]]
                bsp_min = float(combined_bsp.min().min())
                bsp_max = float(combined_bsp.max().max())
        else:
            if base_df_for_bsp.empty:
                bsp_min = 1.01
                bsp_max = 100.0
            else:
                bsp_min = float(base_df_for_bsp[bsp_col].min())
                bsp_max = float(base_df_for_bsp[bsp_col].max())
        if not math.isfinite(bsp_min):
            bsp_min = 1.01
        if not math.isfinite(bsp_max):
            bsp_max = max(bsp_min, 100.0)
        bsp_min = max(1.01, round(bsp_min, 2))
        bsp_max = max(bsp_min, round(bsp_max, 2))

        if "bsp_low" not in st.session_state or "bsp_high" not in st.session_state:
            st.session_state["bsp_low"] = float(bsp_min)
            st.session_state["bsp_high"] = float(bsp_max)
        else:
            st.session_state["bsp_low"] = max(bsp_min, min(bsp_max, float(st.session_state["bsp_low"])))
            st.session_state["bsp_high"] = max(bsp_min, min(bsp_max, float(st.session_state["bsp_high"])))
            if st.session_state["bsp_low"] > st.session_state["bsp_high"]:
                st.session_state["bsp_high"] = st.session_state["bsp_low"]

        if "bsp_slider" not in st.session_state:
            st.session_state["bsp_slider"] = (float(st.session_state["bsp_low"]), float(st.session_state["bsp_high"]))

        def _sync_bsp_from_slider() -> None:
            low, high = st.session_state["bsp_slider"]
            st.session_state["bsp_low"] = float(low)
            st.session_state["bsp_high"] = float(high)

        def _sync_bsp_low() -> None:
            low = max(bsp_min, min(bsp_max, float(st.session_state["bsp_low"])))
            high = float(st.session_state.get("bsp_high", bsp_max))
            if low > high:
                high = low
                st.session_state["bsp_high"] = high
            st.session_state["bsp_slider"] = (float(low), float(high))

        def _sync_bsp_high() -> None:
            high = max(bsp_min, min(bsp_max, float(st.session_state["bsp_high"])))
            low = float(st.session_state.get("bsp_low", bsp_min))
            if high < low:
                low = high
                st.session_state["bsp_low"] = low
            st.session_state["bsp_slider"] = (float(low), float(high))

        st.slider(
            "Faixa BSP alvo",
            min_value=float(bsp_min),
            max_value=float(bsp_max),
            value=(float(st.session_state["bsp_low"]), float(st.session_state["bsp_high"])),
            step=0.01,
            key="bsp_slider",
            on_change=_sync_bsp_from_slider,
        )
        c41, c42, _ = st.columns([1, 1, 2])
        with c41:
            st.number_input(
                "BSP minimo",
                min_value=float(bsp_min),
                max_value=float(bsp_max),
                value=float(st.session_state["bsp_low"]),
                step=0.01,
                format="%.2f",
                key="bsp_low",
                on_change=_sync_bsp_low,
                label_visibility="collapsed",
            )
            st.caption("BSP min.")
        with c42:
            st.number_input(
                "BSP maximo",
                min_value=float(bsp_min),
                max_value=float(bsp_max),
                value=float(st.session_state["bsp_high"]),
                step=0.01,
                format="%.2f",
                key="bsp_high",
                on_change=_sync_bsp_high,
                label_visibility="collapsed",
            )
            st.caption("BSP max.")

        # Volume total negociado mínimo: default = menor valor disponivel nos dados (como nos galgos: valor inicial que nao exclui nada)
        volume_key = "min_total_volume"
        if volume_key not in st.session_state:
            vol_series = df_filtered["total_matched_volume"].dropna()
            if len(vol_series):
                default_vol = max(0.0, float(vol_series.min()))
            else:
                default_vol = 0.0
            st.session_state[volume_key] = default_vol
        vcol, _ = st.columns([3, 7])
        with vcol:
            if rule != "forecast_odds":
                st.number_input(
                    "Volume total negociado mínimo",
                    min_value=0.0,
                    step=100.0,
                    format="%.0f",
                    key=volume_key,
                    help="Considera apenas corridas cuja soma de pptradedvol atinge o mínimo desejado.",
                    disabled=is_consolidated,
                )
            else:
                st.number_input(
                    "Volume total negociado mínimo",
                    min_value=0.0,
                    step=100.0,
                    format="%.0f",
                    key=volume_key,
                    help="Forecast Odds nao usa filtro de volume; campo desabilitado.",
                    disabled=True,
                )

    # Enriquecimento: num_runners fallback se faltar
    if "num_runners" not in df_filtered.columns:
        num_index = _build_num_runners_index()
        if not df_filtered.empty:
            df_filtered["_key_track"] = df_filtered["track_name"].astype(str).map(normalize_track_name)
            df_filtered["_key_race"] = df_filtered["race_time_iso"].astype(str)
            df_filtered["num_runners"] = df_filtered.apply(
                lambda r: num_index.get((str(r["_key_track"]), str(r["_key_race"])), pd.NA),
                axis=1,
            )

    # Aplicar filtro de volume (widget esta em col_f3)
    volume_key = "min_total_volume"
    if rule != "forecast_odds":
        min_total_volume = max(0.0, float(st.session_state.get(volume_key, 0.0)))
        volume_series_raw = pd.to_numeric(df_filtered.get("total_matched_volume", pd.Series(dtype=float)), errors="coerce")
        mask_volume = volume_series_raw.fillna(0.0)
        if not mask_volume.index.equals(df_filtered.index):
            mask_volume = mask_volume.reindex(df_filtered.index, fill_value=0.0)
        df_filtered = df_filtered[mask_volume >= min_total_volume]

    # Bucket de num_runners e enriquecimento de categoria (para opcoes da linha 2)
    df_filtered = df_filtered.copy()
    df_filtered["num_runners_bucket"] = _bucket_num_runners(df_filtered.get("num_runners"))
    if ("category" not in df_filtered.columns) or ("category_token" not in df_filtered.columns):
        cat_index = _build_category_index()
        if not df_filtered.empty:
            df_filtered["_key_track"] = df_filtered["track_name"].astype(str).map(normalize_track_name)
            df_filtered["_key_race"] = df_filtered["race_time_iso"].astype(str)
            df_filtered["category"] = df_filtered.apply(
                lambda r: (cat_index.get((str(r["_key_track"]), str(r["_key_race"])), {}) or {}).get("letter", ""),
                axis=1,
            )
            df_filtered["category_token"] = df_filtered.apply(
                lambda r: (cat_index.get((str(r["_key_track"]), str(r["_key_race"])), {}) or {}).get("token", ""),
                axis=1,
            )
    cat_letters = sorted([c for c in df_filtered["category"].dropna().unique().tolist() if isinstance(c, str) and c]) if "category" in df_filtered.columns and not df_filtered.empty else []
    sel_cats_for_sub = st.session_state.get("cats_ms") or (cat_letters if cat_letters else [])
    token_source = df_filtered[df_filtered["category"].isin(sel_cats_for_sub)] if sel_cats_for_sub and "category" in df_filtered.columns else df_filtered
    raw_tokens = [t for t in token_source["category_token"].dropna().astype(str).unique().tolist() if isinstance(t, str) and t] if "category_token" in df_filtered.columns else []

    def _sub_sort_key(tok: str) -> tuple:
        m = re.match(r"^([A-Z]+)(\d+)$", str(tok))
        if m:
            return (m.group(1), int(m.group(2)))
        m2 = re.match(r"^([A-Z]+)", str(tok))
        return ((m2.group(1) if m2 else str(tok)), 0)

    sub_tokens = sorted(raw_tokens, key=_sub_sort_key) if raw_tokens else []

    # Linha 2 de filtros (como no dashboard dos galgos): col_s1 | col_s2 | col_s3
    col_s1, col_s2, col_s3 = st.columns(3)
    wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
    with col_s1:
        st.caption("Dias da semana")
        if not df_filtered.empty and "race_time_iso" in df_filtered.columns:
            df_filtered["hour_bucket"] = _compute_hour_bucket_series(df_filtered["race_time_iso"])
            bucket_order = ["08:00-12:00", "12:01-16:00", "16:01-20:00", "20:01-23:59"]
            bucket_options = [b for b in bucket_order if b in df_filtered["hour_bucket"].dropna().unique().tolist()]
            tmp_ts = pd.to_datetime(df_filtered["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            wd_series = tmp_ts.dt.weekday.dropna().astype(int)
            wd_unique = sorted(wd_series.unique().tolist())
            weekday_options = [wd_names[w] for w in wd_unique if w in wd_names]
            if weekday_options:
                wb1, wb2, _ = st.columns([1, 1, 2])
                with wb1:
                    st.button("Todos", key="weekdays_all", on_click=lambda: st.session_state.update({"weekdays_ms": list(weekday_options)}), disabled=is_consolidated)
                with wb2:
                    st.button("Limpar", key="weekdays_none", on_click=lambda: st.session_state.update({"weekdays_ms": []}), disabled=is_consolidated)
                if "weekdays_ms" not in st.session_state:
                    st.session_state["weekdays_ms"] = list(weekday_options)
                else:
                    existing_wd = st.session_state["weekdays_ms"]
                    sanitized_wd = [w for w in existing_wd if w in weekday_options]
                    if not sanitized_wd and existing_wd:
                        sanitized_wd = list(weekday_options)
                    st.session_state["weekdays_ms"] = sanitized_wd
                st.multiselect("Dias da semana", weekday_options, key="weekdays_ms", label_visibility="collapsed", disabled=is_consolidated)
            else:
                st.session_state["weekdays_ms"] = []
            st.caption("Faixa de horário")
            if bucket_options:
                hb1, hb2, _ = st.columns([1, 1, 2])
                with hb1:
                    st.button("Todos", key="hour_buckets_all", on_click=lambda: st.session_state.update({"hour_bucket_ms": list(bucket_options)}), disabled=is_consolidated)
                with hb2:
                    st.button("Limpar", key="hour_buckets_none", on_click=lambda: st.session_state.update({"hour_bucket_ms": []}), disabled=is_consolidated)
                if "hour_bucket_ms" not in st.session_state:
                    st.session_state["hour_bucket_ms"] = list(bucket_options)
                else:
                    existing_hb = st.session_state["hour_bucket_ms"]
                    sanitized_hb = [h for h in existing_hb if h in bucket_options]
                    if not sanitized_hb and existing_hb:
                        sanitized_hb = list(bucket_options)
                    st.session_state["hour_bucket_ms"] = sanitized_hb
                st.multiselect("Faixa de horário", bucket_options, key="hour_bucket_ms", label_visibility="collapsed", disabled=is_consolidated)
            else:
                st.session_state["hour_bucket_ms"] = []
        else:
            st.caption("Faixa de horário")

    with col_s2:
        st.caption("Categorias")
        if cat_letters:
            btn_all, btn_clear, _ = st.columns([1, 1, 6])
            with btn_all:
                st.button("Todas", key="cats_all", on_click=lambda: st.session_state.update({"cats_ms": cat_letters}), disabled=is_consolidated)
            with btn_clear:
                st.button("Limpar", key="cats_none", on_click=lambda: st.session_state.update({"cats_ms": []}), disabled=is_consolidated)
            if "cats_ms" not in st.session_state:
                st.session_state["cats_ms"] = list(cat_letters)
            else:
                existing_cats = st.session_state["cats_ms"]
                sanitized_cats = [c for c in existing_cats if c in cat_letters]
                if not sanitized_cats and existing_cats:
                    sanitized_cats = list(cat_letters)
                st.session_state["cats_ms"] = sanitized_cats
            st.multiselect("Categoria (G/H/M/N/...)", cat_letters, key="cats_ms", label_visibility="collapsed", disabled=is_consolidated)
        st.caption("Subcategorias")
        if sub_tokens:
            sb1, sb2, _ = st.columns([1, 1, 2])
            with sb1:
                st.button("Todas", key="subcats_all", on_click=lambda: st.session_state.update({"subcats_ms": sub_tokens}), disabled=is_consolidated)
            with sb2:
                st.button("Limpar", key="subcats_none", on_click=lambda: st.session_state.update({"subcats_ms": []}), disabled=is_consolidated)
            if "subcats_ms" not in st.session_state:
                st.session_state["subcats_ms"] = list(sub_tokens)
            else:
                existing_sc = st.session_state["subcats_ms"]
                sanitized_sc = [t for t in existing_sc if t in sub_tokens]
                if not sanitized_sc and existing_sc:
                    sanitized_sc = list(sub_tokens)
                st.session_state["subcats_ms"] = sanitized_sc
            st.multiselect("Subcategorias (G1/HCP_CHS/MDN/...)", sub_tokens, key="subcats_ms", label_visibility="collapsed", disabled=is_consolidated)
        else:
            st.session_state["subcats_ms"] = []

    with col_s3:
        st.caption("Numero de corredores")
        nr_buckets = [b for b in NR_BUCKET_LABELS if b in df_filtered["num_runners_bucket"].dropna().unique().tolist()]
        if nr_buckets:
            btn_all, btn_clear, _ = st.columns([1, 1, 6])
            with btn_all:
                st.button("Todos", key="nr_bucket_all", on_click=lambda: st.session_state.update({"num_runners_bucket_ms": nr_buckets}), disabled=is_consolidated)
            with btn_clear:
                st.button("Limpar", key="nr_bucket_none", on_click=lambda: st.session_state.update({"num_runners_bucket_ms": []}), disabled=is_consolidated)
            if "num_runners_bucket_ms" not in st.session_state:
                st.session_state["num_runners_bucket_ms"] = nr_buckets.copy()
            else:
                existing_nr = st.session_state["num_runners_bucket_ms"]
                sanitized_nr = [v for v in existing_nr if v in nr_buckets]
                if not sanitized_nr and existing_nr:
                    sanitized_nr = nr_buckets.copy()
                st.session_state["num_runners_bucket_ms"] = sanitized_nr
            st.multiselect("Numero de corredores (faixas)", nr_buckets, key="num_runners_bucket_ms", label_visibility="collapsed", disabled=is_consolidated)
        sel_nr = [v for v in st.session_state.get("num_runners_bucket_ms", []) if v in nr_buckets] if nr_buckets else []
        st.session_state["sel_num_runners_bucket"] = sel_nr

    # Regra líder (apenas widget; aplicacao em _apply_filters_to_df_filtered)
    if rule == "lider_volume_total":
        col_l1, col_l2 = st.columns([1, 2])
        with col_l1:
            if "leader_min" not in st.session_state:
                st.session_state["leader_min"] = 50.0
            st.number_input(
                "Participacao do lider (%) min.",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["leader_min"]),
                step=1.0,
                format="%.0f",
                key="leader_min",
            )

    # Filtros especificos para regra forecast_odds (apenas widgets; aplicacao em _apply_filters_to_df_filtered)
    if rule == "forecast_odds":
        if "forecast_rank" in df_filtered.columns:
            df_filtered = df_filtered.copy()
            df_filtered["forecast_rank"] = pd.to_numeric(df_filtered["forecast_rank"], errors="coerce")
        if "value_ratio" in df_filtered.columns:
            df_filtered = df_filtered.copy()
            df_filtered["value_ratio"] = pd.to_numeric(df_filtered["value_ratio"], errors="coerce")
        if not df_filtered.empty and "forecast_rank" in df_filtered.columns:
            raw_ranks = df_filtered["forecast_rank"].dropna().unique().tolist()
            rank_vals = sorted(set(int(x) for x in raw_ranks if pd.notna(x) and isinstance(x, (int, float))))
            if not rank_vals:
                rank_vals = sorted(set(int(x) for x in raw_ranks if str(x).replace(".", "").isdigit()))
        else:
            rank_vals = []
        if rank_vals:
            if "forecast_rank_ms" not in st.session_state:
                st.session_state["forecast_rank_ms"] = [3, 4] if 3 in rank_vals or 4 in rank_vals else rank_vals[:2]
            existing_ranks = st.session_state.get("forecast_rank_ms") or []
            sanitized_ranks = [r for r in existing_ranks if r in rank_vals]
            if not sanitized_ranks and existing_ranks:
                sanitized_ranks = [3, 4] if (3 in rank_vals or 4 in rank_vals) else rank_vals[:2]
            st.session_state["forecast_rank_ms"] = sanitized_ranks
            st.caption("Forecast Odds: rank e value ratio")
            col_rank, _ = st.columns([2, 8])
            with col_rank:
                st.multiselect(
                    "Forecast rank",
                    rank_vals,
                    default=sanitized_ranks,
                    key="forecast_rank_ms",
                )
        vmin, vmax = 0.0, 3.0
        if "value_ratio" in df_filtered.columns and not df_filtered["value_ratio"].isna().all():
            vr_valid = df_filtered["value_ratio"].dropna()
            if not vr_valid.empty:
                vmin = float(vr_valid.min())
                vmax = float(vr_valid.max())
        if "value_ratio_min" not in st.session_state:
            st.session_state["value_ratio_min"] = vmin
        if "value_ratio_max" not in st.session_state:
            st.session_state["value_ratio_max"] = vmax
        vr_low = max(vmin, min(vmax, float(st.session_state.get("value_ratio_min", vmin))))
        vr_high = max(vmin, min(vmax, float(st.session_state.get("value_ratio_max", vmax))))
        if vr_low > vr_high:
            vr_high = vr_low
        st.session_state["value_ratio_min"] = vr_low
        st.session_state["value_ratio_max"] = vr_high
        col_vr, _ = st.columns([2, 8])
        with col_vr:
            st.number_input("Value ratio min", min_value=vmin, max_value=vmax, value=vr_low, step=0.05, key="value_ratio_min")
            st.number_input("Value ratio max", min_value=vmin, max_value=vmax, value=vr_high, step=0.05, key="value_ratio_max")
        if "only_value_bets" not in st.session_state:
            st.session_state["only_value_bets"] = False
        st.checkbox("Somente value bets (value_ratio >= 1.0)", key="only_value_bets")

    # Fonte unica de dados filtrados (Fase 2): _filt é construido no bloco early via _apply_filters_to_df_filtered
    filt = st.session_state.get("_filt", pd.DataFrame())

    # Eixo X
    x_axis_mode = st.radio(
        "Eixo X dos graficos de evolucao",
        ["Dia", "Bet"],
        index=0,
        horizontal=True,
        help="Altere entre datas ou sequencia de apostas",
    )

    def _render_weekday_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        if df_block.empty or "race_time_iso" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = base_amount
        plot = df_block.copy()
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        if plot.empty:
            return
        plot["date_only"] = plot["ts"].dt.date
        daily = (
            plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]]
            .sum()
            .sort_values("date_only")
        )
        daily["weekday"] = pd.to_datetime(daily["date_only"]).dt.weekday
        wd_order = [0, 1, 2, 3, 4, 5, 6]
        wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
        wd_order_names = [wd_names[w] for w in wd_order]
        by_wd = daily.groupby("weekday", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum()
        by_wd["weekday_name"] = by_wd["weekday"].map(wd_names)
        by_wd["weekday_name"] = pd.Categorical(by_wd["weekday_name"], categories=wd_order_names, ordered=True)
        by_wd["pnl_stake"] = by_wd["pnl_stake_fixed_10"] * scale_factor
        if entry_kind == "lay":
            by_wd["pnl_liab"] = by_wd["pnl_liability_fixed_10"] * scale_factor

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        bar_stake = (
            alt.Chart(by_wd)
            .mark_bar()
            .encode(
                x=alt.X("weekday_name:N", sort=wd_order_names, title=""),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por dia da semana (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                bar_liab = (
                    alt.Chart(by_wd)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("weekday_name:N", sort=wd_order_names, title=""),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                liab_chart = alt.layer(zero_line_liab, bar_liab)
                st.altair_chart(
                    alt.vconcat(
                        stake_chart.properties(title="Stake"),
                        liab_chart.properties(title="Liability"),
                    ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1),
                    use_container_width=True,
                )
            else:
                st.altair_chart(
                    stake_chart.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake"),
                    use_container_width=True,
                )

    def _render_hour_bucket_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por faixa horaria (mesmos buckets do filtro)."""
        if df_block.empty or "race_time_iso" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = base_amount
        plot = df_block.copy()
        plot["hour_bucket"] = _compute_hour_bucket_series(plot["race_time_iso"])
        plot = plot.dropna(subset=["hour_bucket"])
        if plot.empty:
            return
        plot["_pnl_stake"] = plot["pnl_stake_fixed_10"]
        plot["_pnl_liab"] = plot["pnl_liability_fixed_10"] if entry_kind == "lay" else pd.Series(0.0, index=plot.index)
        bucket_order = ["08:00-12:00", "12:01-16:00", "16:01-20:00", "20:01-23:59"]
        by_bucket = plot.groupby("hour_bucket", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        by_bucket = by_bucket[by_bucket["hour_bucket"].isin(bucket_order)]
        if by_bucket.empty:
            return
        by_bucket["hour_bucket"] = pd.Categorical(by_bucket["hour_bucket"], categories=bucket_order, ordered=True)
        by_bucket = by_bucket.sort_values("hour_bucket")
        by_bucket["pnl_stake"] = by_bucket["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_bucket["pnl_liab"] = by_bucket["_pnl_liab"] * scale_factor
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        bar_stake = (
            alt.Chart(by_bucket)
            .mark_bar()
            .encode(
                x=alt.X("hour_bucket:N", sort=bucket_order, title=""),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por faixa horaria (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                bar_liab = (
                    alt.Chart(by_bucket)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("hour_bucket:N", sort=bucket_order, title=""),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                st.altair_chart(
                    alt.vconcat(
                        stake_chart.properties(title="Stake"),
                        alt.layer(zero_line_liab, bar_liab).properties(title="Liability"),
                    ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1),
                    use_container_width=True,
                )
            else:
                st.altair_chart(stake_chart.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake"), use_container_width=True)

    def _render_num_runners_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por numero de corredores (bucket: 1-4, 5-8, etc.)."""
        if df_block.empty:
            return
        plot = df_block.copy()
        if "num_runners_bucket" not in plot.columns and "num_runners" in plot.columns:
            plot["num_runners_bucket"] = _bucket_num_runners(plot["num_runners"])
        if "num_runners_bucket" not in plot.columns:
            return
        plot = plot.dropna(subset=["num_runners_bucket"])
        if plot.empty:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = base_amount
        plot["_pnl_stake"] = pd.to_numeric(plot["pnl_stake_fixed_10"], errors="coerce").fillna(0.0)
        plot["_pnl_liab"] = pd.to_numeric(plot["pnl_liability_fixed_10"], errors="coerce").fillna(0.0) if entry_kind == "lay" else pd.Series(0.0, index=plot.index)
        by_nr = plot.groupby("num_runners_bucket", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        bucket_order = [b for b in NR_BUCKET_LABELS if b in by_nr["num_runners_bucket"].tolist()]
        if not bucket_order:
            return
        by_nr = by_nr[by_nr["num_runners_bucket"].isin(bucket_order)]
        by_nr["num_runners_bucket"] = pd.Categorical(by_nr["num_runners_bucket"], categories=bucket_order, ordered=True)
        by_nr = by_nr.sort_values("num_runners_bucket")
        by_nr["pnl_stake"] = by_nr["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_nr["pnl_liab"] = by_nr["_pnl_liab"] * scale_factor
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        bar_stake = (
            alt.Chart(by_nr)
            .mark_bar()
            .encode(
                x=alt.X("num_runners_bucket:N", sort=bucket_order, title=""),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por numero de corredores (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                bar_liab = (
                    alt.Chart(by_nr)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("num_runners_bucket:N", sort=bucket_order, title=""),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                st.altair_chart(
                    alt.vconcat(
                        stake_chart.properties(title="Stake"),
                        alt.layer(zero_line_liab, bar_liab).properties(title="Liability"),
                    ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1),
                    use_container_width=True,
                )
            else:
                st.altair_chart(stake_chart.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake"), use_container_width=True)

    def _render_forecast_rank_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por forecast_rank (quando regra e forecast_odds e ha coluna forecast_rank)."""
        if df_block.empty or "forecast_rank" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = base_amount
        plot = df_block.copy()
        plot["forecast_rank"] = pd.to_numeric(plot["forecast_rank"], errors="coerce")
        plot = plot.dropna(subset=["forecast_rank"])
        if plot.empty:
            return
        plot["_pnl_stake"] = plot["pnl_stake_fixed_10"]
        plot["_pnl_liab"] = plot["pnl_liability_fixed_10"] if entry_kind == "lay" else pd.Series(0.0, index=plot.index)
        by_rank = plot.groupby("forecast_rank", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        by_rank["forecast_rank"] = by_rank["forecast_rank"].astype(int)
        by_rank["rank_label"] = by_rank["forecast_rank"].astype(str)
        order_labels = sorted(by_rank["rank_label"].unique(), key=lambda x: int(x))
        if not order_labels:
            return
        by_rank["rank_label"] = pd.Categorical(by_rank["rank_label"], categories=order_labels, ordered=True)
        by_rank["pnl_stake"] = by_rank["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_rank["pnl_liab"] = by_rank["_pnl_liab"] * scale_factor
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        bar_stake = (
            alt.Chart(by_rank)
            .mark_bar()
            .encode(
                x=alt.X("rank_label:N", sort=order_labels, title=""),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por forecast rank (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                bar_liab = (
                    alt.Chart(by_rank)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("rank_label:N", sort=order_labels, title=""),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                st.altair_chart(
                    alt.vconcat(
                        stake_chart.properties(title="Stake"),
                        alt.layer(zero_line_liab, bar_liab).properties(title="Liability"),
                    ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1),
                    use_container_width=True,
                )
            else:
                st.altair_chart(stake_chart.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake"), use_container_width=True)

    def _render_monthly_table(df_block: pd.DataFrame, entry_kind: str, base_amount: float | None = None) -> None:
        if base_amount is None:
            base_amount = float(st.session_state.get("base_amount", 1.0))
        working = df_block.copy()
        if "race_ts" in working.columns:
            working["race_ts"] = pd.to_datetime(working["race_ts"], errors="coerce")
        elif "race_time_iso" in working.columns:
            working["race_ts"] = pd.to_datetime(working["race_time_iso"], errors="coerce")
        else:
            st.info("Sem dados para compor o relatório mensal (coluna de data/hora da corrida não disponível).")
            return
        working = working.dropna(subset=["race_ts"])
        if working.empty:
            st.info("Sem dados para compor o relatório mensal.")
            return

        working["month_period"] = working["race_ts"].dt.to_period("M")
        working["month_label"] = working["race_ts"].apply(_format_month_label)
        rows: list[dict[str, float | str]] = []
        for (period_val, label), grp in working.groupby(["month_period", "month_label"]):
            summary = _compute_summary_metrics(grp, entry_kind, base_amount)
            row: dict[str, float | str] = {
                "Mes/Ano": label,
                "_period": period_val,
                "Pistas": summary["tracks"],
                "Sinais": summary["signals"],
                "Greens": summary["greens"],
                "Reds": summary["reds"],
                "Media BSP Alvo": summary["avg_bsp"],
                "Assertividade": summary["accuracy"],
                "Base (Stake)": summary["base_stake"],
                "PnL Stake": summary["pnl_stake"],
                "ROI Stake": summary["roi_stake"],
                "Menor PnL acumulado": summary["min_pnl_stake"],
                "Drawdown máximo (Stake)": summary["drawdown_stake"],
            }
            if entry_kind == "lay":
                row.update(
                    {
                        "Stake (Liability)": summary.get("stake_liab", 0.0),
                        "PnL Liability": summary.get("pnl_liab", 0.0),
                        "ROI Liability": summary.get("roi_liab", 0.0),
                        "Menor PnL (Liability)": summary.get("min_pnl_liab", 0.0),
                        "Drawdown max (Liability)": summary.get("drawdown_liab", 0.0),
                    }
                )
            rows.append(row)

        monthly_df = pd.DataFrame(rows)
        month_order: list[str] = []
        if not monthly_df.empty:
            monthly_df = monthly_df.sort_values("_period").drop(columns=["_period"], errors="ignore")
            month_order = monthly_df["Mes/Ano"].tolist()
        with st.expander(f"Relatorio mensal ({entry_kind.upper()})", expanded=False):
            st.dataframe(monthly_df, use_container_width=True)
            if not working.empty and "pnl_stake_fixed_10" in working.columns:
                chart_data = working.copy()
                chart_data["date_only"] = chart_data["race_ts"].dt.date
                chart_data["_pnl_stake"] = pd.to_numeric(chart_data["pnl_stake_fixed_10"], errors="coerce").fillna(0.0)
                daily_raw = (
                    chart_data.groupby(["month_label", "month_period", "date_only"], as_index=False)[["_pnl_stake"]]
                    .sum()
                    .sort_values("date_only")
                )
                filled_frames: list[pd.DataFrame] = []
                for (mlabel, mperiod), grp in daily_raw.groupby(["month_label", "month_period"]):
                    try:
                        start_date = mperiod.start_time.date()
                        end_date = mperiod.end_time.date()
                    except Exception:
                        start_date = grp["date_only"].min()
                        end_date = grp["date_only"].max()
                    full_dates = pd.date_range(start_date, end_date, freq="D").date
                    base = pd.DataFrame({"date_only": full_dates})
                    merged = base.merge(grp[["date_only", "_pnl_stake"]], on="date_only", how="left")
                    merged["month_label"] = mlabel
                    merged["_pnl_stake"] = merged["_pnl_stake"].fillna(0)
                    filled_frames.append(merged)
                daily = pd.concat(filled_frames, ignore_index=True) if filled_frames else pd.DataFrame()
                daily = daily[daily["month_label"].notna()]
                if not daily.empty:
                    local_scale = base_amount
                    daily["cum_stake"] = daily.groupby("month_label")["_pnl_stake"].cumsum() * local_scale
                    month_sort = month_order or list(daily["month_label"].unique())
                    col_count = min(4, max(1, len(month_sort)))
                    zero_line = alt.Chart().mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    base_line = (
                        alt.Chart()
                        .mark_line(color="#85b4ff")
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%d/%m")),
                            y=alt.Y("cum_stake:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    month_chart = (
                        alt.layer(zero_line, base_line)
                        .facet(
                            facet=alt.Facet("month_label:N", sort=month_sort, title=""),
                            columns=col_count,
                            data=daily,
                        )
                        .resolve_scale(x="independent", y="independent")
                    )
                    st.markdown("**Evolucao mensal (PnL acumulado por dia)**")
                    st.altair_chart(month_chart.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    def render_block(title_suffix: str, df_block: pd.DataFrame, entry_kind: str) -> None:
        # Evita SettingWithCopy em atribuicoes posteriores
        df_block = df_block.copy()
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = base_amount
        summary = _compute_summary_metrics(df_block, entry_kind, base_amount)

        st.subheader(title_suffix)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Pistas", summary["tracks"])
        with c2:
            st.metric("Sinais", summary["signals"])
        with c3:
            st.metric("Greens", summary["greens"])
        with c4:
            st.metric("Reds", summary["reds"])
        with c5:
            st.metric("Media BSP Alvo", f"{summary['avg_bsp']:.2f}")
        with c6:
            st.metric("Assertividade", f"{summary['accuracy']:.2%}")

        total_base_stake = summary["base_stake"]
        total_pnl_stake = summary["pnl_stake"]
        roi_stake = summary["roi_stake"]

        st.subheader(f"Stake (valor fixo {base_amount:.2f})")
        st.markdown(
            """
            <style>
            #stake-row [data-testid="stHorizontalBlock"] { gap: 0.25rem !important; }
            #stake-row [data-testid="column"] { padding-left: 0.25rem !important; padding-right: 0.25rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div id="stake-row">', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric(" Base (Stake)", f"{total_base_stake:.2f}")
        with s2:
            st.metric("PnL Stake", f"{total_pnl_stake:.2f}")
        with s3:
            st.metric("ROI Stake", f"{roi_stake:.2%}")
        min_pnl = summary["min_pnl_stake"]
        with s4:
            st.metric("Menor PnL acumulado", f"{min_pnl:.2f}")
        with s5:
            st.metric("Drawdown máximo (Stake)", f"{summary['drawdown_stake']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        if entry_kind == "lay":
            total_stake_liab = summary.get("stake_liab", 0.0)
            total_pnl_liab = summary.get("pnl_liab", 0.0)
            roi_liab = summary.get("roi_liab", 0.0)
            min_pnl_liab = summary.get("min_pnl_liab", 0.0)
            drawdown_liab = summary.get("drawdown_liab", 0.0)
            st.subheader(f"Liability (valor fixo {base_amount:.2f})")
            st.markdown(
                """
                <style>
                #liab-row [data-testid=\"stHorizontalBlock\"] { gap: 0.15rem !important; }
                #liab-row [data-testid=\"column\"] { padding-left: 0.15rem !important; padding-right: 0.15rem !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div id="liab-row">', unsafe_allow_html=True)
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                st.metric(" Stake (Liability)", f"{total_stake_liab:.2f}")
            with r2:
                st.metric("PnL Liability", f"{total_pnl_liab:.2f}")
            with r3:
                st.metric("ROI Liability", f"{roi_liab:.2%}")
            with r4:
                st.metric("Menor PnL (Liability)", f"{min_pnl_liab:.2f}")
            with r5:
                st.metric("Drawdown max (Liability)", f"{drawdown_liab:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        plot = df_block.copy()
        if not plot.empty and "race_time_iso" in plot.columns:
            plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
            plot = plot.dropna(subset=["ts"]).sort_values("ts")

            if x_axis_mode == "Dia":
                plot["date_only"] = plot["ts"].dt.date
                daily = (
                    plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]]
                    .sum()
                    .sort_values("date_only")
                )
                daily["cum_stake"] = (daily["pnl_stake_fixed_10"] * scale_factor).cumsum()
                daily["cum_liab"] = (daily["pnl_liability_fixed_10"] * scale_factor).cumsum()
                with st.expander("Evolucao Stake (PnL acumulado por dia)", expanded=False):
                    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                    ch = (
                        alt.Chart(daily)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("cum_stake:Q", title="PnL"),
                        )
                    )
                    st.altair_chart(alt.layer(zero_line, ch).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                plot_intraday = plot.sort_values("ts").copy()
                plot_intraday["date_only"] = plot_intraday["ts"].dt.date
                plot_intraday["_pnl_stake_scaled"] = plot_intraday["pnl_stake_fixed_10"] * scale_factor
                plot_intraday["intraday_cum_stake"] = plot_intraday.groupby("date_only")["_pnl_stake_scaled"].cumsum()
                daily_min_stake = (
                    plot_intraday.groupby("date_only")["intraday_cum_stake"]
                    .min()
                    .to_frame("min_intraday_stake")
                    .reset_index()
                )
                daily_min_stake["min_intraday_stake"] = daily_min_stake["min_intraday_stake"].clip(upper=0.0)
                daily_close_stake = (
                    plot_intraday.groupby("date_only")["intraday_cum_stake"]
                    .last()
                    .to_frame("close_stake")
                    .reset_index()
                )
                stake_dd = daily_min_stake.merge(daily_close_stake, on="date_only", how="left")
                with st.expander("Drawdown Stake por dia (minimo intradiario)", expanded=False):
                    zero_line_dd = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                    min_line = alt.Chart(stake_dd).mark_line().encode(
                        x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                        y=alt.Y("min_intraday_stake:Q", title="Pior nivel intradiario do dia"),
                    )
                    close_line = alt.Chart(stake_dd).mark_line().encode(
                        x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                        y=alt.Y("close_stake:Q"),
                        color=alt.value("#06D6A0"),
                    )
                    st.altair_chart(alt.layer(zero_line_dd, min_line, close_line).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                if entry_kind == "lay":
                    with st.expander("Evolucao Liability (PnL acumulado por dia)", expanded=False):
                        zero_line2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                        ch2 = (
                            alt.Chart(daily)
                            .mark_line()
                            .encode(
                                x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                                y=alt.Y("cum_liab:Q", title="PnL"),
                            )
                        )
                        st.altair_chart(alt.layer(zero_line2, ch2).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                    plot_intraday["_pnl_liab_scaled"] = plot_intraday["pnl_liability_fixed_10"] * scale_factor
                    plot_intraday["intraday_cum_liab"] = plot_intraday.groupby("date_only")["_pnl_liab_scaled"].cumsum()
                    daily_min_liab = (
                        plot_intraday.groupby("date_only")["intraday_cum_liab"]
                        .min()
                        .to_frame("min_intraday_liab")
                        .reset_index()
                    )
                    daily_min_liab["min_intraday_liab"] = daily_min_liab["min_intraday_liab"].clip(upper=0.0)
                    daily_close_liab = (
                        plot_intraday.groupby("date_only")["intraday_cum_liab"]
                        .last()
                        .to_frame("close_liab")
                        .reset_index()
                    )
                    liab_dd = daily_min_liab.merge(daily_close_liab, on="date_only", how="left")
                    with st.expander("Drawdown Liability por dia (minimo intradiario)", expanded=False):
                        zero_line_dd2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                        min_line_liab = alt.Chart(liab_dd).mark_line().encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("min_intraday_liab:Q", title="Pior nivel intradiario do dia"),
                        )
                        close_line_liab = alt.Chart(liab_dd).mark_line().encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("close_liab:Q"),
                            color=alt.value("#FDE74C"),
                        )
                        st.altair_chart(alt.layer(zero_line_dd2, min_line_liab, close_line_liab).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
            else:
                plot["bet_idx"] = range(1, len(plot) + 1)
                plot["cum_stake"] = (plot["pnl_stake_fixed_10"] * scale_factor).cumsum()
                if entry_kind == "lay":
                    plot["cum_liab"] = (plot["pnl_liability_fixed_10"] * scale_factor).cumsum()
                with st.expander("Evolucao Stake (PnL acumulado por bet)", expanded=False):
                    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                    ch = (
                        alt.Chart(plot)
                        .mark_line()
                        .encode(
                            x=alt.X("bet_idx:Q", title="Bet #"),
                            y=alt.Y("cum_stake:Q", title="PnL"),
                        )
                    )
                    st.altair_chart(alt.layer(zero_line, ch).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                if entry_kind == "lay":
                    with st.expander("Evolucao Liability (PnL acumulado por bet)", expanded=False):
                        zero_line2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                        ch2 = (
                            alt.Chart(plot)
                            .mark_line()
                            .encode(
                                x=alt.X("bet_idx:Q", title="Bet #"),
                                y=alt.Y("cum_liab:Q", title="PnL"),
                            )
                        )
                        st.altair_chart(alt.layer(zero_line2, ch2).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        _render_weekday_perf(df_block, entry_kind)
        _render_hour_bucket_perf(df_block, entry_kind)
        _render_num_runners_perf(df_block, entry_kind)
        _render_forecast_rank_perf(df_block, entry_kind)
        _render_monthly_table(df_block, entry_kind, base_amount)

        show_cols = [
            "date",
            "track_name",
            "category_token",
            "race_time_iso",
            "num_runners",
            "total_matched_volume",
            "tf_top1",
            "tf_top2",
            "tf_top3",
            "vol_top1",
            "vol_top2",
            "vol_top3",
            "second_name_by_volume",
            "third_name_by_volume",
            "pct_diff_second_vs_third",
        ]
        if entry_kind == "lay":
            show_cols += [
                "lay_target_name",
                "lay_target_bsp",
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
            ]
        else:
            show_cols += [
                "back_target_name",
                "back_target_bsp",
                "stake_fixed_10",
                "win_lose",
                "is_green",
                "pnl_stake_fixed_10",
                "roi_row_stake_fixed_10",
            ]
        missing = [c for c in show_cols if c not in df_block.columns]
        for c in missing:
            df_block[c] = ""
        table_label = f"Tabela {title_suffix}"
        with st.expander(table_label, expanded=False):
            st.dataframe(df_block[show_cols], use_container_width=True)

    # Campo base (único) pos-filtros
    _render_base_amount_input(default=1.0)

    filt = st.session_state.get("_filt", pd.DataFrame())
    if entry_type == "both":
        render_block("Resultados BACK", filt[filt["entry_type"] == "back"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "back")
        render_block("Resultados LAY", filt[filt["entry_type"] == "lay"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "lay")
    else:
        render_block(f"Resultados {entry_type.upper()}", filt[filt["entry_type"] == entry_type] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), entry_type)

    # Gráficos menores e cruzados (portados do modelo)
    def _render_small_charts(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot2 = df_block.copy()
        if plot2.empty or "race_time_iso" not in plot2.columns:
            return
        plot2["ts"] = pd.to_datetime(plot2["race_time_iso"], errors="coerce")
        plot2 = plot2.dropna(subset=["ts"]).sort_values("ts")
        plot2["date_only"] = plot2["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 1.0))

        # Por pista
        if x_axis_mode == "Dia":
            td = plot2.groupby(["track_name", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
            if not td.empty:
                td["cum"] = td.groupby("track_name")["pnl_stake_fixed_10"].cumsum() * local_scale
                track_counts = plot2.groupby("track_name", as_index=False).size().rename(columns={"size": "count"})
                td = td.merge(track_counts, on="track_name", how="left")
                td["count"] = td["count"].fillna(0).astype(int)
                td["track_title"] = td["track_name"].astype(str) + " (" + td["count"].astype(str) + ")"
                st.subheader(f"Evolucao por pista (PnL acumulado) - {entry_kind.upper()}")
                base_tracks = (
                    alt.Chart(td)
                    .mark_line()
                    .encode(
                        x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                        y=alt.Y("cum:Q", title="PnL"),
                    )
                    .properties(width=small_width, height=small_height)
                )
                zero_line_t = alt.Chart(td).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                chart_tracks = alt.layer(zero_line_t, base_tracks).facet(facet="track_title:N", columns=4)
                st.altair_chart(chart_tracks.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
        else:
            plot2["bet_idx"] = plot2.groupby("track_name").cumcount() + 1
            plot2["cum"] = plot2.groupby("track_name")["pnl_stake_fixed_10"].cumsum() * local_scale
            td = plot2[["track_name", "bet_idx", "cum"]].copy()
            if not td.empty:
                track_counts = plot2.groupby("track_name", as_index=False).size().rename(columns={"size": "count"})
                td = td.merge(track_counts, on="track_name", how="left")
                td["count"] = td["count"].fillna(0).astype(int)
                td["track_title"] = td["track_name"].astype(str) + " (" + td["count"].astype(str) + ")"
                st.subheader(f"Evolucao por pista (PnL acumulado) - {entry_kind.upper()}")
                base_tracks = (
                    alt.Chart(td)
                    .mark_line()
                    .encode(
                        x=alt.X("bet_idx:Q", title="Bet #"),
                        y=alt.Y("cum:Q", title="PnL"),
                    )
                    .properties(width=small_width, height=small_height)
                )
                zero_line_t = alt.Chart(td).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                chart_tracks = alt.layer(zero_line_t, base_tracks).facet(facet="track_title:N", columns=4)
                st.altair_chart(chart_tracks.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        # Por categoria
        if "category" in plot2.columns:
            cd = plot2[plot2["category"].astype(str).str.len() > 0].copy()
            if not cd.empty:
                cat_counts = cd.groupby("category", as_index=False).size().rename(columns={"size": "count"})
                if x_axis_mode == "Dia":
                    cd = cd.groupby(["category", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
                    cd = cd.merge(cat_counts, on="category", how="left")
                    cd["count"] = cd["count"].fillna(0).astype(int)
                    cd["cum"] = cd.groupby("category")["pnl_stake_fixed_10"].cumsum() * local_scale
                    cd["category_title"] = cd["category"].astype(str) + " (" + cd["count"].astype(str) + ")"
                    st.subheader(f"Evolucao por categoria (PnL acumulado) - {entry_kind.upper()}")
                    base_cats = (
                        alt.Chart(cd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_c = alt.Chart(cd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    chart_cats = alt.layer(zero_line_c, base_cats).facet(facet="category_title:N", columns=4)
                    st.altair_chart(chart_cats.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                else:
                    cd["bet_idx"] = cd.groupby("category").cumcount() + 1
                    cd["cum"] = cd.groupby("category")["pnl_stake_fixed_10"].cumsum() * local_scale
                    cd = cd.merge(cat_counts, on="category", how="left")
                    cd["count"] = cd["count"].fillna(0).astype(int)
                    cd["category_title"] = cd["category"].astype(str) + " (" + cd["count"].astype(str) + ")"
                    st.subheader(f"Evolucao por categoria (PnL acumulado) - {entry_kind.upper()}")
                    base_cats = (
                        alt.Chart(cd)
                        .mark_line()
                        .encode(
                            x=alt.X("bet_idx:Q", title="Bet #"),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_c = alt.Chart(cd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    chart_cats = alt.layer(zero_line_c, base_cats).facet(facet="category_title:N", columns=4)
                    st.altair_chart(chart_cats.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        # Por subcategoria
        if "category_token" in plot2.columns:
            sd = plot2[plot2["category_token"].astype(str).str.len() > 0].copy()
            if not sd.empty:
                sub_counts = sd.groupby("category_token", as_index=False).size().rename(columns={"size": "count"})

                def _subkey(x: str) -> tuple[str, int]:
                    m = re.match(r"^([A-Z]+)(\d+)$", str(x))
                    if m:
                        return (m.group(1), int(m.group(2)))
                    m2 = re.match(r"^([A-Z]+)", str(x))
                    return ((m2.group(1) if m2 else str(x)), 0)

                ordered_tokens = sorted(sub_counts["category_token"].astype(str).unique().tolist(), key=_subkey)
                if x_axis_mode == "Dia":
                    sd = sd.groupby(["category_token", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
                    sd = sd.merge(sub_counts, on="category_token", how="left")
                    sd["count"] = sd["count"].fillna(0).astype(int)
                    sd["cum"] = sd.groupby("category_token")["pnl_stake_fixed_10"].cumsum() * local_scale
                    sd["subcat_title"] = sd["category_token"].astype(str) + " (" + sd["count"].astype(str) + ")"
                    sd["category_token"] = pd.Categorical(sd["category_token"], categories=ordered_tokens, ordered=True)
                    st.subheader(f"Evolucao por subcategoria (PnL acumulado) - {entry_kind.upper()}")
                    base_sub = (
                        alt.Chart(sd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_s = alt.Chart(sd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    count_text_s = (
                        alt.Chart(sd)
                        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
                        .encode(text=alt.Text("count:Q", format=".0f"))
                    )
                    chart_sub = alt.layer(zero_line_s, base_sub, count_text_s).facet(
                        facet=alt.Facet("category_token:N", sort=ordered_tokens, header=alt.Header(title="subcat_title")),
                        columns=4,
                    )
                    st.altair_chart(chart_sub.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                else:
                    sd["bet_idx"] = sd.groupby("category_token").cumcount() + 1
                    sd["cum"] = sd.groupby("category_token")["pnl_stake_fixed_10"].cumsum() * local_scale
                    sd = sd.merge(sub_counts, on="category_token", how="left")
                    sd["count"] = sd["count"].fillna(0).astype(int)
                    sd["subcat_title"] = sd["category_token"].astype(str) + " (" + sd["count"].astype(str) + ")"
                    sd["category_token"] = pd.Categorical(sd["category_token"], categories=ordered_tokens, ordered=True)
                    st.subheader(f"Evolucao por subcategoria (PnL acumulado) - {entry_kind.upper()}")
                    base_sub = (
                        alt.Chart(sd)
                        .mark_line()
                        .encode(
                            x=alt.X("bet_idx:Q", title="Bet #"),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_s = alt.Chart(sd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    count_text_s = (
                        alt.Chart(sd)
                        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
                        .encode(text=alt.Text("count:Q", format=".0f"))
                    )
                    chart_sub = alt.layer(zero_line_s, base_sub, count_text_s).facet(
                        facet=alt.Facet("category_token:N", sort=ordered_tokens, header=alt.Header(title="subcat_title")),
                        columns=4,
                    )
                    st.altair_chart(chart_sub.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        # Por numero de corredores (faixas)
        if "num_runners_bucket" in plot2.columns:
            nd = plot2.dropna(subset=["num_runners_bucket"]).copy()
            if not nd.empty:
                nd["nr_bucket"] = pd.Categorical(nd["num_runners_bucket"], categories=NR_BUCKET_LABELS, ordered=True)
                nr_counts = nd.groupby("nr_bucket", as_index=False, observed=False).size().rename(columns={"size": "count"})
                if x_axis_mode == "Dia":
                    nd = (
                        nd.groupby(["nr_bucket", "date_only"], as_index=False, observed=False)[["pnl_stake_fixed_10"]]
                        .sum()
                        .sort_values("date_only")
                    )
                    nd = nd.merge(nr_counts, on="nr_bucket", how="left")
                    nd["count"] = nd["count"].fillna(0).astype(int)
                    nd["cum"] = nd.groupby("nr_bucket", observed=False)["pnl_stake_fixed_10"].cumsum() * local_scale
                    nd["nr_title"] = nd["nr_bucket"].astype(str) + " (" + nd["count"].astype(str) + ")"
                    st.subheader(f"Evolucao por numero de corredores (PnL acumulado) - {entry_kind.upper()}")
                    base_nr = (
                        alt.Chart(nd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_n = alt.Chart(nd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    chart_nr = alt.layer(zero_line_n, base_nr).facet(
                        facet=alt.Facet("nr_bucket:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
                        columns=4,
                    )
                    st.altair_chart(chart_nr.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                else:
                    nd["bet_idx"] = nd.groupby("nr_bucket", observed=False).cumcount() + 1
                    nd["cum"] = nd.groupby("nr_bucket", observed=False)["pnl_stake_fixed_10"].cumsum() * local_scale
                    nd = nd.merge(nr_counts, on="nr_bucket", how="left")
                    nd["count"] = nd["count"].fillna(0).astype(int)
                    nd["nr_title"] = nd["nr_bucket"].astype(str) + " (" + nd["count"].astype(str) + ")"
                    st.subheader(f"Evolucao por numero de corredores (PnL acumulado) - {entry_kind.upper()}")
                    base_nr = (
                        alt.Chart(nd)
                        .mark_line()
                        .encode(
                            x=alt.X("bet_idx:Q", title="Bet #"),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    zero_line_n = alt.Chart(nd).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                    chart_nr = alt.layer(zero_line_n, base_nr).facet(
                        facet=alt.Facet("nr_bucket:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
                        columns=4,
                    )
                    st.altair_chart(chart_nr.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    def _render_cross_nr_category(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners_bucket" not in plot.columns or "category" not in plot.columns or "race_time_iso" not in plot.columns:
            return
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 1.0))
        plot["nr_bucket"] = pd.Categorical(plot["num_runners_bucket"], categories=NR_BUCKET_LABELS, ordered=True)
        counts = plot.groupby(["category", "nr_bucket"], as_index=False, observed=False).size().rename(columns={"size": "count"})
        if x_axis_mode == "Dia":
            agg = plot.groupby(["category", "nr_bucket", "date_only"], as_index=False, observed=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["category", "nr_bucket"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["category", "nr_bucket"], observed=False)['pnl_stake_fixed_10'].cumsum() * local_scale
            agg["facet_col"] = agg["category"].astype(str) + " (" + agg["count"].astype(str) + ")"
            base = (
                alt.Chart(agg)
                .mark_line()
                .encode(
                    x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                    y=alt.Y("cum:Q", title="PnL"),
                )
                .properties(width=small_width, height=small_height)
            )
        else:
            plot["bet_idx"] = plot.groupby(["category", "nr_bucket"], observed=False).cumcount() + 1
            plot["cum"] = plot.groupby(["category", "nr_bucket"], observed=False)['pnl_stake_fixed_10'].cumsum() * local_scale
            agg = plot[["category", "nr_bucket", "bet_idx", "cum"]].copy()
            agg = agg.merge(counts, on=["category", "nr_bucket"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["facet_col"] = agg["category"].astype(str) + " (" + agg["count"].astype(str) + ")"
            base = (
                alt.Chart(agg)
                .mark_line()
                .encode(
                    x=alt.X("bet_idx:Q", title="Bet #"),
                    y=alt.Y("cum:Q", title="PnL"),
                )
                .properties(width=small_width, height=small_height)
            )
        zero = alt.Chart(agg).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
        count_text = (
            alt.Chart(agg)
            .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
            .encode(text=alt.Text("count:Q", format=".0f"))
        )
        st.subheader(f"Evolucao por categoria  numero de corredores - {entry_kind.upper()}")
        ch = alt.layer(zero, base, count_text).facet(
            row=alt.Facet("nr_bucket:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
            column=alt.Facet("facet_col:N", header=alt.Header(title="")),
        )
        st.altair_chart(ch.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    def _render_cross_nr_track(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners_bucket" not in plot.columns or "race_time_iso" not in plot.columns:
            return
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 1.0))
        top_k = 12
        track_sizes = plot.groupby("track_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        top_tracks = set(track_sizes.head(top_k)["track_name"].tolist())
        plot = plot[plot["track_name"].isin(top_tracks)]
        plot["nr_bucket"] = pd.Categorical(plot["num_runners_bucket"], categories=NR_BUCKET_LABELS, ordered=True)
        counts = plot.groupby(["track_name", "nr_bucket"], as_index=False, observed=False).size().rename(columns={"size": "count"})
        plot = plot.merge(track_sizes[["track_name", "count"]].rename(columns={"count": "track_total"}), on="track_name", how="left")
        if x_axis_mode == "Dia":
            agg = plot.groupby(["track_name", "nr_bucket", "date_only"], as_index=False, observed=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["track_name", "nr_bucket"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["track_name", "nr_bucket"], observed=False)['pnl_stake_fixed_10'].cumsum() * local_scale
            agg["facet_col"] = agg["track_name"].astype(str) + " (" + agg["count"].astype(str) + ")"
            base = (
                alt.Chart(agg)
                .mark_line()
                .encode(
                    x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                    y=alt.Y("cum:Q", title="PnL"),
                )
                .properties(width=small_width, height=small_height)
            )
        else:
            plot["bet_idx"] = plot.groupby(["track_name", "nr_bucket"], observed=False).cumcount() + 1
            plot["cum"] = plot.groupby(["track_name", "nr_bucket"], observed=False)['pnl_stake_fixed_10'].cumsum() * local_scale
            agg = plot[["track_name", "nr_bucket", "bet_idx", "cum"]].copy()
            agg = agg.merge(counts, on=["track_name", "nr_bucket"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["facet_col"] = agg["track_name"].astype(str) + " (" + agg["count"].astype(str) + ")"
            base = (
                alt.Chart(agg)
                .mark_line()
                .encode(
                    x=alt.X("bet_idx:Q", title="Bet #"),
                    y=alt.Y("cum:Q", title="PnL"),
                )
                .properties(width=small_width, height=small_height)
            )
        zero = alt.Chart(agg).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
        count_text = (
            alt.Chart(agg)
            .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
            .encode(text=alt.Text("count:Q", format=".0f"))
        )
        st.subheader(f"Evolucao por pista  numero de corredores - {entry_kind.upper()}")
        ch = alt.layer(zero, base, count_text).facet(
            row=alt.Facet("nr_bucket:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
            column=alt.Facet("facet_col:N", header=alt.Header(title="")),
        )
        st.altair_chart(ch.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    if entry_type == "both":
        _render_small_charts(filt[filt["entry_type"] == "back"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "back")
        _render_cross_nr_category(filt[filt["entry_type"] == "back"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "back")
        _render_cross_nr_track(filt[filt["entry_type"] == "back"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "back")
        _render_small_charts(filt[filt["entry_type"] == "lay"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "lay")
        _render_cross_nr_category(filt[filt["entry_type"] == "lay"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "lay")
        _render_cross_nr_track(filt[filt["entry_type"] == "lay"] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), "lay")
    else:
        _render_small_charts(filt[filt["entry_type"] == entry_type] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), entry_type)
        _render_cross_nr_category(filt[filt["entry_type"] == entry_type] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), entry_type)
        _render_cross_nr_track(filt[filt["entry_type"] == entry_type] if not filt.empty and "entry_type" in filt.columns else pd.DataFrame(), entry_type)


if __name__ == "__main__":
    main()


