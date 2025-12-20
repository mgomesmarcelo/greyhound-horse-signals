import sys
import math
import datetime
import calendar
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
import re
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.horses.config import settings
from src.horses.utils.text import normalize_track_name
from src.horses.analysis.signals import _signals_raw_path, _signals_snapshot_path, _to_iso_series


# Regras disponiveis para cavalos
HORSE_RULE_LABELS: dict[str, str] = {
    "terceiro_queda50": "Regra 1 – 3º com queda ≥ 50% vs 2º",
    "lider_volume_total": "Regra 2 – Líder com volume dominante",
}
HORSE_RULE_LABELS_INV = {v: k for k, v in HORSE_RULE_LABELS.items()}

# Tipo de entrada (Back/Lay)
ENTRY_TYPE_LABELS: dict[str, str] = {
    "back": "Back",
    "lay": "Lay",
}
ENTRY_TYPE_LABELS_INV = {v: k for k, v in ENTRY_TYPE_LABELS.items()}


def _iter_result_paths(pattern: str) -> List[Path]:
    result_dir = settings.DATA_DIR / "Result"
    parquet_paths = sorted(result_dir.glob(f"{pattern}.parquet"))
    if parquet_paths:
        return parquet_paths
    return sorted(result_dir.glob(f"{pattern}.csv"))


def _stat_signature(paths: List[Path]) -> Tuple[Tuple[str, float, str], ...]:
    signature: List[Tuple[str, float, str]] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path), stat.st_mtime, path.suffix.lstrip(".").lower()))
    return tuple(signature)


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
    paths = _iter_result_paths("dwbfprices*win*")
    signature = _stat_signature(paths)
    return _cached_category_index(signature)


def _build_num_runners_index() -> dict[tuple[str, str], int]:
    paths = _iter_result_paths("dwbfprices*win*")
    signature = _stat_signature(paths)
    return _cached_num_runners_index(signature)


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

    ordered = df_block.sort_values("race_time_iso")
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
def _extract_track_from_menu_hint(menu_hint: str) -> str:
    text = menu_hint or ""
    m = re.match(r"^([A-Za-z\s]+?)(?:\s*\d|$)", text)
    base = m.group(1) if m else text
    return normalize_track_name(base)


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
    enriquecimento básico. Filtragem por entry_type deve ser feita no front.
    """
    part_mtime = _get_signals_mtime(source, market, rule, provider)
    df = load_signals(
        source=source,
        market=market,
        rule=rule,
        provider=provider,
        signals_mtime=part_mtime or signals_mtime,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["rule"] = rule
    if "date" in df.columns and "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "race_time_iso" in df.columns and "race_ts" not in df.columns:
        df["race_ts"] = pd.to_datetime(df["race_time_iso"], errors="coerce")
    if "track_name" in df.columns and "race_time_iso" in df.columns:
        df["_key_track"] = df["track_name"].astype(str).map(normalize_track_name)
        df["_key_race"] = df["race_time_iso"].astype(str)

    if "num_runners" not in df.columns:
        num_index = _build_num_runners_index()
        df["num_runners"] = df.apply(
            lambda r: num_index.get((str(r.get("_key_track", "")), str(r.get("_key_race", ""))), pd.NA),
            axis=1,
        )

    if ("category" not in df.columns) or ("category_token" not in df.columns):
        cat_index = _build_category_index()
        df["category"] = df.apply(
            lambda r: (cat_index.get((str(r.get("_key_track", "")), str(r.get("_key_race", ""))), {}) or {}).get("letter", ""),
            axis=1,
        )
        df["category_token"] = df.apply(
            lambda r: (cat_index.get((str(r.get("_key_track", "")), str(r.get("_key_race", ""))), {}) or {}).get("token", ""),
            axis=1,
        )

    return df


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

    def _reset_rule_dependent_state() -> None:
        keys_to_clear = [
            "date_start_input",
            "date_end_input",
            "date_range_slider",
            "dates_range",
            "dates_ms",
            "tracks_ms",
            "num_runners_ms",
            "cats_ms",
            "subcats_ms",
            "sel_num_runners",
            "sel_cats",
            "sel_subcats",
            "bsp_low",
            "bsp_high",
            "bsp_slider",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

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
        provider = st.selectbox("Provedor", ["timeform", "sportinglife"], index=0)

    with col_src:
        source_options = ["top3", "forecast"]
        source_label_options = ["Top 3", "Forecast"]
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
        market = st.selectbox("Mercado", ["win", "place"], index=0)

    with col_entry:
        entry_opt_labels = ["ambos", ENTRY_TYPE_LABELS["back"], ENTRY_TYPE_LABELS["lay"]]
        entry_label = st.selectbox("Tipo de entrada", entry_opt_labels, index=0)
        if entry_label == "ambos":
            entry_type = "both"
        else:
            entry_type = "back" if entry_label == ENTRY_TYPE_LABELS["back"] else "lay"

    st.caption(f"Regra: {selected_rule_label} · Fonte: {source_label} · Mercado: {market} · Entrada: {entry_label} · Provedor: {provider}")

    df = load_signals_enriched(
        source=source,
        market=market,
        rule=rule,
        provider=provider,
    )
    if df.empty:
        st.info(
            "Nenhum sinal encontrado para a selecao. "
            "Gere antes com: python scripts/horses/generate_horse_signals.py --source {src} --market {mkt} --rule {rule} --entry_type {et}".format(
                src=source, mkt=market, rule=rule, et=entry_type
            )
        )
        return

    df_filtered = df.copy()
    if entry_type != "both":
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

            start_col, end_col = st.columns(2)
            with start_col:
                start_selected = st.date_input(
                    "Data inicial",
                    value=sanitized_start,
                    min_value=min_date,
                    max_value=max_date,
                    key=date_start_key,
                )
            with end_col:
                end_selected = st.date_input(
                    "Data final",
                    value=sanitized_end,
                    min_value=min_date,
                    max_value=max_date,
                    key=date_end_key,
                )

            range_start = min(start_selected, end_selected)
            range_end = max(start_selected, end_selected)

            slider_key = "date_range_slider"
            slider_min_dt = datetime.datetime.combine(min_date, datetime.time.min)
            slider_max_dt = datetime.datetime.combine(max_date, datetime.time.min)
            default_slider_value = (
                datetime.datetime.combine(range_start, datetime.time.min),
                datetime.datetime.combine(range_end, datetime.time.min),
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

            if st.session_state.get(slider_key) != current_slider_value:
                st.session_state[slider_key] = current_slider_value

            slider_start_dt, slider_end_dt = st.slider(
                "Intervalo de datas (barra)",
                min_value=slider_min_dt,
                max_value=slider_max_dt,
                value=st.session_state[slider_key],
                format="YYYY-MM-DD",
                key=slider_key,
            )
            slider_start_date = slider_start_dt.date()
            slider_end_date = slider_end_dt.date()

            slider_changed = (slider_start_date != sanitized_start) or (slider_end_date != sanitized_end)

            if slider_changed:
                range_start = slider_start_date
                range_end = slider_end_date
                final_slider_state = (
                    datetime.datetime.combine(range_start, datetime.time.min),
                    datetime.datetime.combine(range_end, datetime.time.min),
                )
                if st.session_state.get(slider_key) != final_slider_state:
                    st.session_state[slider_key] = final_slider_state
            else:
                range_start = sanitized_start
                range_end = sanitized_end
                desired_slider_state = (
                    datetime.datetime.combine(range_start, datetime.time.min),
                    datetime.datetime.combine(range_end, datetime.time.min),
                )
                if st.session_state.get(slider_key) != desired_slider_state:
                    st.session_state[slider_key] = desired_slider_state

            range_start = max(min_date, range_start)
            range_end = min(max_date, range_end)
            if range_start > range_end:
                range_end = range_start

            dates_in_range = [d for d, parsed in parsed_dates if range_start <= parsed <= range_end]
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

    # Volume mínimo por corrida
    vol_col, _ = st.columns([2, 1])
    volume_key = "min_total_volume"
    volume_series_raw = pd.to_numeric(df_filtered.get("total_matched_volume", pd.Series(dtype=float)), errors="coerce")
    volume_series = volume_series_raw.dropna()
    if volume_series.empty:
        # Sem volume disponível: não filtra, mas informa
        with vol_col:
            st.info("Sem coluna de volume para filtrar (total_matched_volume).")
    else:
        vol_min_data = max(0.0, float(volume_series.min()))
        vol_max_data = float(volume_series.max())
        if volume_key not in st.session_state:
            st.session_state[volume_key] = vol_min_data
        # Clampa valor salvo
        saved_val = float(st.session_state.get(volume_key, vol_min_data))
        saved_val = min(max(saved_val, vol_min_data), vol_max_data)
        st.session_state[volume_key] = saved_val
        with vol_col:
            vcol, _ = st.columns([3, 7])
            with vcol:
                st.number_input(
                    "Volume total negociado mínimo",
                    min_value=vol_min_data,
                    max_value=vol_max_data,
                    value=float(st.session_state[volume_key]),
                    step=100.0 if vol_max_data - vol_min_data > 1000 else max(10.0, (vol_max_data - vol_min_data) / 20),
                    format="%.0f",
                    key=volume_key,
                    help="Considera apenas corridas cuja soma de pptradedvol atinge o mínimo desejado.",
                )
                st.caption("Volume min. por corrida (soma de pptradedvol)")
        min_total_volume = float(st.session_state.get(volume_key, vol_min_data))
        mask_volume = volume_series_raw.fillna(0.0)
        if not mask_volume.index.equals(df_filtered.index):
            mask_volume = mask_volume.reindex(df_filtered.index, fill_value=0.0)
        df_filtered = df_filtered[mask_volume >= min_total_volume]

    # Numero de corredores (faixas)
    df_filtered = df_filtered.copy()
    df_filtered["num_runners_bucket"] = _bucket_num_runners(df_filtered.get("num_runners"))
    st.caption("Numero de corredores (faixas)")
    nr_buckets = [b for b in NR_BUCKET_LABELS if b in df_filtered["num_runners_bucket"].dropna().unique().tolist()]
    if nr_buckets:
        btn_all, btn_clear, _ = st.columns([1, 1, 6])
        with btn_all:
            st.button(
                "Todos",
                key="nr_bucket_all",
                on_click=lambda: st.session_state.update({"num_runners_bucket_ms": nr_buckets}),
            )
        with btn_clear:
            st.button(
                "Limpar",
                key="nr_bucket_none",
                on_click=lambda: st.session_state.update({"num_runners_bucket_ms": []}),
            )
        col_ms, _ = st.columns([2, 8])
        with col_ms:
            prev_nr = st.session_state.get("sel_num_runners_bucket", nr_buckets.copy())
            default_nr = [v for v in prev_nr if v in nr_buckets]
            sel_nr = st.multiselect(
                "Numero de corredores (faixas)",
                nr_buckets,
                default=default_nr if default_nr else nr_buckets,
                key="num_runners_bucket_ms",
                label_visibility="collapsed",
            )
        sel_nr = [v for v in sel_nr if v in nr_buckets]
        st.session_state["sel_num_runners_bucket"] = sel_nr
        if sel_nr:
            df_filtered = df_filtered[df_filtered["num_runners_bucket"].isin(sel_nr)]
        else:
            df_filtered = df_filtered.iloc[0:0]

    # Regra líder
    if rule == "lider_volume_total":
        col_l1, col_l2 = st.columns([1, 2])
        with col_l1:
            leader_min = st.number_input(
                "Participacao do lider (%) min.",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                format="%.0f",
            )
        df_filtered = df_filtered[df_filtered["leader_volume_share_pct"].fillna(0) >= float(leader_min)]

    # Categoria e token
    if ("category" in df_filtered.columns) and ("category_token" in df_filtered.columns):
        if not df_filtered.empty:
            cat_letters = sorted(
                [
                    c
                    for c in df_filtered["category"].dropna().unique().tolist()
                    if isinstance(c, str) and c
                ]
            )
        else:
            cat_letters = []
    else:
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
            cat_letters = sorted(
                [
                    c
                    for c in df_filtered["category"].dropna().unique().tolist()
                    if isinstance(c, str) and c
                ]
            )
        else:
            cat_letters = []

    if sel_tracks:
        df_filtered = df_filtered[df_filtered["track_name"].isin(sel_tracks)]

    if rule == "terceiro_queda50":
        df_filtered = df_filtered[df_filtered["pct_diff_second_vs_third"].fillna(0) > 50.0]

    bsp_low = float(st.session_state.get("bsp_low", bsp_min))
    bsp_high = float(st.session_state.get("bsp_high", bsp_max))
    if entry_type == "both":
        df_filtered = df_filtered[
            ((df_filtered["entry_type"] == "lay") & (df_filtered["lay_target_bsp"].between(bsp_low, bsp_high)))
            | ((df_filtered["entry_type"] == "back") & (df_filtered["back_target_bsp"].between(bsp_low, bsp_high)))
        ]
    else:
        current_bsp_col = "lay_target_bsp" if entry_type == "lay" else "back_target_bsp"
        df_filtered = df_filtered[(df_filtered[current_bsp_col] >= bsp_low) & (df_filtered[current_bsp_col] <= bsp_high)]

    # Categorias
    st.caption("Categorias")
    if cat_letters:
        btn_all, btn_clear, _ = st.columns([1, 1, 6])
        with btn_all:
            st.button(
                "Todas",
                key="cats_all",
                on_click=lambda: st.session_state.update({"cats_ms": cat_letters}),
            )
        with btn_clear:
            st.button(
                "Limpar",
                key="cats_none",
                on_click=lambda: st.session_state.update({"cats_ms": []}),
            )
        col_ms, _ = st.columns([3, 7])
        with col_ms:
            prev = st.session_state.get("sel_cats", cat_letters.copy())
            default_cats = [c for c in prev if c in cat_letters]
            sel_cats = st.multiselect(
                "Categoria (G/H/M/N/...)",
                cat_letters,
                default=default_cats,
                key="cats_ms",
                label_visibility="collapsed",
            )
            sel_cats = [c for c in sel_cats if c in cat_letters]
            st.session_state["sel_cats"] = sel_cats
        if sel_cats:
            df_filtered = df_filtered[df_filtered["category"].isin(sel_cats)]
        else:
            df_filtered = df_filtered.iloc[0:0]

    # Subcategorias
    sub_tokens = []
    if "category_token" in df_filtered.columns:
        raw_tokens = [
            t
            for t in df_filtered["category_token"].dropna().astype(str).unique().tolist()
            if isinstance(t, str) and t
        ]

        def _sub_sort_key(tok: str) -> tuple[str, int]:
            m = re.match(r"^([A-Z]+)(\d+)$", str(tok))
            if m:
                return (m.group(1), int(m.group(2)))
            m2 = re.match(r"^([A-Z]+)", str(tok))
            return ((m2.group(1) if m2 else str(tok)), 0)

        sub_tokens = sorted(raw_tokens, key=_sub_sort_key)
    if sub_tokens:
        st.caption("Subcategorias")
        scw, _ = st.columns([2, 5])
        with scw:
            sb1, sb2 = st.columns([1, 1])
            with sb1:
                st.button(
                    "Todas",
                    key="subcats_all",
                    on_click=lambda: st.session_state.update({"subcats_ms": sub_tokens}),
                )
            with sb2:
                st.button(
                    "Limpar",
                    key="subcats_none",
                    on_click=lambda: st.session_state.update({"subcats_ms": []}),
                )
            prev_sc = st.session_state.get("sel_subcats", sub_tokens.copy())
            default_sc = [t for t in prev_sc if t in sub_tokens]
            sel_subcats = st.multiselect(
                "Subcategorias (G1/HCP_CHS/MDN/...)",
                sub_tokens,
                default=default_sc,
                key="subcats_ms",
                label_visibility="collapsed",
            )
            sel_subcats = [t for t in sel_subcats if t in sub_tokens]
            st.session_state["sel_subcats"] = sel_subcats
        if sel_subcats:
            df_filtered = df_filtered[df_filtered["category_token"].isin(sel_subcats)]
        else:
            df_filtered = df_filtered.iloc[0:0]

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

    def _render_monthly_table(df_block: pd.DataFrame, entry_kind: str, base_amount: float | None = None) -> None:
        if base_amount is None:
            base_amount = float(st.session_state.get("base_amount", 1.0))
        working = df_block.copy()
        if "race_ts" not in working.columns:
            working["race_ts"] = pd.to_datetime(working["race_time_iso"], errors="coerce")
        else:
            working["race_ts"] = pd.to_datetime(working["race_ts"], errors="coerce")
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
        if not monthly_df.empty:
            monthly_df = monthly_df.sort_values("_period").drop(columns=["_period"], errors="ignore")
        with st.expander(f"Relatorio mensal ({entry_kind.upper()})", expanded=False):
            st.dataframe(monthly_df, use_container_width=True)

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
        if not plot.empty:
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

    if entry_type == "both":
        render_block("Resultados BACK", df_filtered[df_filtered["entry_type"] == "back"], "back")
        render_block("Resultados LAY", df_filtered[df_filtered["entry_type"] == "lay"], "lay")
    else:
        # Quando apenas um tipo é exibido, renderize bloco único
        render_block(f"Resultados {entry_type.upper()}", df_filtered[df_filtered["entry_type"] == entry_type], entry_type)

    # Gráficos menores e cruzados (portados do modelo)
    def _render_small_charts(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot2 = df_block.copy()
        if plot2.empty:
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
                        facet=alt.Facet("nr_title:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
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
                        facet=alt.Facet("nr_title:N", sort=NR_BUCKET_LABELS, header=alt.Header(title="")),
                        columns=4,
                    )
                    st.altair_chart(chart_nr.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    def _render_cross_nr_category(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners_bucket" not in plot.columns or "category" not in plot.columns:
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
        if plot.empty or "num_runners_bucket" not in plot.columns:
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
        _render_small_charts(df_filtered[df_filtered["entry_type"] == "back"], "back")
        _render_cross_nr_category(df_filtered[df_filtered["entry_type"] == "back"], "back")
        _render_cross_nr_track(df_filtered[df_filtered["entry_type"] == "back"], "back")
        _render_small_charts(df_filtered[df_filtered["entry_type"] == "lay"], "lay")
        _render_cross_nr_category(df_filtered[df_filtered["entry_type"] == "lay"], "lay")
        _render_cross_nr_track(df_filtered[df_filtered["entry_type"] == "lay"], "lay")
    else:
        _render_small_charts(df_filtered[df_filtered["entry_type"] == entry_type], entry_type)
        _render_cross_nr_category(df_filtered[df_filtered["entry_type"] == entry_type], entry_type)
        _render_cross_nr_track(df_filtered[df_filtered["entry_type"] == entry_type], entry_type)


if __name__ == "__main__":
    main()


