import datetime
import calendar
import math
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
import re
from dateutil import parser as date_parser
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.greyhounds.config import settings
from src.greyhounds.config import RULE_LABELS, RULE_LABELS_INV, ENTRY_TYPE_LABELS, SOURCE_LABELS, SOURCE_LABELS_INV
from src.greyhounds.utils.text import normalize_track_name
from scripts.greyhounds.units_helper import get_ref_factor, get_scale, get_col, to_bool_series

# Fator de referência global (definido em tempo de execução com base no dataset carregado).
_REF_FACTOR: float = 10.0

# Formato ISO para race_time_iso (ex: 2025-09-06T17:25 ou 2025-09-06T17:25:00).
_RACE_TIME_ISO_FORMAT = "%Y-%m-%dT%H:%M"


def _iter_result_paths(pattern: str) -> List[Path]:
    parquet_paths = sorted(settings.PROCESSED_RESULT_DIR.glob(f"{pattern}.parquet"))
    if parquet_paths:
        return parquet_paths
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
        df_r["race_iso"] = df_r["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        df_r["cat_letter"] = df_r["event_name"].astype(str).map(_extract_category_letter)
        df_r["cat_token"] = df_r["event_name"].astype(str).map(_extract_category_token)
        mask = (df_r["track_key"].astype(str) != "") & (df_r["race_iso"].astype(str) != "")
        df_r = df_r.loc[mask].drop_duplicates(subset=["track_key", "race_iso"], keep="first")
        if df_r.empty:
            continue
        keys = list(zip(df_r["track_key"].astype(str), df_r["race_iso"].astype(str)))
        values = [
            {"letter": str(a), "token": str(b)}
            for a, b in zip(df_r["cat_letter"], df_r["cat_token"])
        ]
        for k, v in zip(keys, values):
            if k not in mapping:
                mapping[k] = v
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
        df_r["race_iso"] = df_r["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        mask = (df_r["track_key"].astype(str) != "") & (df_r["race_iso"].astype(str) != "")
        df_r = df_r.loc[mask]
        grp = df_r.groupby(["track_key", "race_iso"]).size()
        counts.update(((str(k[0]), str(k[1])), int(v)) for k, v in grp.items())
    return counts


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    text = menu_hint or ""
    m = re.match(r"^([A-Za-z\s]+?)(?:\s*\d|$)", text)
    base = m.group(1) if m else text
    return normalize_track_name(base)


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


def _build_category_index() -> dict[tuple[str, str], dict[str, str]]:
    paths = _iter_result_paths("dwbfgreyhoundwin*")
    signature = _stat_signature(paths)
    return _cached_category_index(signature)


def _calc_drawdown(series: pd.Series) -> float:
    """Retorna o drawdown máximo (maior perda acumulada) em valor negativo."""
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = series - running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _compute_summary_metrics(df_block: pd.DataFrame, entry_kind: str, base_amount: float) -> dict[str, float]:
    """Calcula métricas agregadas usadas no cabeçalho e no relatório mensal."""
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

    ref_factor = _REF_FACTOR if _REF_FACTOR else get_ref_factor(df_block)
    scale_factor = get_scale(base_amount, ref_factor)
    metrics["tracks"] = int(df_block["track_name"].nunique())
    metrics["signals"] = int(len(df_block))
    metrics["greens"] = int((df_block["is_green"] == True).sum())
    metrics["reds"] = int(metrics["signals"] - metrics["greens"])
    metrics["accuracy"] = (metrics["greens"] / metrics["signals"]) if metrics["signals"] > 0 else 0.0

    if entry_kind == "lay":
        metrics["avg_bsp"] = float(df_block["lay_target_bsp"].mean())
    else:
        metrics["avg_bsp"] = float(df_block["back_target_bsp"].mean())

    # Séries canônicas
    stake_series = get_col(df_block, "stake_ref", "stake_fixed_10")
    pnl_stake_series = get_col(df_block, "pnl_stake_ref", "pnl_stake_fixed_10")

    liability_series = get_col(df_block, "liability_ref", "liability_fixed_10")
    pnl_liab_series = get_col(df_block, "pnl_liability_ref", "pnl_liability_fixed_10") if entry_kind == "lay" else None

    exposure_series = get_col(df_block, "liability_from_stake_ref", "liability_from_stake_fixed_10")

    # Stake ROI: PnL stake / base stake (ambos reescalados)
    total_base_stake_raw = float(stake_series.sum())
    total_pnl_stake_raw = float(pnl_stake_series.sum())
    metrics["base_stake"] = total_base_stake_raw * scale_factor
    metrics["pnl_stake"] = total_pnl_stake_raw * scale_factor
    metrics["roi_stake"] = (metrics["pnl_stake"] / metrics["base_stake"]) if metrics["base_stake"] > 0 else 0.0

    # ROI exposição (PnL stake / liability derivada da stake) – opcional, mantido como campo separado
    total_exposure_raw = float(exposure_series.sum())
    metrics["roi_stake_exposure"] = (metrics["pnl_stake"] / (total_exposure_raw * scale_factor)) if total_exposure_raw > 0 else 0.0

    ordered = df_block.sort_values("race_time_iso")
    cumulative_stake = (get_col(ordered, "pnl_stake_ref", "pnl_stake_fixed_10") * scale_factor).cumsum()
    metrics["min_pnl_stake"] = float(cumulative_stake.min()) if not cumulative_stake.empty else 0.0
    metrics["drawdown_stake"] = _calc_drawdown(cumulative_stake)

    if entry_kind == "lay":
        # Liability ROI: PnL liability / base liability (ambos reescalados)
        total_base_liab_raw = float(liability_series.sum())
        metrics["stake_liab"] = total_base_liab_raw * scale_factor
        total_pnl_liab_raw = float(pnl_liab_series.sum()) if pnl_liab_series is not None else 0.0
        metrics["pnl_liab"] = total_pnl_liab_raw * scale_factor
        metrics["roi_liab"] = (metrics["pnl_liab"] / metrics["stake_liab"]) if metrics["stake_liab"] > 0 else 0.0

        cumulative_liab = (get_col(ordered, "pnl_liability_ref", "pnl_liability_fixed_10") * scale_factor).cumsum()
        metrics["min_pnl_liab"] = float(cumulative_liab.min()) if not cumulative_liab.empty else 0.0
        metrics["drawdown_liab"] = _calc_drawdown(cumulative_liab)
    return metrics

# Teste rápido de assertividade esperado:
# signals=10, greens=6 -> accuracy=0.6 -> 60.00%
# signals=0, greens=0 -> accuracy=0.0 -> 0.00%


def _format_month_label(ts: pd.Timestamp) -> str:
    """Retorna rótulo no formato Jan/2024 (abreviação em inglês)."""
    if pd.isna(ts):
        return ""
    month_abbr = calendar.month_abbr[ts.month] if ts.month in range(1, 13) else ""
    return f"{month_abbr}/{ts.year}"


def _render_base_amount_input() -> float:
    """Campo único para definir unidades por aposta (% da banca)."""
    if "base_amount" not in st.session_state:
        st.session_state["base_amount"] = 1.00
    col_base, _ = st.columns([1, 6])
    with col_base:
        val = st.number_input(
            "Unidades por aposta (% da banca)",
            min_value=0.01,
            max_value=100000.0,
            step=0.50,
            format="%.2f",
            key="base_amount",
            label_visibility="collapsed",
            help="1 unidade = 1% da banca por aposta; 10 unidades = 10% por aposta.",
        )
        st.caption("1 unidade = 1% da banca; 10 unidades = 10%")
    return float(val)


def _build_num_runners_index() -> dict[tuple[str, str], int]:
    """Conta corredores por corrida a partir dos CSVs WIN (linhas por evento)."""
    paths = _iter_result_paths("dwbfgreyhoundwin*")
    signature = _stat_signature(paths)
    return _cached_num_runners_index(signature)


def _get_signals_mtime(source: str, market: str, rule: str) -> float:
    parquet_path = settings.PROCESSED_SIGNALS_DIR / f"signals_{source}_{market}_{rule}.parquet"
    if parquet_path.exists():
        try:
            return parquet_path.stat().st_mtime
        except OSError:
            return 0.0
    csv_path = settings.RAW_SIGNALS_DIR / f"signals_{source}_{market}_{rule}.csv"
    if csv_path.exists():
        try:
            return csv_path.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


def load_signals(source: str = "top3", market: str = "win", rule: str = "terceiro_queda50") -> pd.DataFrame:
    parquet_path = settings.PROCESSED_SIGNALS_DIR / f"signals_{source}_{market}_{rule}.parquet"
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

    csv_path = settings.RAW_SIGNALS_DIR / f"signals_{source}_{market}_{rule}.csv"
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
    source: str = "top3",
    market: str = "win",
    rule: str = "terceiro_queda50",
    signals_mtime: float = 0.0,
) -> pd.DataFrame:
    """
    Carrega os sinais e faz um enriquecimento pesado (num_runners, categoria, etc.)
    apenas uma vez, reaproveitando via cache entre interações do Streamlit.
    O parâmetro signals_mtime é usado apenas para invalidar o cache quando o
    arquivo de sinais for atualizado em disco.
    """
    del signals_mtime  # usado somente para a chave de cache
    df = load_signals(source=source, market=market, rule=rule)
    if df.empty:
        return df

    # Colunas auxiliares de data/tempo
    if "date" in df.columns and "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "race_time_iso" in df.columns and "race_ts" not in df.columns:
        df["race_ts"] = pd.to_datetime(df["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")

    # Chaves de corrida
    if "track_name" in df.columns and "race_time_iso" in df.columns:
        df["_key_track"] = df["track_name"].astype(str).map(normalize_track_name)
        df["_key_race"] = df["race_time_iso"].astype(str)

    if "_key_track" in df.columns and "_key_race" in df.columns:
        keys = list(zip(df["_key_track"].astype(str), df["_key_race"].astype(str)))
        keys_series = pd.Series(keys, index=df.index)

        # Enriquecimento: numero de corredores (fallback se ausente no parquet)
        if "num_runners" not in df.columns:
            num_index = _build_num_runners_index()
            df["num_runners"] = keys_series.map(num_index).astype("Int64")

        # Enriquecimento: categoria por corrida (A/B/D etc.) somente se faltar
        if ("category" not in df.columns) or ("category_token" not in df.columns):
            cat_index = _build_category_index()
            cat_letter = {k: v.get("letter", "") for k, v in cat_index.items()}
            cat_token = {k: v.get("token", "") for k, v in cat_index.items()}
            df["category"] = keys_series.map(cat_letter).fillna("").astype(str)
            df["category_token"] = keys_series.map(cat_token).fillna("").astype(str)

    return df


def main() -> None:
    st.set_page_config(page_title="Sinais LAY/BACK - Galgos", layout="wide")
    st.title("Sinais LAY/BACK - Estrategias Greyhounds")

    # (Sem CSS custom)  Restaurado layout padrao do Streamlit

    # Dimensoes padrao para graficos pequenos (usadas em varias secoes)
    small_width = 360
    small_height = 180

    def _render_weekday_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por dia da semana com linha zero."""
        if df_block.empty or "race_time_iso" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        plot = df_block.copy()
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        if plot.empty:
            return
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
        plot["date_only"] = plot["ts"].dt.date
        daily = (
            plot.groupby("date_only", as_index=False)[["_pnl_stake", "_pnl_liab"]]
            .sum()
            .sort_values("date_only")
        )
        daily["weekday"] = pd.to_datetime(daily["date_only"]).dt.weekday
        wd_order = [0, 1, 2, 3, 4, 5, 6]
        wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
        wd_order_names = [wd_names[w] for w in wd_order]
        by_wd = daily.groupby("weekday", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        by_wd["weekday_name"] = by_wd["weekday"].map(wd_names)
        by_wd["weekday_name"] = pd.Categorical(by_wd["weekday_name"], categories=wd_order_names, ordered=True)
        by_wd["pnl_stake"] = by_wd["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_wd["pnl_liab"] = by_wd["_pnl_liab"] * scale_factor

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

    def _render_trap_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por trap com mesma lógica do gráfico semanal."""
        if df_block.empty or "trap_number" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        plot = df_block.copy()
        plot["trap_number"] = pd.to_numeric(plot["trap_number"], errors="coerce").astype("Int64")
        plot = plot.dropna(subset=["trap_number"])
        if plot.empty:
            return
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
        by_trap = plot.groupby("trap_number", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        order_labels = ["1", "2", "3", "4", "5", "6"]
        by_trap["trap_label"] = by_trap["trap_number"].astype(int).astype(str)
        by_trap = by_trap[by_trap["trap_label"].isin(order_labels)]
        if by_trap.empty:
            return
        by_trap["trap_label"] = pd.Categorical(by_trap["trap_label"], categories=order_labels, ordered=True)
        by_trap["pnl_stake"] = by_trap["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_trap["pnl_liab"] = by_trap["_pnl_liab"] * scale_factor

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        bar_stake = (
            alt.Chart(by_trap)
            .mark_bar()
            .encode(
                x=alt.X("trap_label:N", sort=order_labels, title=""),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por trap (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                bar_liab = (
                    alt.Chart(by_trap)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("trap_label:N", sort=order_labels, title=""),
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
        """Barra por faixa de horario (mesmos buckets do filtro)."""
        if df_block.empty or "race_time_iso" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        plot = df_block.copy()

        def _bucket(series: pd.Series) -> pd.Series:
            ts = pd.to_datetime(series, format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            minutes = ts.dt.hour * 60 + ts.dt.minute
            bucket = pd.Series(index=series.index, dtype="object")
            bucket[(minutes >= 8 * 60) & (minutes <= 12 * 60)] = "08:00-12:00"
            bucket[(minutes >= 12 * 60 + 1) & (minutes <= 16 * 60)] = "12:01-16:00"
            bucket[(minutes >= 16 * 60 + 1) & (minutes <= 20 * 60)] = "16:01-20:00"
            bucket[(minutes >= 20 * 60 + 1)] = "20:01-23:59"
            return bucket

        bucket_order = ["08:00-12:00", "12:01-16:00", "16:01-20:00", "20:01-23:59"]
        plot["hour_bucket"] = _bucket(plot["race_time_iso"])
        plot = plot.dropna(subset=["hour_bucket"])
        if plot.empty:
            return
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
        by_bucket = plot.groupby("hour_bucket", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        by_bucket = by_bucket[by_bucket["hour_bucket"].isin(bucket_order)]
        if by_bucket.empty:
            return
        by_bucket["hour_bucket"] = pd.Categorical(by_bucket["hour_bucket"], categories=bucket_order, ordered=True)
        by_bucket = by_bucket.sort_values("hour_bucket")
        by_bucket["pnl_stake"] = by_bucket["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_bucket["pnl_liab"] = by_bucket["_pnl_liab"] * scale_factor

        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="red", strokeWidth=1)
            .encode(y="y:Q")
        )
        bar_stake = (
            alt.Chart(by_bucket)
            .mark_bar()
            .encode(
                x=alt.X(
                    "hour_bucket:N",
                    sort=bucket_order,
                    title="",
                ),
                y=alt.Y("pnl_stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        stake_chart = alt.layer(zero_line, bar_stake)
        with st.expander("Desempenho por faixa horária (PnL agregado)", expanded=False):
            if entry_kind == "lay":
                zero_line_liab = (
                    alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(color="red", strokeWidth=1)
                    .encode(y="y:Q")
                )
                bar_liab = (
                    alt.Chart(by_bucket)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X(
                            "hour_bucket:N",
                            sort=bucket_order,
                            title="",
                        ),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                liab_chart = alt.layer(zero_line_liab, bar_liab)

                chart = alt.vconcat(
                    stake_chart.properties(title="Stake"),
                    liab_chart.properties(title="Liability"),
                ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1)
            else:
                chart = stake_chart.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake")

            st.altair_chart(chart, use_container_width=True)

    def _render_forecast_rank_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por forecast_rank (so aparece quando a regra e forecast_odds e ha coluna forecast_rank)."""
        if df_block.empty or "forecast_rank" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        plot = df_block.copy()
        plot["forecast_rank"] = pd.to_numeric(plot["forecast_rank"], errors="coerce")
        plot = plot.dropna(subset=["forecast_rank"])
        if plot.empty:
            return
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
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

    # seletores de regra, fonte, mercado e tipo de entrada
    # Controles principais mais compactos; coluna extra para respiro
    col_rule, col_src, col_mkt, col_entry, _ = st.columns([1, 1, 1, 1, 2])

    rule_label_pairs = [
        ("terceiro_queda50", RULE_LABELS.get("terceiro_queda50", "terceiro_queda50")),
        ("lider_volume_total", RULE_LABELS.get("lider_volume_total", "líder volume total")),
        ("forecast_odds", RULE_LABELS.get("forecast_odds", "Forecast Odds (Timeform)")),
    ]
    # Diagnóstico temporário (descomente para ver qual config está sendo importado):
    # import src.greyhounds.config as _cfg
    # st.write("config.__file__:", getattr(_cfg, "__file__", "?"))
    # st.write("RULE_LABELS.keys():", list(getattr(_cfg, "RULE_LABELS", {}).keys()))
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
            "forecast_rank_ms",
            "value_ratio_min",
            "value_ratio_max",
            "only_value_bets",
            "weekdays_ms",
            "sel_weekdays",
            "hour_bucket_ms",
            "trap_ms",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

    with col_rule:
        if "rule_select_label" not in st.session_state:
            st.session_state["rule_select_label"] = rule_labels[0]

        def _on_rule_change() -> None:
            _reset_rule_dependent_state()

        selected_rule_label = st.selectbox(
            "Regra de selecao",
            rule_labels,
            key="rule_select_label",
            on_change=_on_rule_change,
        )
        # label -> rule (robusto, mesmo se RULE_LABELS_INV estiver incompleto)
        rule = RULE_LABELS_INV.get(selected_rule_label)
        if not rule and selected_rule_label == "Forecast Odds (Timeform)":
            rule = "forecast_odds"
        if not rule:
            rule = "terceiro_queda50"

    with col_src:
        source_options = ["top3", "forecast", "betfair_resultado"]
        source_label_options = [SOURCE_LABELS.get(opt, opt) for opt in source_options]
        if "source_select_label" not in st.session_state:
            st.session_state["source_select_label"] = source_label_options[0]
        selected_source_label = st.selectbox(
            "Fonte de dados",
            source_label_options,
            key="source_select_label",
        )
        source = SOURCE_LABELS_INV.get(selected_source_label, "top3")
    source_label = SOURCE_LABELS.get(source, source)

    with col_mkt:
        market = st.selectbox("Mercado", ["win", "place"], index=0)
    with col_entry:
        entry_opt_labels = ["ambos", ENTRY_TYPE_LABELS["back"], ENTRY_TYPE_LABELS["lay"]]
        entry_label = st.selectbox("Tipo de entrada", entry_opt_labels, index=0)
        if entry_label == "ambos":
            entry_type = "both"
        else:
            entry_type = "back" if entry_label == ENTRY_TYPE_LABELS["back"] else "lay"

    st.caption(f"Regra selecionada: {selected_rule_label} · Fonte de dados: {source_label}")

    signals_mtime = _get_signals_mtime(source, market, rule)
    df = load_signals_enriched(source=source, market=market, rule=rule, signals_mtime=signals_mtime)
    with st.expander("Debug (carregamento de sinais)", expanded=False):
        st.write("PROCESSED_SIGNALS_DIR:", str(settings.PROCESSED_SIGNALS_DIR))
        st.write("Selecionado:", {"source": source, "market": market, "rule": rule})
        expected = settings.PROCESSED_SIGNALS_DIR / f"signals_{source}_{market}_{rule}.parquet"
        st.write("Esperado:", str(expected), "exists=", expected.exists())
        st.write("df.shape:", df.shape)
        st.write("df.columns(sample):", list(df.columns)[:80])
    if df.empty:
        st.info("Nenhum sinal encontrado para a selecao. Gere antes com: python scripts/greyhounds/generate_greyhound_signals.py --source {src} --market {mkt} --rule {rule} --entry_type both".format(src=source, mkt=market, rule=rule))
        return

    # Normaliza is_green (bool) para evitar assertividade zerada por tipos diferentes.
    if "is_green" in df.columns:
        df["is_green"] = to_bool_series(df["is_green"])
    elif "win_lose" in df.columns:
        win_numeric = pd.to_numeric(df["win_lose"], errors="coerce")
        entry_series = df.get("entry_type", pd.Series(dtype=str)).astype(str)
        df["is_green"] = False
        df.loc[entry_series == "back", "is_green"] = win_numeric.loc[entry_series == "back"] == 1
        df.loc[entry_series == "lay", "is_green"] = win_numeric.loc[entry_series == "lay"] == 0

    # Normaliza is_green (bool) para evitar assertividade zerada por tipos diferentes.
    if "is_green" in df.columns:
        df["is_green"] = to_bool_series(df["is_green"])
    elif "win_lose" in df.columns:
        win_numeric = pd.to_numeric(df["win_lose"], errors="coerce")
        entry_series = df.get("entry_type", pd.Series(dtype=str)).astype(str)
        df["is_green"] = False
        df.loc[entry_series == "back", "is_green"] = win_numeric.loc[entry_series == "back"] == 1
        df.loc[entry_series == "lay", "is_green"] = win_numeric.loc[entry_series == "lay"] == 0

    global _REF_FACTOR
    _REF_FACTOR = get_ref_factor(df)

    df_filtered = df.copy()
    if "total_matched_volume" not in df_filtered.columns:
        df_filtered["total_matched_volume"] = pd.NA
    df_filtered["total_matched_volume"] = pd.to_numeric(df_filtered["total_matched_volume"], errors="coerce")

    # Filtros (sem ratio; regra fixa >50%)
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
            )
            st.session_state["date_mode"] = active_mode

            start_col, end_col = st.columns(2)
            with start_col:
                start_selected = st.date_input(
                    "Data inicial",
                    min_value=min_date,
                    max_value=max_date,
                    key=date_start_key,
                    disabled=active_mode == "Barra",
                )
            with end_col:
                end_selected = st.date_input(
                    "Data final",
                    min_value=min_date,
                    max_value=max_date,
                    key=date_end_key,
                    disabled=active_mode == "Barra",
                )

            # Faixa inicial derivada do calendário (default)
            range_start = min(start_selected, end_selected)
            range_end = max(start_selected, end_selected)

            slider_key = "date_range_slider"
            slider_min_dt = datetime.datetime.combine(min_date, datetime.time.min)
            slider_max_dt = datetime.datetime.combine(max_date, datetime.time.min)
            default_slider_value = (
                datetime.datetime.combine(range_start, datetime.time.min),
                datetime.datetime.combine(range_end, datetime.time.min),
            )

            # Valor do slider: session_state como unica fonte; clamp antes do widget
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
                disabled=active_mode == "Calendário",
            )
            slider_start_date = slider_start_dt.date()
            slider_end_date = slider_end_dt.date()

            # Faixas derivadas de cada modo
            cal_range_start = min(start_selected, end_selected)
            cal_range_end = max(start_selected, end_selected)
            bar_range_start = slider_start_date
            bar_range_end = slider_end_date

            # Modo ativo decide qual faixa usar
            if active_mode == "Barra":
                range_start, range_end = bar_range_start, bar_range_end
            else:
                range_start, range_end = cal_range_start, cal_range_end

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
        if "tracks_ms" not in st.session_state:
            st.session_state["tracks_ms"] = list(tracks)
        else:
            existing_tracks = st.session_state["tracks_ms"]
            sanitized_tracks = [t for t in existing_tracks if t in tracks]
            if not sanitized_tracks and existing_tracks:
                sanitized_tracks = list(tracks)
            st.session_state["tracks_ms"] = sanitized_tracks
        sel_tracks = st.multiselect("Pistas", tracks, key="tracks_ms")
    with col_f3:
        if entry_type == "lay":
            bsp_col = "lay_target_bsp"
            base_df_for_bsp = df_filtered[df_filtered["entry_type"] == "lay"]
        elif entry_type == "back":
            bsp_col = "back_target_bsp"
            base_df_for_bsp = df_filtered[df_filtered["entry_type"] == "back"]
        else:
            # ambos: usa faixa unificada
            bsp_col = None
            base_df_for_bsp = df_filtered
        if entry_type == "both":
            if base_df_for_bsp.empty:
                bsp_min, bsp_max = 1.01, 100.0
            elif "lay_target_bsp" not in base_df_for_bsp.columns or "back_target_bsp" not in base_df_for_bsp.columns:
                bsp_min, bsp_max = 1.01, 100.0
            else:
                combined_bsp = base_df_for_bsp[["lay_target_bsp", "back_target_bsp"]]
                if combined_bsp.dropna(how="all").empty:
                    bsp_min, bsp_max = 1.01, 100.0
                else:
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
        # Garante bsp_low/bsp_high existem; sanitiza e sincroniza bsp_slider antes do widget
        if "bsp_low" not in st.session_state or "bsp_high" not in st.session_state:
            st.session_state["bsp_low"] = float(bsp_min)
            st.session_state["bsp_high"] = float(bsp_max)
        low = float(st.session_state["bsp_low"])
        high = float(st.session_state["bsp_high"])
        low = max(bsp_min, min(bsp_max, low))
        high = max(bsp_min, min(bsp_max, high))
        if low > high:
            high = low
        st.session_state["bsp_low"] = low
        st.session_state["bsp_high"] = high
        st.session_state["bsp_slider"] = (low, high)

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
                step=0.01,
                format="%.2f",
                key="bsp_high",
                on_change=_sync_bsp_high,
                label_visibility="collapsed",
            )
            st.caption("BSP max.")

        # Volume total negociado mínimo (dentro de col_f3)
        volume_key = "min_total_volume"
        if volume_key not in st.session_state:
            st.session_state[volume_key] = 2000.0
        vcol, _ = st.columns([3, 7])
        with vcol:
            st.number_input(
                "Volume total negociado mínimo",
                min_value=0.0,
                max_value=1_000_000.0,
                step=100.0,
                format="%.0f",
                key=volume_key,
                help="Considera apenas corridas cuja soma de pptradedvol atinge o mínimo desejado.",
            )

        # (campo movido para a frente do cabecalho de Stake)

    filt = df_filtered.copy()

    min_total_volume = float(st.session_state.get(volume_key, 2000.0))
    volume_series = pd.to_numeric(filt.get("total_matched_volume", pd.Series(dtype=float)), errors="coerce")
    # forecast_odds usa total_matched_volume neutro (0.0); nao aplicar filtro de volume para nao zerar
    if rule != "forecast_odds":
        filt = filt[volume_series.fillna(0.0) >= min_total_volume]

    sel_traps = st.session_state.get("trap_ms", None)
    if sel_traps is not None:
        if sel_traps:
            trap_series_filt = pd.to_numeric(filt.get("trap_number", pd.Series(dtype=float)), errors="coerce")
            trap_series_filt = trap_series_filt.astype("Int64")
            filt = filt[trap_series_filt.isin(sel_traps)]
        else:
            filt = filt.iloc[0:0]

    # Enriquecimento: categoria por corrida (para UI de Categorias/Subcategorias na linha 2)
    if ("category" in filt.columns) and ("category_token" in filt.columns):
        if not filt.empty:
            cat_letters = sorted(
                [c for c in filt["category"].dropna().unique().tolist() if isinstance(c, str) and c]
            )
        else:
            cat_letters = []
    else:
        cat_index = _build_category_index()
        if not filt.empty:
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
            cat_letters = sorted(
                [c for c in filt["category"].dropna().unique().tolist() if isinstance(c, str) and c]
            )
        else:
            cat_letters = []
    # sub_tokens será calculado depois que sel_cats for atualizado
    sub_tokens = []

    def _compute_hour_bucket(series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        minutes = ts.dt.hour * 60 + ts.dt.minute
        bucket = pd.Series(index=series.index, dtype="object")
        bucket[(minutes >= 8 * 60) & (minutes <= 12 * 60)] = "08:00-12:00"
        bucket[(minutes >= 12 * 60 + 1) & (minutes <= 16 * 60)] = "12:01-16:00"
        bucket[(minutes >= 16 * 60 + 1) & (minutes <= 20 * 60)] = "16:01-20:00"
        bucket[(minutes >= 20 * 60 + 1)] = "20:01-23:59"
        return bucket

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if not df_filtered.empty and "race_time_iso" in df_filtered.columns:
            st.caption("Dias da semana")
            tmp_ts = pd.to_datetime(df_filtered["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            wd_series = tmp_ts.dt.weekday.dropna().astype(int)
            wd_unique = sorted(wd_series.unique().tolist())
            wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
            weekday_options = [wd_names[w] for w in wd_unique if w in wd_names]
            if weekday_options:
                wb1, wb2, _ = st.columns([1, 1, 2])
                with wb1:
                    st.button(
                        "Todos",
                        key="weekdays_all",
                        on_click=lambda: st.session_state.update({"weekdays_ms": list(weekday_options)}),
                    )
                with wb2:
                    st.button(
                        "Limpar",
                        key="weekdays_none",
                        on_click=lambda: st.session_state.update({"weekdays_ms": []}),
                    )
                if "weekdays_ms" not in st.session_state:
                    st.session_state["weekdays_ms"] = list(weekday_options)
                else:
                    existing_wd = st.session_state["weekdays_ms"]
                    sanitized_wd = [w for w in existing_wd if w in weekday_options]
                    if not sanitized_wd and existing_wd:
                        sanitized_wd = list(weekday_options)
                    st.session_state["weekdays_ms"] = sanitized_wd
                st.multiselect(
                    "Dias da semana",
                    weekday_options,
                    key="weekdays_ms",
                    label_visibility="collapsed",
                )
                sel_weekdays_nums = [
                    num for num, label in wd_names.items()
                    if label in st.session_state.get("weekdays_ms", [])
                ]
                st.session_state["sel_weekdays"] = sel_weekdays_nums

        if not df_filtered.empty and "race_time_iso" in df_filtered.columns:
            st.caption("Faixa de horário")
            hour_bucket = _compute_hour_bucket(df_filtered["race_time_iso"])
            bucket_order = ["08:00-12:00", "12:01-16:00", "16:01-20:00", "20:01-23:59"]
            bucket_options = [b for b in bucket_order if b in hour_bucket.dropna().unique().tolist()]
            if bucket_options:
                hb1, hb2, _ = st.columns([1, 1, 2])
                with hb1:
                    st.button(
                        "Todos",
                        key="hour_buckets_all",
                        on_click=lambda: st.session_state.update({"hour_bucket_ms": list(bucket_options)}),
                    )
                with hb2:
                    st.button(
                        "Limpar",
                        key="hour_buckets_none",
                        on_click=lambda: st.session_state.update({"hour_bucket_ms": []}),
                    )
                if "hour_bucket_ms" not in st.session_state:
                    st.session_state["hour_bucket_ms"] = list(bucket_options)
                else:
                    existing_hb = st.session_state["hour_bucket_ms"]
                    sanitized_hb = [h for h in existing_hb if h in bucket_options]
                    if not sanitized_hb and existing_hb:
                        sanitized_hb = list(bucket_options)
                    st.session_state["hour_bucket_ms"] = sanitized_hb
                st.multiselect(
                    "Faixa de horário",
                    bucket_options,
                    key="hour_bucket_ms",
                    label_visibility="collapsed",
                )
    with col_s2:
        st.caption("Categorias")
        if cat_letters:
            btn_all, btn_clear, _ = st.columns([1, 1, 6])
            with btn_all:
                st.button(
                    "Todos",
                    key="cats_all",
                    on_click=lambda: st.session_state.update({"cats_ms": cat_letters}),
                )
            with btn_clear:
                st.button(
                    "Limpar",
                    key="cats_none",
                    on_click=lambda: st.session_state.update({"cats_ms": []}),
                )
            if "cats_ms" not in st.session_state:
                st.session_state["cats_ms"] = list(cat_letters)
            else:
                existing_cats = st.session_state["cats_ms"]
                sanitized_cats = [c for c in existing_cats if c in cat_letters]
                if not sanitized_cats and existing_cats:
                    sanitized_cats = list(cat_letters)
                st.session_state["cats_ms"] = sanitized_cats
            sel_cats = st.multiselect(
                "Categoria (A/B/D...)",
                cat_letters,
                key="cats_ms",
                label_visibility="collapsed",
            )
            sel_cats = [c for c in sel_cats if c in cat_letters]
            st.session_state["sel_cats"] = sel_cats

        # Construção de sub_tokens filtrada por categorias selecionadas (DEPOIS que sel_cats foi atualizado)
        if "category_token" in filt.columns and not filt.empty:
            # Usa sel_cats atualizado (ou cats_ms como fallback)
            sel_cats_state = st.session_state.get("sel_cats")
            if not sel_cats_state:
                # fallback para cats_ms, se existir
                sel_cats_state = st.session_state.get("cats_ms")

            token_source = filt

            # Se houver categorias selecionadas e a coluna existir, restringe por elas
            if sel_cats_state and "category" in token_source.columns:
                token_source = token_source[token_source["category"].isin(sel_cats_state)]

            raw_tokens = [t for t in token_source["category_token"].dropna().astype(str).unique().tolist() if isinstance(t, str) and t]

            def _sub_sort_key(tok: str) -> tuple:
                m = re.match(r"^([A-Z]+)(\d+)$", str(tok))
                if m:
                    return (m.group(1), int(m.group(2)))
                m2 = re.match(r"^([A-Z]+)", str(tok))
                return ((m2.group(1) if m2 else str(tok)), 0)

            sub_tokens = sorted(raw_tokens, key=_sub_sort_key)

        st.caption("Subcategorias")
        if sub_tokens:
            sb1, sb2 = st.columns([1, 1])
            with sb1:
                st.button(
                    "Todos",
                    key="subcats_all",
                    on_click=lambda: st.session_state.update({"subcats_ms": sub_tokens}),
                )
            with sb2:
                st.button(
                    "Limpar",
                    key="subcats_none",
                    on_click=lambda: st.session_state.update({"subcats_ms": []}),
                )
            if "subcats_ms" not in st.session_state:
                st.session_state["subcats_ms"] = list(sub_tokens)
            else:
                existing_sc = st.session_state["subcats_ms"]
                sanitized_sc = [t for t in existing_sc if t in sub_tokens]
                if not sanitized_sc and existing_sc:
                    sanitized_sc = list(sub_tokens)
                st.session_state["subcats_ms"] = sanitized_sc
            sel_subcats = st.multiselect(
                "Subcategorias (A1/A2/D1/OR3/...)",
                sub_tokens,
                key="subcats_ms",
                label_visibility="collapsed",
            )
            sel_subcats = [t for t in sel_subcats if t in sub_tokens]
            st.session_state["sel_subcats"] = sel_subcats
    with col_s3:
        sel_weekdays_nums = st.session_state.get("sel_weekdays", None)
        if sel_weekdays_nums is not None:
            if sel_weekdays_nums:
                if "race_time_iso" in filt.columns and not filt.empty:
                    ts_filt = pd.to_datetime(filt["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
                    wd_filt = ts_filt.dt.weekday
                    filt = filt[wd_filt.isin(sel_weekdays_nums)]
            else:
                filt = filt.iloc[0:0]

        sel_hour_buckets = st.session_state.get("hour_bucket_ms", None)
        if sel_hour_buckets is not None:
            if sel_hour_buckets:
                if "race_time_iso" in filt.columns and not filt.empty:
                    hb_filt = _compute_hour_bucket(filt["race_time_iso"])
                    filt = filt[hb_filt.isin(sel_hour_buckets)]
            else:
                filt = filt.iloc[0:0]

        st.caption("Numero de corredores")
        nr_vals = sorted([int(v) for v in pd.to_numeric(filt.get("num_runners", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()])
        if nr_vals:
            btn_all, btn_clear, _ = st.columns([1, 1, 6])
            with btn_all:
                st.button(
                    "Todos",
                    key="nr_all",
                    on_click=lambda: st.session_state.update({"num_runners_ms": nr_vals}),
                )
            with btn_clear:
                st.button(
                    "Limpar",
                    key="nr_none",
                    on_click=lambda: st.session_state.update({"num_runners_ms": []}),
                )
            if "num_runners_ms" not in st.session_state:
                st.session_state["num_runners_ms"] = list(nr_vals)
            else:
                existing_nr = st.session_state["num_runners_ms"]
                sanitized_nr = [v for v in existing_nr if v in nr_vals]
                if not sanitized_nr and existing_nr:
                    sanitized_nr = list(nr_vals)
                st.session_state["num_runners_ms"] = sanitized_nr
            sel_nr = st.multiselect(
                "Numero de corredores",
                nr_vals,
                key="num_runners_ms",
                label_visibility="collapsed",
            )
            sel_nr = [int(v) for v in sel_nr if v in nr_vals]
            st.session_state["sel_num_runners"] = sel_nr
            if sel_nr:
                filt = filt[filt["num_runners"].isin(sel_nr)]
            else:
                filt = filt.iloc[0:0]

        trap_series = pd.to_numeric(df_filtered.get("trap_number", pd.Series(dtype=float)), errors="coerce")
        trap_vals = sorted(trap_series.dropna().astype(int).unique().tolist())
        if trap_vals:
            st.caption("Trap")
            tb1, tb2, _ = st.columns([1, 1, 2])
            with tb1:
                st.button(
                    "Todos",
                    key="traps_all",
                    on_click=lambda: st.session_state.update({"trap_ms": list(trap_vals)}),
                )
            with tb2:
                st.button(
                    "Limpar",
                    key="traps_none",
                    on_click=lambda: st.session_state.update({"trap_ms": []}),
                )
            if "trap_ms" not in st.session_state:
                st.session_state["trap_ms"] = list(trap_vals)
            else:
                existing_traps = st.session_state["trap_ms"]
                sanitized_traps = [t for t in existing_traps if t in trap_vals]
                if not sanitized_traps and existing_traps:
                    sanitized_traps = list(trap_vals)
                st.session_state["trap_ms"] = sanitized_traps
            st.multiselect("Traps", trap_vals, key="trap_ms")

    # Filtro adicional: participacao do lider (somente para regra lider_volume_total)
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
        filt = filt[filt["leader_volume_share_pct"].fillna(0) >= float(leader_min)]

    # Filtros especificos para regra forecast_odds
    if rule == "forecast_odds":
        if "forecast_rank" in filt.columns:
            filt["forecast_rank"] = pd.to_numeric(filt["forecast_rank"], errors="coerce")
        if "value_ratio" in filt.columns:
            filt["value_ratio"] = pd.to_numeric(filt["value_ratio"], errors="coerce")
        st.caption("Filtros Forecast Odds")
        if "forecast_rank" in filt.columns and not filt.empty:
            rank_vals = sorted(pd.to_numeric(filt["forecast_rank"], errors="coerce").dropna().unique().astype(int).tolist())
            if not rank_vals:
                rank_vals = list(range(1, 7))
            if "forecast_rank_ms" not in st.session_state:
                st.session_state["forecast_rank_ms"] = list(rank_vals)
            else:
                existing_ranks = st.session_state["forecast_rank_ms"]
                if not isinstance(existing_ranks, list):
                    existing_ranks = rank_vals
                sanitized_ranks = [r for r in existing_ranks if r in rank_vals]
                if not sanitized_ranks and existing_ranks:
                    sanitized_ranks = list(rank_vals)
                st.session_state["forecast_rank_ms"] = sanitized_ranks
            col_rank, _ = st.columns([2, 8])
            with col_rank:
                sel_ranks = st.multiselect(
                    "Forecast rank",
                    options=rank_vals,
                    key="forecast_rank_ms",
                )
            if sel_ranks:
                filt = filt[filt["forecast_rank"].isin(sel_ranks)]
        if "value_ratio" in filt.columns and not filt.empty:
            vr = pd.to_numeric(filt["value_ratio"], errors="coerce").fillna(float("nan"))
            vmin = float(vr.min()) if vr.notna().any() else 0.0
            vmax = float(vr.max()) if vr.notna().any() else 1.0
            if vmin >= vmax:
                vmin, vmax = 0.0, max(1.0, vmax)
            # Session state como unica fonte; clamp ao intervalo atual
            if "value_ratio_min" not in st.session_state:
                st.session_state["value_ratio_min"] = float(vmin)
            if "value_ratio_max" not in st.session_state:
                st.session_state["value_ratio_max"] = float(vmax)
            vr_low = max(vmin, min(vmax, float(st.session_state["value_ratio_min"])))
            vr_high = max(vmin, min(vmax, float(st.session_state["value_ratio_max"])))
            if vr_low > vr_high:
                vr_high = vr_low
            st.session_state["value_ratio_min"] = vr_low
            st.session_state["value_ratio_max"] = vr_high
            if "only_value_bets" not in st.session_state:
                st.session_state["only_value_bets"] = False
            col_vr_wrapper, _ = st.columns([2, 8])
            with col_vr_wrapper:
                col_vr1, col_vr2 = st.columns(2)
                with col_vr1:
                    value_ratio_min = st.number_input(
                        "Value ratio min",
                        min_value=vmin,
                        max_value=vmax,
                        step=0.05,
                        key="value_ratio_min",
                    )
                with col_vr2:
                    value_ratio_max = st.number_input(
                        "Value ratio max",
                        min_value=vmin,
                        max_value=vmax,
                        step=0.05,
                        key="value_ratio_max",
                    )
            filt = filt[filt["value_ratio"].fillna(float("nan")).between(value_ratio_min, value_ratio_max)]
            only_value_bets = st.checkbox(
                "Somente value bets (value_ratio >= 1.0)",
                key="only_value_bets",
            )
            if only_value_bets:
                filt = filt[filt["value_ratio"].fillna(0.0) >= 1.0]

    if sel_tracks:
        filt = filt[filt["track_name"].isin(sel_tracks)]
    # Regra principal terceiro_queda50: diferenca > 50% em relacao ao vol3
    if rule == "terceiro_queda50":
        filt = filt[filt["pct_diff_second_vs_third"].fillna(0) > 50.0]
    # Aplica filtro por BSP com inputs precisos (respeitando limites)
    bsp_low = float(st.session_state.get("bsp_low", bsp_min))
    bsp_high = float(st.session_state.get("bsp_high", bsp_max))
    if entry_type == "both":
        filt = filt[
            ((filt["entry_type"] == "lay") & (filt["lay_target_bsp"].between(bsp_low, bsp_high))) |
            ((filt["entry_type"] == "back") & (filt["back_target_bsp"].between(bsp_low, bsp_high)))
        ]
    else:
        current_bsp_col = "lay_target_bsp" if entry_type == "lay" else "back_target_bsp"
        filt = filt[(filt[current_bsp_col] >= bsp_low) & (filt[current_bsp_col] <= bsp_high)]

    if cat_letters:
        sel_cats = st.session_state.get("sel_cats", [])
        if sel_cats:
            filt = filt[filt["category"].isin(sel_cats)]
        else:
            filt = filt.iloc[0:0]

    if sub_tokens:
        sel_subcats = st.session_state.get("sel_subcats", [])
        if sel_subcats:
            filt = filt[filt["category_token"].isin(sel_subcats)]
        else:
            filt = filt.iloc[0:0]

    # Filtro por tipo de entrada: Back -> apenas entry_type='back'; Lay -> apenas entry_type='lay';
    # Ambos -> mantem todas as linhas (entry_type IN ('back','lay')). Nao assume duas linhas por galgo/corrida.
    if entry_type != "both":
        filt = filt[filt["entry_type"] == entry_type]

    # Seletor global do eixo X para graficos de evolucao
    x_axis_mode = st.radio(
        "Eixo X dos graficos de evolucao",
        ["Dia", "Bet"],
        index=0,
        horizontal=True,
        help="Altere entre datas ou sequencia de apostas",
    )

    def render_block(title_suffix: str, df_block: pd.DataFrame, entry_kind: str) -> None:
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        summary = _compute_summary_metrics(df_block, entry_kind, base_amount)

        st.subheader(title_suffix)

        # Linha 1: Pistas, Sinais, Greens, Reds, Media BSP, Assertividade
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Pistas", summary["tracks"])
        with c2:
            st.metric("Sinais", summary["signals"])
        num_greens = summary["greens"]
        num_reds = summary["reds"]
        with c3:
            st.metric("Greens", num_greens)
        with c4:
            st.metric("Reds", num_reds)
        with c5:
            st.metric("Media BSP Alvo", f"{summary['avg_bsp']:.2f}")
        with c6:
            st.metric("Assertividade", f"{summary['accuracy']:.2%}")

        # Campo valor base (reutiliza ja existente)
        total_base_stake = summary["base_stake"]
        total_pnl_stake = summary["pnl_stake"]
        roi_stake = summary["roi_stake"]

        # Linha 2: Stake(10)
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

        # Linha 3: Liability(10)  apenas para LAY
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

        # Graficos de evolucao
        plot = df_block.copy()
        if not plot.empty:
            plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            plot = plot.dropna(subset=["ts"]).sort_values("ts")
            plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
            plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")

            if x_axis_mode == "Dia":
                plot["date_only"] = plot["ts"].dt.date
                daily = plot.groupby("date_only", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum().sort_values("date_only")
                daily["cum_stake"] = (daily["_pnl_stake"] * scale_factor).cumsum()
                daily["cum_liab"] = (daily["_pnl_liab"] * scale_factor).cumsum()
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

                # --- Drawdown intradiário por dia (Stake) ---
                plot_intraday = plot.copy()
                plot_intraday = plot_intraday.sort_values("ts").copy()
                plot_intraday["date_only"] = plot_intraday["ts"].dt.date
                plot_intraday["_pnl_stake_scaled"] = plot_intraday["_pnl_stake"] * scale_factor
                plot_intraday["intraday_cum_stake"] = (
                    plot_intraday.groupby("date_only")["_pnl_stake_scaled"].cumsum()
                )
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

                with st.expander("Drawdown Stake por dia (mínimo intradiário)", expanded=False):
                    zero_line_dd = (
                        alt.Chart(pd.DataFrame({"y": [0]}))
                        .mark_rule(color="red", strokeWidth=1)
                        .encode(y="y:Q")
                    )
                    min_line = (
                        alt.Chart(stake_dd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y(
                                "min_intraday_stake:Q",
                                title="Pior nível intradiário do dia",
                            ),
                        )
                    )
                    close_line = (
                        alt.Chart(stake_dd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("close_stake:Q"),
                            color=alt.value("#06D6A0"),
                        )
                    )
                    st.altair_chart(
                        alt.layer(zero_line_dd, min_line, close_line).configure_view(stroke="#888", strokeWidth=1),
                        use_container_width=True,
                    )

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

                    plot_intraday["_pnl_liab_scaled"] = plot_intraday["_pnl_liab"] * scale_factor
                    plot_intraday["intraday_cum_liab"] = (
                        plot_intraday.groupby("date_only")["_pnl_liab_scaled"].cumsum()
                    )
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

                    with st.expander("Drawdown Liability por dia (mínimo intradiário)", expanded=False):
                        zero_line_dd2 = (
                            alt.Chart(pd.DataFrame({"y": [0]}))
                            .mark_rule(color="red", strokeWidth=1)
                            .encode(y="y:Q")
                        )
                        min_line_liab = (
                            alt.Chart(liab_dd)
                            .mark_line()
                            .encode(
                                x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                                y=alt.Y(
                                    "min_intraday_liab:Q",
                                    title="Pior nível intradiário do dia",
                                ),
                            )
                        )
                        close_line_liab = (
                            alt.Chart(liab_dd)
                            .mark_line()
                            .encode(
                                x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                                y=alt.Y("close_liab:Q"),
                                color=alt.value("#FDE74C"),
                            )
                        )
                        st.altair_chart(
                            alt.layer(zero_line_dd2, min_line_liab, close_line_liab).configure_view(stroke="#888", strokeWidth=1),
                            use_container_width=True,
                        )
            else:
                # Evolucao por sequencia de apostas (ordem temporal)
                plot["bet_idx"] = range(1, len(plot) + 1)
                plot["cum_stake"] = (plot["_pnl_stake"] * scale_factor).cumsum()
                if entry_kind == "lay":
                    plot["cum_liab"] = (plot["_pnl_liab"] * scale_factor).cumsum()
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

        # Desempenho por dia da semana (antes da tabela)
        _render_weekday_perf(df_block, entry_kind)
        _render_trap_perf(df_block, entry_kind)
        _render_hour_bucket_perf(df_block, entry_kind)
        _render_forecast_rank_perf(df_block, entry_kind)

        # Relatorio mensal (entre desempenho semanal e tabela)
        _render_monthly_table(df_block, entry_kind)

        # Tabela
        def _pref(ref_name: str, legacy_name: str) -> str:
            return ref_name if ref_name in df_block.columns else legacy_name

        _volume_cols = [
            "total_matched_volume", "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume", "pct_diff_second_vs_third",
        ]
        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "num_runners", "tf_top1", "tf_top2", "tf_top3",
        ]
        if rule != "forecast_odds":
            show_cols += _volume_cols
        if rule == "forecast_odds":
            show_cols += ["forecast_rank", "forecast_odds", "forecast_name_clean", "value_ratio", "value_log"]
        stake_col = _pref("stake_ref", "stake_fixed_10")
        pnl_stake_col = _pref("pnl_stake_ref", "pnl_stake_fixed_10")
        roi_stake_col = _pref("roi_row_stake_ref", "roi_row_stake_fixed_10")
        if entry_kind == "lay":
            liab_from_stake_col = _pref("liability_from_stake_ref", "liability_from_stake_fixed_10")
            stake_for_liab_col = _pref("stake_for_liability_ref", "stake_for_liability_10")
            liab_col = _pref("liability_ref", "liability_fixed_10")
            pnl_liab_col = _pref("pnl_liability_ref", "pnl_liability_fixed_10")
            roi_liab_col = _pref("roi_row_liability_ref", "roi_row_liability_fixed_10")
            roi_expo_col = _pref("roi_row_exposure_ref", "roi_row_exposure_fixed_10")
            show_cols += [
                "lay_target_name", "lay_target_bsp",
                stake_col, liab_from_stake_col,
                stake_for_liab_col, liab_col,
                "win_lose", "is_green", pnl_stake_col, pnl_liab_col,
                roi_stake_col, roi_liab_col, roi_expo_col,
            ]
        else:
            show_cols += [
                "back_target_name", "back_target_bsp",
                stake_col,
                "win_lose", "is_green", pnl_stake_col,
                roi_stake_col,
            ]
        missing = [c for c in show_cols if c not in df_block.columns]
        for c in missing:
            df_block[c] = ""
        table_label = f"Tabela {title_suffix}"
        with st.expander(table_label, expanded=False):
            st.dataframe(df_block[show_cols], use_container_width=True)

    def _render_monthly_table(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Tabela mensal com mesmas métricas do cabeçalho."""
        base_amount = float(st.session_state.get("base_amount", 1.0))
        working = df_block.copy()
        if "race_ts" not in working.columns:
            working["race_ts"] = pd.to_datetime(working["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
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
            monthly_df = monthly_df.sort_values("_period")
            # ROI mensal já presente (stake/liab). Agora adicionamos tendências.
            monthly_df["base_cum"] = monthly_df["Base (Stake)"].cumsum()
            monthly_df["pnl_cum"] = monthly_df["PnL Stake"].cumsum()
            monthly_df["ROI Stake (acum)"] = monthly_df.apply(
                lambda r: (r["pnl_cum"] / r["base_cum"]) if r["base_cum"] > 0 else 0.0,
                axis=1,
            )
            monthly_df["base_3m"] = monthly_df["Base (Stake)"].rolling(window=3, min_periods=1).sum()
            monthly_df["pnl_3m"] = monthly_df["PnL Stake"].rolling(window=3, min_periods=1).sum()
            monthly_df["ROI Stake (3M)"] = monthly_df.apply(
                lambda r: (r["pnl_3m"] / r["base_3m"]) if r["base_3m"] > 0 else 0.0,
                axis=1,
            )
            if entry_kind == "lay" and "Stake (Liability)" in monthly_df.columns:
                monthly_df["base_liab_cum"] = monthly_df["Stake (Liability)"].cumsum()
                monthly_df["pnl_liab_cum"] = monthly_df["PnL Liability"].cumsum()
                monthly_df["ROI Liability (acum)"] = monthly_df.apply(
                    lambda r: (r["pnl_liab_cum"] / r["base_liab_cum"]) if r["base_liab_cum"] > 0 else 0.0,
                    axis=1,
                )
                monthly_df["base_liab_3m"] = monthly_df["Stake (Liability)"].rolling(window=3, min_periods=1).sum()
                monthly_df["pnl_liab_3m"] = monthly_df["PnL Liability"].rolling(window=3, min_periods=1).sum()
                monthly_df["ROI Liability (3M)"] = monthly_df.apply(
                    lambda r: (r["pnl_liab_3m"] / r["base_liab_3m"]) if r["base_liab_3m"] > 0 else 0.0,
                    axis=1,
                )
            monthly_df = monthly_df.drop(columns=["_period"], errors="ignore")
            # Nomes mais claros para colunas de tendencia
            monthly_df = monthly_df.rename(columns={
                "base_cum": "Base acum.",
                "pnl_cum": "PnL acum.",
                "ROI Stake (acum)": "ROI acum.",
                "base_3m": "Base (3M)",
                "pnl_3m": "PnL (3M)",
                "ROI Stake (3M)": "ROI (3M)",
                "base_liab_cum": "Base acum. (Liab)",
                "pnl_liab_cum": "PnL acum. (Liab)",
                "ROI Liability (acum)": "ROI acum. (Liab)",
                "base_liab_3m": "Base (3M Liab)",
                "pnl_liab_3m": "PnL (3M Liab)",
                "ROI Liability (3M)": "ROI (3M Liab)",
            })
        month_order = monthly_df["Mes/Ano"].tolist() if not monthly_df.empty else []
        with st.expander(f"Relatorio mensal ({entry_kind.upper()})", expanded=False):
            _fmt_monthly = {}
            for c in monthly_df.columns:
                if c == "Mes/Ano":
                    continue
                if c in ("Assertividade", "ROI Stake", "ROI acum.", "ROI (3M)") or "ROI" in c:
                    _fmt_monthly[c] = "{:.2%}"
                elif c in ("Pistas", "Sinais", "Greens", "Reds"):
                    _fmt_monthly[c] = "{:.0f}"
                else:
                    _fmt_monthly[c] = "{:.2f}"
            st.dataframe(monthly_df.style.format(_fmt_monthly), use_container_width=True)
            chart_data = working.copy()
            chart_data["date_only"] = chart_data["race_ts"].dt.date
            chart_data["_pnl_stake"] = get_col(chart_data, "pnl_stake_ref", "pnl_stake_fixed_10")
            daily_raw = (
                chart_data.groupby(["month_label", "month_period", "date_only"], as_index=False)[["_pnl_stake"]]
                .sum()
                .sort_values("date_only")
            )
            # Garante que cada mes apareca na janela inteira (preenche dias sem apostas com 0)
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
            if daily.empty:
                st.info("Sem dados suficientes para gerar os gráficos mensais.")
            else:
                local_scale = get_scale(base_amount, _REF_FACTOR)
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
                st.markdown("**Evolução mensal (PnL acumulado por dia)**")
                st.altair_chart(month_chart.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # Campo base (único) pos-filtros, antes de renderizar resultados
    _render_base_amount_input()

    # Enriquecimento feito antes; agora renderizacao por bloco
    if entry_type == "both":
        render_block("Resultados BACK", filt[filt["entry_type"] == "back"], "back")
        render_block("Resultados LAY", filt[filt["entry_type"] == "lay"], "lay")
    else:
        # Evita duplicacao quando selecionado apenas um tipo: usar apenas os paineis agregados abaixo
        pass

    # Quando apenas um tipo e selecionado, exibimos o agregado desse tipo.
    # Quando "both", os blocos BACK e LAY ja foram exibidos acima; evitamos repetir dados agregados.
    if entry_type != "both":
        summary = _compute_summary_metrics(filt, entry_type, float(st.session_state.get("base_amount", 1.0)))
        # Linha 1: Pistas, Sinais, Greens, Reds, Media BSP, Assertividade
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

        # Recalcula fatores apos possivel alteracao do input
        base_amount = float(st.session_state.get("base_amount", 1.0))
        scale_factor = get_scale(base_amount, _REF_FACTOR)
        summary = _compute_summary_metrics(filt, entry_type, base_amount)
        total_base_stake = summary["base_stake"]
        total_pnl_stake = summary["pnl_stake"]
        roi_stake = summary["roi_stake"]
        drawdown_stake = summary["drawdown_stake"]

        # Linha 2: Stake(10)
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
        min_pnl_global = summary["min_pnl_stake"]
        with s4:
            st.metric("Menor PnL acumulado", f"{min_pnl_global:.2f}")
        with s5:
            st.metric("Drawdown máximo (Stake)", f"{drawdown_stake:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Linha 3: Liability(10)  apenas para LAY
        if entry_type == "lay":
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

        # Graficos de evolucao (por dia ou por bet) com opcao de minimizar
        plot = filt.copy()
        if not plot.empty:
            plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
            plot = plot.dropna(subset=["ts"]).sort_values("ts")
            # Sempre calcula agregacao diaria para uso em graficos semanais, mesmo quando o eixo X e "Bet"
            plot["date_only"] = plot["ts"].dt.date
            plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
            plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
            daily = plot.groupby("date_only", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum().sort_values("date_only")
            daily["cum_stake"] = (daily["_pnl_stake"] * scale_factor).cumsum()
            daily["cum_liab"] = (daily["_pnl_liab"] * scale_factor).cumsum()

            if x_axis_mode == "Dia":
                plot["date_only"] = plot["ts"].dt.date
                daily = plot.groupby("date_only", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum().sort_values("date_only")
                daily["cum_stake"] = (daily["_pnl_stake"] * scale_factor).cumsum()
                daily["cum_liab"] = (daily["_pnl_liab"] * scale_factor).cumsum()

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

                # --- Drawdown intradiário por dia (Stake) ---
                plot_intraday = plot.copy()
                plot_intraday = plot_intraday.sort_values("ts").copy()
                plot_intraday["date_only"] = plot_intraday["ts"].dt.date
                plot_intraday["_pnl_stake_scaled"] = plot_intraday["_pnl_stake"] * scale_factor
                plot_intraday["intraday_cum_stake"] = (
                    plot_intraday.groupby("date_only")["_pnl_stake_scaled"].cumsum()
                )
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

                with st.expander("Drawdown Stake por dia (mínimo intradiário)", expanded=False):
                    zero_line_dd = (
                        alt.Chart(pd.DataFrame({"y": [0]}))
                        .mark_rule(color="red", strokeWidth=1)
                        .encode(y="y:Q")
                    )
                    min_line = (
                        alt.Chart(stake_dd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y(
                                "min_intraday_stake:Q",
                                title="Pior nível intradiário do dia",
                            ),
                        )
                    )
                    close_line = (
                        alt.Chart(stake_dd)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("close_stake:Q"),
                            color=alt.value("#06D6A0"),
                        )
                    )
                    st.altair_chart(
                        alt.layer(zero_line_dd, min_line, close_line).configure_view(stroke="#888", strokeWidth=1),
                        use_container_width=True,
                    )

                if entry_type == "lay":
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

                    plot_intraday["_pnl_liab_scaled"] = plot_intraday["_pnl_liab"] * scale_factor
                    plot_intraday["intraday_cum_liab"] = (
                        plot_intraday.groupby("date_only")["_pnl_liab_scaled"].cumsum()
                    )
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

                    with st.expander("Drawdown Liability por dia (mínimo intradiário)", expanded=False):
                        zero_line_dd2 = (
                            alt.Chart(pd.DataFrame({"y": [0]}))
                            .mark_rule(color="red", strokeWidth=1)
                            .encode(y="y:Q")
                        )
                        min_line_liab = (
                            alt.Chart(liab_dd)
                            .mark_line()
                            .encode(
                                x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                                y=alt.Y(
                                    "min_intraday_liab:Q",
                                    title="Pior nível intradiário do dia",
                                ),
                            )
                        )
                        close_line_liab = (
                            alt.Chart(liab_dd)
                            .mark_line()
                            .encode(
                                x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                                y=alt.Y("close_liab:Q"),
                                color=alt.value("#FDE74C"),
                            )
                        )
                        st.altair_chart(
                            alt.layer(zero_line_dd2, min_line_liab, close_line_liab).configure_view(stroke="#888", strokeWidth=1),
                            use_container_width=True,
                        )
            else:
                # Evolucao por sequencia de apostas (ordem temporal)
                plot["bet_idx"] = range(1, len(plot) + 1)
                plot["cum_stake"] = (plot["_pnl_stake"] * scale_factor).cumsum()
                plot["cum_liab"] = (plot["_pnl_liab"] * scale_factor).cumsum()

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

                if entry_type == "lay":
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

        # Desempenho por dia da semana (antes da tabela)
        _render_weekday_perf(filt, entry_type)
        _render_trap_perf(filt, entry_type)
        _render_hour_bucket_perf(filt, entry_type)
        _render_forecast_rank_perf(filt, entry_type)

        # Relatorio mensal (entre desempenho semanal e tabela)
        _render_monthly_table(filt, entry_type)

        # Tabela (agregada para o tipo selecionado)
        def _pref(ref_name: str, legacy_name: str) -> str:
            return ref_name if ref_name in filt.columns else legacy_name

        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "num_runners", "total_matched_volume",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
            "lay_target_name", "lay_target_bsp",
            _pref("stake_ref", "stake_fixed_10"), _pref("liability_from_stake_ref", "liability_from_stake_fixed_10"),
            _pref("stake_for_liability_ref", "stake_for_liability_10"), _pref("liability_ref", "liability_fixed_10"),
            "win_lose", "is_green", _pref("pnl_stake_ref", "pnl_stake_fixed_10"), _pref("pnl_liability_ref", "pnl_liability_fixed_10"),
            _pref("roi_row_stake_ref", "roi_row_stake_fixed_10"), _pref("roi_row_liability_ref", "roi_row_liability_fixed_10"),
            _pref("roi_row_exposure_ref", "roi_row_exposure_fixed_10"),
            "pct_diff_second_vs_third",
        ]
        missing = [c for c in show_cols if c not in filt.columns]
        for c in missing:
            filt[c] = ""

        _vol_cols = [
            "total_matched_volume", "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume", "pct_diff_second_vs_third",
        ]
        if entry_type == "back":
            show_cols = [
                "date", "track_name", "category_token", "race_time_iso",
                "num_runners", "tf_top1", "tf_top2", "tf_top3",
            ]
            if rule != "forecast_odds":
                show_cols += _vol_cols
            show_cols += [
                "back_target_name", "back_target_bsp",
                _pref("stake_ref", "stake_fixed_10"),
                "win_lose", "is_green", _pref("pnl_stake_ref", "pnl_stake_fixed_10"),
                _pref("roi_row_stake_ref", "roi_row_stake_fixed_10"),
            ]
            if rule == "forecast_odds":
                show_cols += ["forecast_rank", "forecast_odds", "forecast_name_clean", "value_ratio", "value_log"]
        else:
            show_cols = [
                "date", "track_name", "category_token", "race_time_iso",
                "num_runners", "tf_top1", "tf_top2", "tf_top3",
            ]
            if rule != "forecast_odds":
                show_cols += _vol_cols
            show_cols += [
                "lay_target_name", "lay_target_bsp",
                _pref("stake_ref", "stake_fixed_10"), _pref("liability_from_stake_ref", "liability_from_stake_fixed_10"),
                _pref("stake_for_liability_ref", "stake_for_liability_10"), _pref("liability_ref", "liability_fixed_10"),
                "win_lose", "is_green", _pref("pnl_stake_ref", "pnl_stake_fixed_10"), _pref("pnl_liability_ref", "pnl_liability_fixed_10"),
                _pref("roi_row_stake_ref", "roi_row_stake_fixed_10"), _pref("roi_row_liability_ref", "roi_row_liability_fixed_10"),
                _pref("roi_row_exposure_ref", "roi_row_exposure_fixed_10"),
            ]
            if rule == "forecast_odds":
                show_cols += ["forecast_rank", "forecast_odds", "forecast_name_clean", "value_ratio", "value_log"]

        missing = [c for c in show_cols if c not in filt.columns]
        for c in missing:
            filt[c] = ""
        table_label = f"Tabela de resultados ({entry_type.upper()})"
        with st.expander(table_label, expanded=False):
            st.dataframe(filt[show_cols], use_container_width=True)

    # Graficos pequenos: evolucao por pista e por categoria (um bloco por tipo de entrada)
    def _render_small_charts(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot2 = df_block.copy()
        if plot2.empty:
            return
        plot2["ts"] = pd.to_datetime(plot2["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        plot2 = plot2.dropna(subset=["ts"]).sort_values("ts")
        plot2["date_only"] = plot2["ts"].dt.date
        local_scale = get_scale(float(st.session_state.get("base_amount", 1.0)), _REF_FACTOR)
        plot2["_pnl_stake"] = get_col(plot2, "pnl_stake_ref", "pnl_stake_fixed_10")
        pnl_stake_col = "_pnl_stake"

        # Por pista
        if x_axis_mode == "Dia":
            td = plot2.groupby(["track_name", "date_only"], as_index=False)[["_pnl_stake"]].sum().sort_values("date_only")
            if not td.empty:
                td["cum"] = td.groupby("track_name")["_pnl_stake"].cumsum() * local_scale
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
            # Evolucao por sequencia de apostas dentro de cada pista
            plot2["bet_idx"] = plot2.groupby("track_name").cumcount() + 1
            plot2["cum"] = plot2.groupby("track_name")["_pnl_stake"].cumsum() * local_scale
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

        # Por categoria (apenas letras A/B/D/OR/IV...) sem subcategorias)
        if "category" in plot2.columns:
            cd = plot2[plot2["category"].astype(str).str.len() > 0].copy()
            if not cd.empty:
                cat_counts = cd.groupby("category", as_index=False).size().rename(columns={"size": "count"})
                if x_axis_mode == "Dia":
                    cd = cd.groupby(["category", "date_only"], as_index=False)[["_pnl_stake"]].sum().sort_values("date_only")
                    cd = cd.merge(cat_counts, on="category", how="left")
                    cd["count"] = cd["count"].fillna(0).astype(int)
                    cd["cum"] = cd.groupby("category")["_pnl_stake"].cumsum() * local_scale
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
                    # Evolucao por sequencia de apostas dentro de cada categoria
                    cd["bet_idx"] = cd.groupby("category").cumcount() + 1
                    cd["cum"] = cd.groupby("category")["_pnl_stake"].cumsum() * local_scale
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

        # Por subcategoria (tokens completos A1/A2/D1/OR3/...)
        if "category_token" in plot2.columns:
            sd = plot2[plot2["category_token"].astype(str).str.len() > 0].copy()
            if not sd.empty:
                sub_counts = sd.groupby("category_token", as_index=False).size().rename(columns={"size": "count"})
                # Ordenacao numerica de subcategorias (A1, A2, ..., A10)
                def _subkey(x: str) -> tuple[str, int]:
                    m = re.match(r"^([A-Z]+)(\d+)$", str(x))
                    if m:
                        return (m.group(1), int(m.group(2)))
                    m2 = re.match(r"^([A-Z]+)", str(x))
                    return ((m2.group(1) if m2 else str(x)), 0)
                ordered_tokens = sorted(sub_counts["category_token"].astype(str).unique().tolist(), key=_subkey)
                if x_axis_mode == "Dia":
                    sd = sd.groupby(["category_token", "date_only"], as_index=False)[[pnl_stake_col]].sum().sort_values("date_only")
                    sd = sd.merge(sub_counts, on="category_token", how="left")
                    sd["count"] = sd["count"].fillna(0).astype(int)
                    sd["cum"] = sd.groupby("category_token")[pnl_stake_col].cumsum() * local_scale
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
                    # label com contagem
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
                    sd["cum"] = sd.groupby("category_token")[pnl_stake_col].cumsum() * local_scale
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

        # Subcategorias por pista (versao simples + contador em cada facet)
        if "category_token" in plot2.columns:
            sp = plot2[plot2["category_token"].astype(str).str.len() > 0].copy()
            if not sp.empty:
                top_k = 12
                track_sizes = sp.groupby("track_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
                top_tracks = set(track_sizes.head(top_k)["track_name"].tolist())
                sp = sp[sp["track_name"].isin(top_tracks)]
                sp = sp.merge(track_sizes[["track_name", "count"]].rename(columns={"count": "track_total"}), on="track_name", how="left")
                sp["track_total"] = sp["track_total"].fillna(0).astype(int)
                sp["track_title"] = sp["track_name"].astype(str) + " (" + sp["track_total"].astype(str) + ")"

                st.subheader(f"Subcategorias por pista (PnL acumulado) - {entry_kind.upper()}")
                if x_axis_mode == "Dia":
                    gp = sp.groupby(["track_name", "category_token", "date_only"], as_index=False)[[pnl_stake_col]].sum().sort_values("date_only")
                    gp = gp.merge(sp[["track_name", "track_title"]].drop_duplicates(), on="track_name", how="left")
                    # contador por celula (pista  subcategoria)
                    cell_counts = sp.groupby(["track_name", "category_token"], as_index=False).size().rename(columns={"size": "cell_count"})
                    gp = gp.merge(cell_counts, on=["track_name", "category_token"], how="left")
                    gp["cum"] = gp.groupby(["track_name", "category_token"])[pnl_stake_col].cumsum() * local_scale
                    base_sp = (
                        alt.Chart(gp)
                        .mark_line()
                        .encode(
                            x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    count_text_sp = (
                        alt.Chart(gp)
                        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
                        .encode(text=alt.Text("cell_count:Q", format=".0f"))
                    )
                else:
                    sp["bet_idx"] = sp.groupby(["track_name", "category_token"]).cumcount() + 1
                    sp["cum"] = sp.groupby(["track_name", "category_token"])[pnl_stake_col].cumsum() * local_scale
                    gp = sp[["track_name", "track_title", "category_token", "bet_idx", "cum"]].copy()
                    cell_counts = sp.groupby(["track_name", "category_token"], as_index=False).size().rename(columns={"size": "cell_count"})
                    gp = gp.merge(cell_counts, on=["track_name", "category_token"], how="left")
                    base_sp = (
                        alt.Chart(gp)
                        .mark_line()
                        .encode(
                            x=alt.X("bet_idx:Q", title="Bet #"),
                            y=alt.Y("cum:Q", title="PnL"),
                        )
                        .properties(width=small_width, height=small_height)
                    )
                    count_text_sp = (
                        alt.Chart(gp)
                        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
                        .encode(text=alt.Text("cell_count:Q", format=".0f"))
                    )
                zero_line_sp = alt.Chart(gp).mark_rule(color="red", strokeWidth=1).encode(y=alt.datum(0))
                chart_sp = alt.layer(zero_line_sp, base_sp, count_text_sp).facet(
                    row=alt.Facet("category_token:N", header=alt.Header(title="Subcategoria")),
                    column=alt.Facet("track_title:N", header=alt.Header(title="Pista")),
                )
                st.altair_chart(chart_sp.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
        # Por numero de corredores
        if "num_runners" in plot2.columns:
            nd = plot2.dropna(subset=["num_runners"]).copy()
            if not nd.empty:
                nd["num_runners"] = pd.to_numeric(nd["num_runners"], errors="coerce").astype("Int64")
                nr_counts = nd.groupby("num_runners", as_index=False).size().rename(columns={"size": "count"})
                if x_axis_mode == "Dia":
                    nd = nd.groupby(["num_runners", "date_only"], as_index=False)[[pnl_stake_col]].sum().sort_values("date_only")
                    nd = nd.merge(nr_counts, on="num_runners", how="left")
                    nd["count"] = nd["count"].fillna(0).astype(int)
                    nd["cum"] = nd.groupby("num_runners")[pnl_stake_col].cumsum() * local_scale
                    nd["nr_title"] = nd["num_runners"].astype(str) + " (" + nd["count"].astype(str) + ")"
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
                        facet=alt.Facet("nr_title:N", sort=alt.SortField(field="num_runners", order="ascending"), header=alt.Header(title="")),
                        columns=4,
                    )
                    st.altair_chart(chart_nr.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
                else:
                    nd["bet_idx"] = nd.groupby("num_runners").cumcount() + 1
                    nd["cum"] = nd.groupby("num_runners")[pnl_stake_col].cumsum() * local_scale
                    nd = nd.merge(nr_counts, on="num_runners", how="left")
                    nd["count"] = nd["count"].fillna(0).astype(int)
                    nd["nr_title"] = nd["num_runners"].astype(str) + " (" + nd["count"].astype(str) + ")"
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
                        facet=alt.Facet("nr_title:N", sort=alt.SortField(field="num_runners", order="ascending"), header=alt.Header(title="")),
                        columns=4,
                    )
                    st.altair_chart(chart_nr.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    def _render_weekday_perf(df_block: pd.DataFrame, entry_kind: str) -> None:
        """Barra por dia da semana com linha zero."""
        if df_block.empty or "race_time_iso" not in df_block.columns:
            return
        base_amount = float(st.session_state.get("base_amount", 1.0))
        ref_factor = _REF_FACTOR
        scale_factor = get_scale(base_amount, ref_factor)
        plot = df_block.copy()
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        if plot.empty:
            return
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        plot["_pnl_liab"] = get_col(plot, "pnl_liability_ref", "pnl_liability_fixed_10")
        plot["date_only"] = plot["ts"].dt.date
        daily = (
            plot.groupby("date_only", as_index=False)[["_pnl_stake", "_pnl_liab"]]
            .sum()
            .sort_values("date_only")
        )
        daily["weekday"] = pd.to_datetime(daily["date_only"]).dt.weekday
        wd_order = [0, 1, 2, 3, 4, 5, 6]
        wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
        wd_order_names = [wd_names[w] for w in wd_order]
        by_wd = daily.groupby("weekday", as_index=False)[["_pnl_stake", "_pnl_liab"]].sum()
        by_wd["weekday_name"] = by_wd["weekday"].map(wd_names)
        by_wd["weekday_name"] = pd.Categorical(by_wd["weekday_name"], categories=wd_order_names, ordered=True)
        by_wd["pnl_stake"] = by_wd["_pnl_stake"] * scale_factor
        if entry_kind == "lay":
            by_wd["pnl_liab"] = by_wd["_pnl_liab"] * scale_factor

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

    # --- Graficos cruzados: Categoria  Numero de corredores (Evolucao) ---
    def _render_cross_nr_category(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners" not in plot.columns or "category" not in plot.columns:
            return
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = get_scale(float(st.session_state.get("base_amount", 1.0)), _REF_FACTOR)
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        pnl_stake_col = "_pnl_stake"
        # Titulos
        counts = plot.groupby(["category", "num_runners"], as_index=False).size().rename(columns={"size": "count"})
        if x_axis_mode == "Dia":
            agg = plot.groupby(["category", "num_runners", "date_only"], as_index=False)[[pnl_stake_col]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["category", "num_runners"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["category", "num_runners"])[pnl_stake_col].cumsum() * local_scale
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
            plot["bet_idx"] = plot.groupby(["category", "num_runners"]).cumcount() + 1
            plot["cum"] = plot.groupby(["category", "num_runners"])[pnl_stake_col].cumsum() * local_scale
            agg = plot[["category", "num_runners", "bet_idx", "cum"]].copy()
            agg = agg.merge(counts, on=["category", "num_runners"], how="left")
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
        # Rotulo de contagem por celula (count por categoriaN)
        count_text = (
            alt.Chart(agg)
            .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
            .encode(text=alt.Text("count:Q", format=".0f"))
        )
        st.subheader(f"Evolucao por categoria  numero de corredores - {entry_kind.upper()}")
        ch = alt.layer(zero, base, count_text).facet(
            row=alt.Facet("num_runners:O", sort="ascending", header=alt.Header(title="")),
            column=alt.Facet("facet_col:N", header=alt.Header(title="")),
        )
        st.altair_chart(ch.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # --- Graficos cruzados: Pista  Numero de corredores (Evolucao) ---
    def _render_cross_nr_track(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners" not in plot.columns:
            return
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], format=_RACE_TIME_ISO_FORMAT, errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = get_scale(float(st.session_state.get("base_amount", 1.0)), _REF_FACTOR)
        plot["_pnl_stake"] = get_col(plot, "pnl_stake_ref", "pnl_stake_fixed_10")
        pnl_stake_col = "_pnl_stake"
        # Seleciona Top K pistas por quantidade
        top_k = 12
        track_sizes = plot.groupby("track_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        top_tracks = set(track_sizes.head(top_k)["track_name"].tolist())
        plot = plot[plot["track_name"].isin(top_tracks)]
        counts = plot.groupby(["track_name", "num_runners"], as_index=False).size().rename(columns={"size": "count"})
        plot = plot.merge(track_sizes[["track_name", "count"]].rename(columns={"count": "track_total"}), on="track_name", how="left")
        if x_axis_mode == "Dia":
            agg = plot.groupby(["track_name", "num_runners", "date_only"], as_index=False)[[pnl_stake_col]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["track_name", "num_runners"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["track_name", "num_runners"])[pnl_stake_col].cumsum() * local_scale
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
            plot["bet_idx"] = plot.groupby(["track_name", "num_runners"]).cumcount() + 1
            plot["cum"] = plot.groupby(["track_name", "num_runners"])[pnl_stake_col].cumsum() * local_scale
            agg = plot[["track_name", "num_runners", "bet_idx", "cum"]].copy()
            agg = agg.merge(counts, on=["track_name", "num_runners"], how="left")
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
        # Rotulo de contagem por celula (count por pistaN)
        count_text = (
            alt.Chart(agg)
            .mark_text(align="left", baseline="top", dx=4, dy=4, color="#AAAAAA", fontSize=10)
            .encode(text=alt.Text("count:Q", format=".0f"))
        )
        st.subheader(f"Evolucao por pista  numero de corredores - {entry_kind.upper()}")
        ch = alt.layer(zero, base, count_text).facet(
            row=alt.Facet("num_runners:O", sort="ascending", header=alt.Header(title="")),
            column=alt.Facet("facet_col:N", header=alt.Header(title="")),
        )
        st.altair_chart(ch.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # Renderizacao dos blocos na ordem BACK depois LAY, agrupando cross charts junto
    if entry_type == "both":
        _render_small_charts(filt[filt["entry_type"] == "back"], "back")
        _render_cross_nr_category(filt[filt["entry_type"] == "back"], "back")
        _render_cross_nr_track(filt[filt["entry_type"] == "back"], "back")
        _render_small_charts(filt[filt["entry_type"] == "lay"], "lay")
        _render_cross_nr_category(filt[filt["entry_type"] == "lay"], "lay")
        _render_cross_nr_track(filt[filt["entry_type"] == "lay"], "lay")
    else:
        _render_small_charts(filt[filt["entry_type"] == entry_type], entry_type)
        _render_cross_nr_category(filt[filt["entry_type"] == entry_type], entry_type)
        _render_cross_nr_track(filt[filt["entry_type"] == entry_type], entry_type)


if __name__ == "__main__":
    main()


