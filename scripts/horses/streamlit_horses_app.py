import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import re
from dateutil import parser as date_parser
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.horses.config import settings
from src.horses.utils.text import normalize_track_name
from src.horses.analysis.signals import _signals_snapshot_path


# Regras disponiveis para cavalos
HORSE_RULE_LABELS: dict[str, str] = {
    "terceiro_queda50": "Regra 1 – 3º com queda ≥ 50% vs 2º (LAY)",
    "lider_volume_total": "Regra 2 – Líder com volume dominante (BACK)",
}
HORSE_RULE_LABELS_INV = {v: k for k, v in HORSE_RULE_LABELS.items()}

# Tipo de entrada (Back/Lay)
ENTRY_TYPE_LABELS: dict[str, str] = {
    "back": "Back (stake fixa)",
    "lay": "Lay (liability fixa)",
}
ENTRY_TYPE_LABELS_INV = {v: k for k, v in ENTRY_TYPE_LABELS.items()}


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

    scale_factor = base_amount / 10.0
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
        total_base_stake10 = 10.0 * metrics["signals"]
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


def _render_base_amount_input(default: float = 10.0) -> float:
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
            step=0.50,
            format="%.2f",
            key="base_amount",
            label_visibility="collapsed",
            help="Padrão: 10 unidades. Ajuste para reescalar PnL/ROI/drawdown.",
        )
        st.caption("Stake/Liab base (padrão: 10 unidades)")
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


def _to_iso_yyyy_mm_dd_thh_mm(value: str) -> str:
    try:
        dt = date_parser.parse(value, dayfirst=True)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _extract_category_letter(event_name: str) -> str:
    letter, _ = _classify_category_from_event_name(event_name)
    return letter


def _extract_category_token(event_name: str) -> str:
    _, token = _classify_category_from_event_name(event_name)
    return token


@st.cache_data(show_spinner=False)
def _cached_signals_csv(path: Path, encoding: str) -> pd.DataFrame:
    try:
        # Usa o path como chave de cache; Streamlit invalida se o conteudo mudar
        return pd.read_csv(path, encoding=encoding)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _cached_signals_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _build_category_index() -> dict[tuple[str, str], dict[str, str]]:
    result_dir = settings.DATA_DIR / "Result"
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for csv_path in sorted(result_dir.glob("dwbfprices*win*.csv")):
        try:
            df_r = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, usecols=["menu_hint", "event_dt", "event_name"])
        except Exception:
            continue
        df_r = df_r.dropna(subset=["menu_hint", "event_dt", "event_name"], how="any")
        if df_r.empty:
            continue
        df_r["track_key"] = df_r["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df_r["race_iso"] = df_r["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        df_r["cat_letter"] = df_r["event_name"].astype(str).map(_extract_category_letter)
        df_r["cat_token"] = df_r["event_name"].astype(str).map(_extract_category_token)
        for _, r in df_r.iterrows():
            key = (str(r["track_key"]), str(r["race_iso"]))
            if not key[0] or not key[1]:
                continue
            if key not in mapping:
                mapping[key] = {"letter": str(r.get("cat_letter", "")), "token": str(r.get("cat_token", ""))}
    return mapping


def load_signals(source: str = "top3", market: str = "win", strategy: str = "lay", provider: str = "timeform") -> pd.DataFrame:
    if strategy == "back":
        if source == "forecast" and market == "place":
            filename = "back_signals_forecast_place.csv"
        elif source == "forecast":
            filename = "back_signals_forecast.csv"
        elif market == "place":
            filename = "back_signals_place.csv"
        else:
            filename = "back_signals.csv"
    else:
        if source == "forecast" and market == "place":
            filename = "lay_signals_forecast_place.csv"
        elif source == "forecast":
            filename = "lay_signals_forecast.csv"
        elif market == "place":
            filename = "lay_signals_place.csv"
        else:
            filename = "lay_signals.csv"
    base = settings.DATA_DIR / "signals"
    if provider == "sportinglife":
        base = base / "sportinglife"
    csv_path = base / filename
    parquet_path = _signals_snapshot_path(
        source=source,
        market=market,
        strategy=strategy,
        provider=provider,
    )
    if parquet_path.exists():
        return _cached_signals_parquet(parquet_path)
    if csv_path.exists():
        return _cached_signals_csv(csv_path, settings.CSV_ENCODING)
    return pd.DataFrame()


def load_signals_for_rule(
    source: str,
    market: str,
    rule: str,
    entry_type: str,
    provider: str,
) -> pd.DataFrame:
    """
    Loader de alto nível para cavalos: dado rule e entry_type (back/lay/both),
    combina os DataFrames de sinais aplicáveis, etiquetando entry_type/rule.
    """
    frames: list[pd.DataFrame] = []

    if entry_type not in ("back", "lay", "both"):
        entry_type = "both"

    strategies_to_load: list[tuple[str, str]] = []
    if entry_type in ("lay", "both"):
        if rule == "terceiro_queda50":
            strategies_to_load.append(("lay", "lay"))
        else:
            strategies_to_load.append(("lay", "lay"))
    if entry_type in ("back", "both"):
        if rule == "lider_volume_total":
            strategies_to_load.append(("back", "back"))
        else:
            strategies_to_load.append(("back", "back"))

    for strategy_val, entry_val in strategies_to_load:
        df_part = load_signals(
            source=source,
            market=market,
            strategy=strategy_val,
            provider=provider,
        )
        if df_part is None or df_part.empty:
            continue
        df_part = df_part.copy()
        df_part["entry_type"] = entry_val
        df_part["rule"] = rule
        frames.append(df_part)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _render_monthly_table(df_block: pd.DataFrame, entry_kind: str, base_amount: float) -> None:
    """Tabela mensal com mesmas métricas do cabeçalho."""
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


def _render_block(title: str, df_block: pd.DataFrame, entry_kind: str, base_amount: float) -> None:
    st.subheader(title)
    summary = _compute_summary_metrics(df_block, entry_kind, base_amount)
    scale_factor = base_amount / 10.0

    # Linha 1
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

    # Linha Stake
    st.markdown('<div id="stake-row">', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric(" Base (Stake)", f"{summary['base_stake']:.2f}")
    with s2:
        st.metric("PnL Stake", f"{summary['pnl_stake']:.2f}")
    with s3:
        st.metric("ROI Stake", f"{summary['roi_stake']:.2%}")
    with s4:
        st.metric("Menor PnL acumulado", f"{summary['min_pnl_stake']:.2f}")
    with s5:
        st.metric("Drawdown máximo (Stake)", f"{summary['drawdown_stake']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Linha Liability (apenas lay)
    if entry_kind == "lay":
        st.markdown('<div id="liab-row">', unsafe_allow_html=True)
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            st.metric(" Stake (Liability)", f"{summary.get('stake_liab', 0.0):.2f}")
        with r2:
            st.metric("PnL Liability", f"{summary.get('pnl_liab', 0.0):.2f}")
        with r3:
            st.metric("ROI Liability", f"{summary.get('roi_liab', 0.0):.2%}")
        with r4:
            st.metric("Menor PnL (Liability)", f"{summary.get('min_pnl_liab', 0.0):.2f}")
        with r5:
            st.metric("Drawdown max (Liability)", f"{summary.get('drawdown_liab', 0.0):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Curvas de PnL
    plot = df_block.copy()
    if not plot.empty:
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["bet_idx"] = range(1, len(plot) + 1)
        plot["cum_stake"] = (plot["pnl_stake_fixed_10"] * scale_factor).cumsum()
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
        ch = (
            alt.Chart(plot)
            .mark_line()
            .encode(x=alt.X("bet_idx:Q", title="Bet #"), y=alt.Y("cum_stake:Q", title="PnL Stake"))
        )
        with st.expander("Evolucao Stake (PnL acumulado)", expanded=False):
            st.altair_chart(alt.layer(zero_line, ch).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        if entry_kind == "lay":
            plot["cum_liab"] = (plot["pnl_liability_fixed_10"] * scale_factor).cumsum()
            zero_line2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
            ch2 = (
                alt.Chart(plot)
                .mark_line()
                .encode(x=alt.X("bet_idx:Q", title="Bet #"), y=alt.Y("cum_liab:Q", title="PnL Liability"))
            )
            with st.expander("Evolucao Liability (PnL acumulado)", expanded=False):
                st.altair_chart(alt.layer(zero_line2, ch2).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # Relatorio mensal
    _render_monthly_table(df_block, entry_kind, base_amount)


def main() -> None:
    st.set_page_config(page_title="Sinais LAY/BACK - Cavalos", layout="wide")
    st.title("Sinais LAY/BACK - Estratégias Cavalos")
    # CSS especifico para limitar a altura dos multiselects de Datas e Pistas
    st.markdown(
        """
        <style>
        /* Limitar altura dos multiselects (Datas/Pistas) e adicionar rolagem */
        #dates-ms, #tracks-ms {
            max-height: 220px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        #dates-ms [data-baseweb="select"],
        #tracks-ms [data-baseweb="select"],
        #dates-ms [data-testid="stMultiSelect"],
        #tracks-ms [data-testid="stMultiSelect"] {
            max-height: 220px !important;
            min-height: 40px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        /* Garantir que o container do combobox (onde ficam os chips) tenha rolagem */
        #dates-ms [role="combobox"],
        #tracks-ms [role="combobox"] {
            max-height: 220px !important;
            min-height: 40px !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # seletores: regra, fonte, mercado, tipo de entrada
    col_rule, col_src, col_mkt, col_entry = st.columns([1.6, 1.2, 1, 1.2])
    with col_rule:
        rule_label_options = list(HORSE_RULE_LABELS.values())
        if "horse_rule_select_label" not in st.session_state:
            st.session_state["horse_rule_select_label"] = rule_label_options[0]
        selected_rule_label = st.selectbox(
            "Regra",
            rule_label_options,
            key="horse_rule_select_label",
        )
        rule = HORSE_RULE_LABELS_INV.get(selected_rule_label, "terceiro_queda50")

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

    col_prov, _ = st.columns([1, 3])
    with col_prov:
        provider = st.selectbox("Provedor", ["timeform", "sportinglife"], index=0)

    st.caption(f"Regra: {selected_rule_label} · Fonte: {source_label} · Mercado: {market} · Entrada: {entry_label} · Provedor: {provider}")

    df = load_signals_for_rule(
        source=source,
        market=market,
        rule=rule,
        entry_type=entry_type,
        provider=provider,
    )
    if df.empty:
        st.info("Nenhum sinal encontrado para a selecao. Gere antes com: python scripts/horses/generate_horse_signals.py --source {src} --market {mkt} --strategy {st}".format(src=source, mkt=market, st=entry_type))
        return

    filt = df.copy()
    # Garante colunas novas e numericas
    for col in ["num_runners", "total_matched_volume"]:
        if col not in filt.columns:
            filt[col] = pd.NA
    filt["num_runners"] = pd.to_numeric(filt["num_runners"], errors="coerce")
    filt["total_matched_volume"] = pd.to_numeric(filt["total_matched_volume"], errors="coerce")

    # Filtros (sem ratio; regra fixa >50%)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dates = sorted(filt["date"].dropna().unique().tolist())
        df_dates = pd.DataFrame({"Datas": dates, "Selecionar": True})
        edited_dates = st.data_editor(
            df_dates,
            column_config={
                "Selecionar": st.column_config.CheckboxColumn("Selecionar", default=True),
                "Datas": st.column_config.TextColumn("Datas"),
            },
            hide_index=True,
            height=240,
            use_container_width=True,
            key="dates_editor",
        )
        sel_dates = edited_dates.loc[edited_dates["Selecionar"] == True, "Datas"].tolist()
    with col_f2:
        tracks = sorted(filt["track_name"].dropna().unique().tolist())
        df_tracks = pd.DataFrame({"Pistas": tracks, "Selecionar": True})
        edited_tracks = st.data_editor(
            df_tracks,
            column_config={
                "Selecionar": st.column_config.CheckboxColumn("Selecionar", default=True),
                "Pistas": st.column_config.TextColumn("Pistas"),
            },
            hide_index=True,
            height=240,
            use_container_width=True,
            key="tracks_editor",
        )
        sel_tracks = edited_tracks.loc[edited_tracks["Selecionar"] == True, "Pistas"].tolist()
    with col_f3:
        if entry_type == "lay":
            bsp_series = filt["lay_target_bsp"]
        elif entry_type == "back":
            bsp_series = filt["back_target_bsp"]
        else:
            bsp_series = filt["lay_target_bsp"].where(filt["entry_type"] == "lay").fillna(filt["back_target_bsp"])
        bsp_min = float(bsp_series.min()) if not filt.empty else 1.01
        bsp_max = float(bsp_series.max()) if not filt.empty else 100.0
        bsp_min = max(1.01, round(bsp_min, 2))
        bsp_max = max(bsp_min, round(bsp_max, 2))
        # Inicializa defaults/clamp atuais em session_state
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
            key="bsp_slider", on_change=_sync_bsp_from_slider,
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

        # (campo movido para a frente do cabecalho de Stake)

    # Filtro adicional: participacao do lider (apenas BACK)
    if entry_type in ("back", "both"):
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
        filt_back_mask = (filt["entry_type"] == "back") if "entry_type" in filt.columns else True
        filt = filt[
            (~filt_back_mask)
            | (
                (filt_back_mask)
                & (filt["leader_volume_share_pct"].fillna(0) >= float(leader_min))
            )
        ]
    if sel_dates:
        filt = filt[filt["date"].isin(sel_dates)]
    if sel_tracks:
        filt = filt[filt["track_name"].isin(sel_tracks)]
    # Regra principal: diferenca > 50% em relacao ao vol3
    filt = filt[filt["pct_diff_second_vs_third"].fillna(0) > 50.0]

    # Aplica filtro por BSP com inputs precisos (respeitando limites)
    bsp_low = float(st.session_state.get("bsp_low", bsp_min))
    bsp_high = float(st.session_state.get("bsp_high", bsp_max))
    if entry_type == "lay":
        filt = filt[(filt["lay_target_bsp"] >= bsp_low) & (filt["lay_target_bsp"] <= bsp_high)]
    elif entry_type == "back":
        filt = filt[(filt["back_target_bsp"] >= bsp_low) & (filt["back_target_bsp"] <= bsp_high)]
    else:
        filt = filt[
            (
                (filt["entry_type"] == "lay")
                & (filt["lay_target_bsp"] >= bsp_low)
                & (filt["lay_target_bsp"] <= bsp_high)
            )
            | (
                (filt["entry_type"] == "back")
                & (filt["back_target_bsp"] >= bsp_low)
                & (filt["back_target_bsp"] <= bsp_high)
            )
        ]

    # Filtro por numero de corredores
    nr_vals = sorted([int(v) for v in filt["num_runners"].dropna().unique().tolist() if pd.notna(v)])
    if nr_vals:
        st.caption("Número de corredores")
        cw, _ = st.columns([2, 5])
        with cw:
            cb1, cb2 = st.columns([1, 1])
            with cb1:
                st.button(
                    "Todos",
                    key="nr_all",
                    on_click=lambda: st.session_state.update({"num_runners_ms": nr_vals}),
                )
            with cb2:
                st.button(
                    "Limpar",
                    key="nr_none",
                    on_click=lambda: st.session_state.update({"num_runners_ms": []}),
                )

        prev_nr = st.session_state.get("sel_num_runners", nr_vals.copy())
        default_nr = [v for v in prev_nr if v in nr_vals]
        sel_nr = st.multiselect(
            "Número de corredores",
            nr_vals,
            default=default_nr if default_nr else nr_vals,
            key="num_runners_ms",
            label_visibility="collapsed",
        )
        sel_nr = [int(v) for v in sel_nr if v in nr_vals]
        st.session_state["sel_num_runners"] = sel_nr
        if sel_nr:
            filt = filt[filt["num_runners"].isin(sel_nr)]
        else:
            filt = filt.iloc[0:0]

    # Filtro por volume total negociado
    vol_series = filt["total_matched_volume"].dropna()
    if not vol_series.empty:
        vol_min = float(vol_series.min())
        vol_max = float(vol_series.max())
        if vol_min < 0:
            vol_min = 0.0
        st.caption("Volume total da corrida (mercado WIN)")
        vol_low, vol_high = st.slider(
            "Volume total negociado",
            min_value=float(vol_min),
            max_value=float(vol_max),
            value=(float(vol_min), float(vol_max)),
            format="%.0f",
            label_visibility="collapsed",
        )
        filt = filt[
            (filt["total_matched_volume"] >= float(vol_low))
            & (filt["total_matched_volume"] <= float(vol_high))
        ]

    # Enriquecimento: categoria e token por corrida
    cat_index = _build_category_index()
    if not filt.empty and cat_index:
        filt["_key_track"] = filt["track_name"].astype(str).map(normalize_track_name)
        filt["_key_race"] = filt["race_time_iso"].astype(str)

        def _get_cat_letter(row) -> str:
            key = (row["_key_track"], row["_key_race"])
            meta = cat_index.get(key) or {}
            return str(meta.get("letter", "")).strip()

        def _get_cat_token(row) -> str:
            key = (row["_key_track"], row["_key_race"])
            meta = cat_index.get(key) or {}
            return str(meta.get("token", "")).strip()

        filt["category"] = filt.apply(_get_cat_letter, axis=1)
        filt["category_token"] = filt.apply(_get_cat_token, axis=1)
        filt = filt.drop(columns=["_key_track", "_key_race"], errors="ignore")

        cat_letters = sorted(
            [
                c
                for c in filt["category"].dropna().unique().tolist()
                if isinstance(c, str) and c
            ]
        )
        cat_tokens = sorted(
            [
                c
                for c in filt["category_token"].dropna().unique().tolist()
                if isinstance(c, str) and c
            ]
        )
    else:
        cat_letters = []
        cat_tokens = []

    # Filtro por categoria (letra)
    if cat_letters:
        st.caption("Categorias (G, H, M, N...)")
        cw, _ = st.columns([2, 5])
        with cw:
            cb1, cb2 = st.columns([1, 1])
            with cb1:
                st.button(
                    "Todas",
                    key="cats_all",
                    on_click=lambda: st.session_state.update({"cats_ms": cat_letters}),
                )
            with cb2:
                st.button(
                    "Limpar",
                    key="cats_none",
                    on_click=lambda: st.session_state.update({"cats_ms": []}),
                )
        prev_cats = st.session_state.get("sel_cats", cat_letters.copy())
        default_cats = [c for c in prev_cats if c in cat_letters]
        sel_cats = st.multiselect(
            "Categoria",
            cat_letters,
            default=default_cats if default_cats else cat_letters,
            key="cats_ms",
            label_visibility="collapsed",
        )
        st.session_state["sel_cats"] = [c for c in sel_cats if c in cat_letters]
        if sel_cats:
            filt = filt[filt["category"].isin(sel_cats)]
        else:
            filt = filt.iloc[0:0]

    # Filtro opcional por token (G1, HCP_CHS, MDN, ...)
    if cat_tokens:
        st.caption("Subcategoria (G1, HCP_CHS, MDN, ...)")
        sel_tokens = st.multiselect(
            "Subcategoria",
            cat_tokens,
            default=cat_tokens,
            key="cat_tokens_ms",
            label_visibility="collapsed",
        )
        if sel_tokens:
            filt = filt[filt["category_token"].isin(sel_tokens)]

    # Campo base (único) pos-filtros, antes de renderizar resultados
    base_amount = _render_base_amount_input(default=10.0)

    # Renderizacao por entry_type
    if entry_type == "both":
        render_back = filt[filt["entry_type"] == "back"] if "entry_type" in filt.columns else filt.iloc[0:0]
        render_lay = filt[filt["entry_type"] == "lay"] if "entry_type" in filt.columns else filt.iloc[0:0]
        _render_block("Resultados BACK", render_back, "back", base_amount)
        _render_block("Resultados LAY", render_lay, "lay", base_amount)
    else:
        render_df = filt[filt["entry_type"] == entry_type] if "entry_type" in filt.columns else filt
        _render_block(f"Resultados {entry_type.upper()}", render_df, entry_type, base_amount)

    # Tabela
    scale_factor = base_amount / 10.0

    show_cols = [
        "date", "track_name", "category", "category_token", "race_time_iso",
        "tf_top1", "tf_top2", "tf_top3",
        "vol_top1", "vol_top2", "vol_top3",
        "second_name_by_volume", "third_name_by_volume",
        "num_runners", "total_matched_volume",
        "entry_type",
        "lay_target_name", "lay_target_bsp",
        "stake_fixed_10", "liability_from_stake_fixed_10",
        "stake_for_liability_10", "liability_fixed_10",
        "win_lose", "is_green", "pnl_stake_fixed_10", "pnl_liability_fixed_10",
        "roi_row_stake_fixed_10", "roi_row_liability_fixed_10",
        "pct_diff_second_vs_third",
    ]
    missing = [c for c in show_cols if c not in filt.columns]
    for c in missing:
        filt[c] = ""

    if entry_type == "back":
        show_cols = [
            "date", "track_name", "category", "category_token", "race_time_iso",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
            "num_runners", "total_matched_volume",
            "entry_type",
            "back_target_name", "back_target_bsp",
            "stake_fixed_10",
            "win_lose", "is_green", "pnl_stake_fixed_10",
            "roi_row_stake_fixed_10",
            "pct_diff_second_vs_third",
        ]
    else:
        show_cols = [
            "date", "track_name", "category", "category_token", "race_time_iso",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
            "num_runners", "total_matched_volume",
            "entry_type",
            "lay_target_name", "lay_target_bsp",
            "stake_fixed_10", "liability_from_stake_fixed_10",
            "stake_for_liability_10", "liability_fixed_10",
            "win_lose", "is_green", "pnl_stake_fixed_10", "pnl_liability_fixed_10",
            "roi_row_stake_fixed_10", "roi_row_liability_fixed_10",
            "pct_diff_second_vs_third",
        ]

    missing = [c for c in show_cols if c not in filt.columns]
    for c in missing:
        filt[c] = ""
    st.dataframe(filt[show_cols], use_container_width=True)

    # Graficos pequenos: evolucao por pista e por categoria
    small_width = 360
    small_height = 180
    plot2 = filt.copy()
    if not plot2.empty:
        plot2["ts"] = pd.to_datetime(plot2["race_time_iso"], errors="coerce")
        plot2 = plot2.dropna(subset=["ts"]).sort_values("ts")
        plot2["date_only"] = plot2["ts"].dt.date

        # Por pista
        td = plot2.groupby(["track_name", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
        if not td.empty:
            td["cum"] = td.groupby("track_name")["pnl_stake_fixed_10"].cumsum() * scale_factor
            # contador de sinais por pista
            track_counts = plot2.groupby("track_name", as_index=False).size().rename(columns={"size": "count"})
            td = td.merge(track_counts, on="track_name", how="left")
            td["count"] = td["count"].fillna(0).astype(int)
            td["track_title"] = td["track_name"].astype(str) + " (" + td["count"].astype(str) + ")"
            st.subheader("Evolucao por pista (PnL acumulado)")
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

        # Por categoria (apenas letras A/B/D/OR/IV...) sem subcategorias
        if "category" in plot2.columns:
            cd = plot2[plot2["category"].astype(str).str.len() > 0].copy()
            if not cd.empty:
                # contador de sinais por categoria
                cat_counts = cd.groupby("category", as_index=False).size().rename(columns={"size": "count"})
                cd = cd.groupby(["category", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
                cd = cd.merge(cat_counts, on="category", how="left")
                cd["count"] = cd["count"].fillna(0).astype(int)
                cd["cum"] = cd.groupby("category")["pnl_stake_fixed_10"].cumsum() * scale_factor
                cd["category_title"] = cd["category"].astype(str) + " (" + cd["count"].astype(str) + ")"
                st.subheader("Evolucao por categoria (PnL acumulado)")
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

        # Por dia da semana (ordem fixa: Seg..Dom)
        st.subheader("Por dia da semana (PnL)")
        dow_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
        plot2["dow_idx"] = plot2["ts"].dt.dayofweek
        plot2["dow_name"] = plot2["dow_idx"].map({i: n for i, n in enumerate(dow_names)})
        dd = plot2.groupby("dow_name", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum()
        dd["stake"] = dd["pnl_stake_fixed_10"] * scale_factor
        dd["liab"] = dd["pnl_liability_fixed_10"] * scale_factor
        ch_dow = (
            alt.Chart(dd)
            .mark_bar()
            .encode(
                x=alt.X("dow_name:N", sort=dow_names, title=""),
                y=alt.Y("stake:Q", title="PnL"),
            )
            .properties(width=small_width * 2, height=small_height)
        )
        st.altair_chart(ch_dow.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        lay_any = (plot2["entry_type"] == "lay").any() if "entry_type" in plot2.columns else True
        if lay_any:
            ch_dow_liab = (
                alt.Chart(dd)
                .mark_bar()
                .encode(
                    x=alt.X("dow_name:N", sort=dow_names, title=""),
                    y=alt.Y("liab:Q", title="PnL (Liability)"),
                )
                .properties(width=small_width * 2, height=small_height)
            )
            st.altair_chart(ch_dow_liab.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)


if __name__ == "__main__":
    main()


