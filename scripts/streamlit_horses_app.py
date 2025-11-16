import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import re
from dateutil import parser as date_parser
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.horses.config import settings
from src.horses.utils.text import normalize_track_name


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
    path = base / filename
    if not path.exists():
        return pd.DataFrame()
    return _cached_signals_csv(path, settings.CSV_ENCODING)


def main() -> None:
    st.set_page_config(page_title="Sinais LAY/BACK - Galgos", layout="wide")
    st.title("Sinais LAY/BACK - Top3/Forecast (Timeform & SportingLife)")
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

    # seletores de provedor, fonte e mercado
    col_prov, col_src, col_mkt, col_strat, _ = st.columns([1, 1, 1, 1, 2])
    with col_prov:
        provider = st.selectbox("Provedor", ["timeform", "sportinglife"], index=0)
    with col_src:
        source = st.selectbox("Fonte", ["top3", "forecast"], index=0)
    with col_mkt:
        market = st.selectbox("Mercado", ["win", "place"], index=0)
    with col_strat:
        strategy = st.selectbox("Estrategia", ["lay", "back"], index=0)
    df = load_signals(source=source, market=market, strategy=strategy, provider=provider)
    if df.empty:
        st.info("Nenhum sinal encontrado para a selecao. Gere antes com: python scripts/generate_signals.py --source {src} --market {mkt} --strategy {st}".format(src=source, mkt=market, st=strategy))
        return

    # Filtros (sem ratio; regra fixa >50%)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dates = sorted(df["date"].dropna().unique().tolist())
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
        tracks = sorted(df["track_name"].dropna().unique().tolist())
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
        bsp_col = "lay_target_bsp" if strategy == "lay" else "back_target_bsp"
        bsp_min = float(df[bsp_col].min()) if not df.empty else 1.01
        bsp_max = float(df[bsp_col].max()) if not df.empty else 100.0
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

    filt = df.copy()
    # Filtro adicional: participacao do lider (apenas BACK)
    if strategy == "back":
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
    # Enriquecimento: categoria por corrida (A/B/D etc.)
    cat_index = _build_category_index()
    if not filt.empty:
        filt["_key_track"] = filt["track_name"].astype(str).map(normalize_track_name)
        filt["_key_race"] = filt["race_time_iso"].astype(str)
        filt["category"] = filt.apply(lambda r: (cat_index.get((r["_key_track"], r["_key_race"]), {}) or {}).get("letter", ""), axis=1)
        filt["category_token"] = filt.apply(lambda r: (cat_index.get((r["_key_track"], r["_key_race"]), {}) or {}).get("token", ""), axis=1)
        # Ordena letras (facilita UI)
        cat_letters = sorted([c for c in filt["category"].dropna().unique().tolist() if isinstance(c, str) and c])
    else:
        cat_letters = []
    if sel_dates:
        filt = filt[filt["date"].isin(sel_dates)]
    if sel_tracks:
        filt = filt[filt["track_name"].isin(sel_tracks)]
    # Regra principal: diferenca > 50% em relacao ao vol3
    filt = filt[filt["pct_diff_second_vs_third"].fillna(0) > 50.0]
    # Aplica filtro por BSP com inputs precisos (respeitando limites)
    bsp_low = float(st.session_state.get("bsp_low", bsp_min))
    bsp_high = float(st.session_state.get("bsp_high", bsp_max))
    filt = filt[(filt[bsp_col] >= bsp_low) & (filt[bsp_col] <= bsp_high)]

    # Filtro por categoria
    if cat_letters:
        st.caption("Categorias")
        cw, _ = st.columns([2, 5])
        with cw:
            if "sel_cats" not in st.session_state:
                st.session_state["sel_cats"] = cat_letters.copy()
            # Garante que os defaults existam nas opcoes atuais
            default_cats = [c for c in st.session_state.get("sel_cats", []) if c in cat_letters]
            if not default_cats:
                default_cats = cat_letters.copy()
                st.session_state["sel_cats"] = default_cats
            sel_cats = st.multiselect(
                "Categoria (A/B/D...)",
                cat_letters,
                default=default_cats,
                key="cats_ms",
                label_visibility="collapsed",
            )
            st.session_state["sel_cats"] = [c for c in sel_cats if c in cat_letters]
        if sel_cats:
            filt = filt[filt["category"].isin(sel_cats)]
        else:
            # sem selecao => nenhum resultado
            filt = filt.iloc[0:0]

    # Metricas
    # ROI e Assertividade, reescalando a partir dos calculos base (10)
    base_amount = float(st.session_state.get("base_amount", 10.0))
    scale_factor = base_amount / 10.0
    if strategy == "lay":
        total_base_stake10 = float(filt["liability_from_stake_fixed_10"].sum()) if not filt.empty else 0.0
    else:
        total_base_stake10 = 10.0 * len(filt)
    total_pnl_stake10 = float(filt["pnl_stake_fixed_10"].sum()) if not filt.empty else 0.0
    total_base_stake = total_base_stake10 * scale_factor
    total_pnl_stake = total_pnl_stake10 * scale_factor
    roi_stake = (total_pnl_stake / total_base_stake) if total_base_stake > 0 else 0.0
    acc_stake10 = (float((filt["is_green"] == True).sum()) / len(filt)) if not filt.empty else 0.0

    # Linha 1: Pistas, Sinais, Greens, Reds, Media BSP, Assertividade
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Pistas", filt["track_name"].nunique())
    with c2:
        st.metric("Sinais", len(filt))
    # Contagem de greens/reds
    num_greens = int((filt["is_green"] == True).sum()) if not filt.empty else 0
    num_reds = int(len(filt) - num_greens) if not filt.empty else 0
    with c3:
        st.metric("Greens", num_greens)
    with c4:
        st.metric("Reds", num_reds)
    with c5:
        avg_bsp = float(filt[bsp_col].mean()) if not filt.empty else 0.0
        st.metric("Media BSP Alvo", f"{avg_bsp:.2f}")
    with c6:
        st.metric("Assertividade", f"{acc_stake10:.2%}")

    # Campo compacto antes do cabecalho de Stake para definir o valor base
    if "base_amount" not in st.session_state:
        st.session_state["base_amount"] = 10.00
    cba1, _ = st.columns([1, 6])
    with cba1:
        st.number_input(
            "Valor base (Stake e Liability)",
            min_value=0.01,
            max_value=100000.0,
            value=float(st.session_state["base_amount"]),
            step=0.50,
            format="%.2f",
            key="base_amount",
            label_visibility="collapsed",
        )
        st.caption("Stake/Liab base")

    # Recalcula fatores apos possivel alteracao do input
    base_amount = float(st.session_state.get("base_amount", 10.0))
    scale_factor = base_amount / 10.0
    total_base_stake = total_base_stake10 * scale_factor
    total_pnl_stake = total_pnl_stake10 * scale_factor
    roi_stake = (total_pnl_stake / total_base_stake) if total_base_stake > 0 else 0.0

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
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric(" Base (Stake)", f"{total_base_stake:.2f}")
    with s2:
        st.metric("PnL Stake", f"{total_pnl_stake:.2f}")
    with s3:
        st.metric("ROI Stake", f"{roi_stake:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Linha 3: Liability(10)  apenas para LAY
    if strategy == "lay":
        total_stake_liab10 = float(filt["stake_for_liability_10"].sum()) if not filt.empty else 0.0
        total_pnl_liab10 = float(filt["pnl_liability_fixed_10"].sum()) if not filt.empty else 0.0
        total_stake_liab = total_stake_liab10 * scale_factor
        total_pnl_liab = total_pnl_liab10 * scale_factor
        roi_liab = (total_pnl_liab / (base_amount * len(filt))) if not filt.empty and base_amount > 0 else 0.0
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
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(" Stake (Liability)", f"{total_stake_liab:.2f}")
        with r2:
            st.metric("PnL Liability", f"{total_pnl_liab:.2f}")
        with r3:
            st.metric("ROI Liability", f"{roi_liab:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Graficos de evolucao (por dia) com opcao de minimizar
    plot = filt.copy()
    if not plot.empty:
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        daily = plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum().sort_values("date_only")
        daily["cum_stake"] = (daily["pnl_stake_fixed_10"] * scale_factor).cumsum()
        daily["cum_liab"] = (daily["pnl_liability_fixed_10"] * scale_factor).cumsum()

        with st.expander("Evolucao Stake (PnL acumulado por dia)", expanded=False):
            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
            ch = (
                alt.Chart(daily)
                .mark_line()
                .encode(x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")), y=alt.Y("cum_stake:Q", title="PnL"))
            )
            st.altair_chart(alt.layer(zero_line, ch).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

        if strategy == "lay":
            with st.expander("Evolucao Liability (PnL acumulado por dia)", expanded=False):
                zero_line2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeWidth=1).encode(y="y:Q")
                ch2 = (
                    alt.Chart(daily)
                    .mark_line()
                    .encode(x=alt.X("date_only:T", title="", axis=alt.Axis(format="%Y-%m-%d")), y=alt.Y("cum_liab:Q", title="PnL"))
                )
                st.altair_chart(alt.layer(zero_line2, ch2).configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # Tabela
    show_cols = [
        "date", "track_name", "category_token", "race_time_iso",
        "tf_top1", "tf_top2", "tf_top3",
        "vol_top1", "vol_top2", "vol_top3",
        "second_name_by_volume", "third_name_by_volume",
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

    if strategy == "back":
        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
            "back_target_name", "back_target_bsp",
            "stake_fixed_10",
            "win_lose", "is_green", "pnl_stake_fixed_10",
            "roi_row_stake_fixed_10",
            "pct_diff_second_vs_third",
        ]
    else:
        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
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

        if strategy == "lay":
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


