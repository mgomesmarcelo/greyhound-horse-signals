import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import re
from dateutil import parser as date_parser
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.greyhounds.config import settings
from src.greyhounds.config import RULE_LABELS, RULE_LABELS_INV, ENTRY_TYPE_LABELS, SOURCE_LABELS, SOURCE_LABELS_INV
from src.greyhounds.utils.text import normalize_track_name


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
    result_dir = settings.DATA_DIR / "Result"
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for csv_path in sorted(result_dir.glob("dwbfgreyhoundwin*.csv")):
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


def _build_num_runners_index() -> dict[tuple[str, str], int]:
    """Conta corredores por corrida a partir dos CSVs WIN (linhas por evento)."""
    result_dir = settings.DATA_DIR / "Result"
    counts: dict[tuple[str, str], int] = {}
    for csv_path in sorted(result_dir.glob("dwbfgreyhoundwin*.csv")):
        try:
            df_r = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, usecols=["menu_hint", "event_dt"], engine="python")
        except Exception:
            continue
        if df_r.empty:
            continue
        df_r["track_key"] = df_r["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df_r["race_iso"] = df_r["event_dt"].astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)
        grp = df_r.groupby(["track_key", "race_iso"], dropna=False).size()
        for (tk, ri), n in grp.items():
            if isinstance(tk, str) and tk and isinstance(ri, str) and ri:
                counts[(tk, ri)] = int(n)
    return counts


def load_signals(source: str = "top3", market: str = "win", rule: str = "terceiro_queda50") -> pd.DataFrame:
    filename = f"signals_{source}_{market}_{rule}.csv"
    path = settings.DATA_DIR / "signals" / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding=settings.CSV_ENCODING)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Sinais LAY/BACK - Galgos", layout="wide")
    st.title("Sinais LAY/BACK - Estrategias Greyhounds")

    # (Sem CSS custom)  Restaurado layout padrao do Streamlit

    # Dimensoes padrao para graficos pequenos (usadas em varias secoes)
    small_width = 360
    small_height = 180

    # seletores de fonte, mercado, regra e tipo de entrada
    col_src, col_mkt, col_rule, col_entry = st.columns([1, 1, 1.5, 1.2])
    with col_src:
        source_options = ["top3", "forecast", "betfair_resultado"]
        source_label_options = [SOURCE_LABELS.get(opt, opt) for opt in source_options]
        selected_source_label = st.selectbox(
            "Estrategia",
            source_label_options,
            index=0,
            key="source_select_label",
        )
        source = SOURCE_LABELS_INV.get(selected_source_label, "top3")
    source_label = SOURCE_LABELS.get(source, source)
    with col_mkt:
        market = st.selectbox("Mercado", ["win", "place"], index=0)
    with col_rule:
        rule_labels = [RULE_LABELS["terceiro_queda50"], RULE_LABELS["lider_volume_total"]]
        selected_rule_label = st.selectbox("Regra de selecao", rule_labels, index=0)
        rule = RULE_LABELS_INV.get(selected_rule_label, "terceiro_queda50")
    with col_entry:
        entry_opt_labels = ["ambos", ENTRY_TYPE_LABELS["back"], ENTRY_TYPE_LABELS["lay"]]
        entry_label = st.selectbox("Tipo de entrada", entry_opt_labels, index=0)
        if entry_label == "ambos":
            entry_type = "both"
        else:
            entry_type = "back" if entry_label == ENTRY_TYPE_LABELS["back"] else "lay"

    st.caption(f"Estrategia selecionada: {source_label}")

    df = load_signals(source=source, market=market, rule=rule)
    if df.empty:
        st.info("Nenhum sinal encontrado para a selecao. Gere antes com: python scripts/generate_greyhound_signals.py --source {src} --market {mkt} --rule {rule} --entry_type both".format(src=source, mkt=market, rule=rule))
        return

    # Filtros (sem ratio; regra fixa >50%)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dates = sorted(df["date"].dropna().unique().tolist())
        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            st.button(
                "Todos",
                key="dates_all",
                on_click=lambda: st.session_state.update({"dates_ms": dates}),
            )
        with b2:
            st.button(
                "Limpar",
                key="dates_none",
                on_click=lambda: st.session_state.update({"dates_ms": []}),
            )
        default_dates = st.session_state.get("dates_ms", dates)
        sel_dates = st.multiselect("Datas", dates, default=default_dates, key="dates_ms")
    with col_f2:
        tracks = sorted(df["track_name"].dropna().unique().tolist())
        tb1, tb2, _ = st.columns([1, 1, 2])
        with tb1:
            st.button(
                "Todos",
                key="tracks_all",
                on_click=lambda: st.session_state.update({"tracks_ms": tracks}),
            )
        with tb2:
            st.button(
                "Limpar",
                key="tracks_none",
                on_click=lambda: st.session_state.update({"tracks_ms": []}),
            )
        default_tracks = st.session_state.get("tracks_ms", tracks)
        sel_tracks = st.multiselect("Pistas", tracks, default=default_tracks, key="tracks_ms")
    with col_f3:
        if entry_type == "lay":
            bsp_col = "lay_target_bsp"
            base_df_for_bsp = df[df["entry_type"] == "lay"]
        elif entry_type == "back":
            bsp_col = "back_target_bsp"
            base_df_for_bsp = df[df["entry_type"] == "back"]
        else:
            # ambos: usa faixa unificada
            bsp_col = None
            base_df_for_bsp = df
        bsp_min = float(base_df_for_bsp[["lay_target_bsp","back_target_bsp"]].min().min()) if entry_type == "both" else float(base_df_for_bsp[bsp_col].min()) if not base_df_for_bsp.empty else 1.01
        bsp_max = float(base_df_for_bsp[["lay_target_bsp","back_target_bsp"]].max().max()) if entry_type == "both" else float(base_df_for_bsp[bsp_col].max()) if not base_df_for_bsp.empty else 100.0
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

    # Enriquecimento: num_runners (fallback se ausente)
    if "num_runners" not in df.columns:
        num_index = _build_num_runners_index()
        if not df.empty:
            df["_key_track"] = df["track_name"].astype(str).map(normalize_track_name)
            df["_key_race"] = df["race_time_iso"].astype(str)
            df["num_runners"] = df.apply(lambda r: num_index.get((str(r["_key_track"]), str(r["_key_race"])), pd.NA), axis=1)

    filt = df.copy()
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

    # Filtro por numero de corredores (dinamico conforme subconjunto atual)
    st.caption("Numero de corredores")
    nr_vals = sorted([int(v) for v in pd.to_numeric(filt.get("num_runners", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()])
    if nr_vals:
        nrw, _ = st.columns([2, 5])
        with nrw:
            nb1, nb2 = st.columns([1, 1])
            with nb1:
                st.button(
                    "Todos",
                    key="nr_all",
                    on_click=lambda: st.session_state.update({"num_runners_ms": nr_vals}),
                )
            with nb2:
                st.button(
                    "Limpar",
                    key="nr_none",
                    on_click=lambda: st.session_state.update({"num_runners_ms": []}),
                )
            prev_nr = st.session_state.get("sel_num_runners", nr_vals.copy())
            default_nr = [v for v in prev_nr if v in nr_vals]
            sel_nr = st.multiselect(
                "Numero de corredores",
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

    # Filtro por categoria
    if cat_letters:
        st.caption("Categorias")
        cw, _ = st.columns([2, 5])
        with cw:
            cb1, cb2 = st.columns([1, 1])
            with cb1:
                st.button(
                    "Todos",
                    key="cats_all",
                    on_click=lambda: st.session_state.update({"cats_ms": cat_letters}),
                )
            with cb2:
                st.button(
                    "Limpar",
                    key="cats_none",
                    on_click=lambda: st.session_state.update({"cats_ms": []}),
                )
            prev = st.session_state.get("sel_cats", cat_letters.copy())
            default_cats = [c for c in prev if c in cat_letters]
            sel_cats = st.multiselect(
                "Categoria (A/B/D...)",
                cat_letters,
                default=default_cats,
                key="cats_ms",
                label_visibility="collapsed",
            )
            sel_cats = [c for c in sel_cats if c in cat_letters]
            st.session_state["sel_cats"] = sel_cats
        if sel_cats:
            filt = filt[filt["category"].isin(sel_cats)]
        else:
            # sem selecao => nenhum resultado
            filt = filt.iloc[0:0]

    # Filtro por subcategoria (tokens completos A1, A2, D1, OR3, ...)
    sub_tokens = []
    if "category_token" in filt.columns:
        raw_tokens = [t for t in filt["category_token"].dropna().astype(str).unique().tolist() if isinstance(t, str) and t]
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
            prev_sc = st.session_state.get("sel_subcats", sub_tokens.copy())
            default_sc = [t for t in prev_sc if t in sub_tokens]
            sel_subcats = st.multiselect(
                "Subcategorias (A1/A2/D1/OR3/...)",
                sub_tokens,
                default=default_sc,
                key="subcats_ms",
                label_visibility="collapsed",
            )
            sel_subcats = [t for t in sel_subcats if t in sub_tokens]
            st.session_state["sel_subcats"] = sel_subcats
        if sel_subcats:
            filt = filt[filt["category_token"].isin(sel_subcats)]
        else:
            filt = filt.iloc[0:0]

    # Seletor global do eixo X para graficos de evolucao
    x_axis_mode = st.radio(
        "Eixo X dos graficos de evolucao",
        ["Dia", "Bet"],
        index=0,
        horizontal=True,
        help="Altere entre datas ou sequencia de apostas",
    )

    def render_block(title_suffix: str, df_block: pd.DataFrame, entry_kind: str) -> None:
        # ROI e Assertividade, reescalando a partir dos calculos base (10)
        base_amount = float(st.session_state.get("base_amount", 10.0))
        scale_factor = base_amount / 10.0
        if entry_kind == "lay":
            total_base_stake10 = float(df_block["liability_from_stake_fixed_10"].sum()) if not df_block.empty else 0.0
        else:
            total_base_stake10 = 10.0 * len(df_block)
        total_pnl_stake10 = float(df_block["pnl_stake_fixed_10"].sum()) if not df_block.empty else 0.0
        total_base_stake = total_base_stake10 * scale_factor
        total_pnl_stake = total_pnl_stake10 * scale_factor
        roi_stake = (total_pnl_stake / total_base_stake) if total_base_stake > 0 else 0.0
        acc_stake10 = (float((df_block["is_green"] == True).sum()) / len(df_block)) if not df_block.empty else 0.0

        st.subheader(title_suffix)

        # Linha 1: Pistas, Sinais, Greens, Reds, Media BSP, Assertividade
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Pistas", df_block["track_name"].nunique())
        with c2:
            st.metric("Sinais", len(df_block))
        num_greens = int((df_block["is_green"] == True).sum()) if not df_block.empty else 0
        num_reds = int(len(df_block) - num_greens) if not df_block.empty else 0
        with c3:
            st.metric("Greens", num_greens)
        with c4:
            st.metric("Reds", num_reds)
        with c5:
            avg_bsp = float((df_block["lay_target_bsp"] if entry_kind == "lay" else df_block["back_target_bsp"]).mean()) if not df_block.empty else 0.0
            st.metric("Media BSP Alvo", f"{avg_bsp:.2f}")
        with c6:
            st.metric("Assertividade", f"{acc_stake10:.2%}")

        # Campo valor base (reutiliza ja existente)
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
        if entry_kind == "lay":
            total_stake_liab10 = float(df_block["stake_for_liability_10"].sum()) if not df_block.empty else 0.0
            total_pnl_liab10 = float(df_block["pnl_liability_fixed_10"].sum()) if not df_block.empty else 0.0
            total_stake_liab = total_stake_liab10 * scale_factor
            total_pnl_liab = total_pnl_liab10 * scale_factor
            roi_liab = (total_pnl_liab / (base_amount * len(df_block))) if not df_block.empty and base_amount > 0 else 0.0
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

        # Graficos de evolucao
        plot = df_block.copy()
        if not plot.empty:
            plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
            plot = plot.dropna(subset=["ts"]).sort_values("ts")

            if x_axis_mode == "Dia":
                plot["date_only"] = plot["ts"].dt.date
                daily = plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum().sort_values("date_only")
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
                # Evolucao por sequencia de apostas (ordem temporal)
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

        # Tabela
        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "num_runners",
            "tf_top1", "tf_top2", "tf_top3",
            "vol_top1", "vol_top2", "vol_top3",
            "second_name_by_volume", "third_name_by_volume",
            "pct_diff_second_vs_third",
        ]
        if entry_kind == "lay":
            show_cols += [
                "lay_target_name", "lay_target_bsp",
                "stake_fixed_10", "liability_from_stake_fixed_10",
                "stake_for_liability_10", "liability_fixed_10",
                "win_lose", "is_green", "pnl_stake_fixed_10", "pnl_liability_fixed_10",
                "roi_row_stake_fixed_10", "roi_row_liability_fixed_10",
            ]
        else:
            show_cols += [
                "back_target_name", "back_target_bsp",
                "stake_fixed_10",
                "win_lose", "is_green", "pnl_stake_fixed_10",
                "roi_row_stake_fixed_10",
            ]
        missing = [c for c in show_cols if c not in df_block.columns]
        for c in missing:
            df_block[c] = ""
        st.dataframe(df_block[show_cols], use_container_width=True)

    # Enriquecimento feito antes; agora renderizacao por bloco
    if entry_type == "both":
        # Campo de valor base acima dos blocos BACK/LAY quando ambos estao selecionados
        if "base_amount" not in st.session_state:
            st.session_state["base_amount"] = 10.00
        cba_top, _ = st.columns([1, 6])
        with cba_top:
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
        render_block("Resultados BACK", filt[filt["entry_type"] == "back"], "back")
        render_block("Resultados LAY", filt[filt["entry_type"] == "lay"], "lay")
    else:
        # Evita duplicacao quando selecionado apenas um tipo: usar apenas os paineis agregados abaixo
        pass

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
        if entry_type == "lay":
            avg_bsp = float(filt["lay_target_bsp"].mean()) if not filt.empty else 0.0
        elif entry_type == "back":
            avg_bsp = float(filt["back_target_bsp"].mean()) if not filt.empty else 0.0
        else:
            lay_series = pd.to_numeric(filt.get("lay_target_bsp", pd.Series(dtype=float)), errors="coerce")
            back_series = pd.to_numeric(filt.get("back_target_bsp", pd.Series(dtype=float)), errors="coerce")
            avg_bsp = float(pd.concat([lay_series, back_series]).mean()) if not filt.empty else 0.0
        st.metric("Media BSP Alvo", f"{avg_bsp:.2f}")
    with c6:
        acc_val = (num_greens / len(filt)) if len(filt) > 0 else 0.0
        st.metric("Assertividade", f"{acc_val:.2%}")

    # Campo compacto antes do cabecalho de Stake para definir o valor base (somente quando nao for "ambos")
    if entry_type != "both":
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
    # Base (Stake) depende do tipo de entrada
    if entry_type == "lay":
        total_base_stake10 = float(filt["liability_from_stake_fixed_10"].sum()) if not filt.empty else 0.0
    elif entry_type == "back":
        total_base_stake10 = 10.0 * len(filt)
    else:
        # ambos: soma base de LAY (liability_from_stake_fixed_10) + base de BACK (10 por aposta)
        lay_part = float(filt.loc[filt["entry_type"] == "lay", "liability_from_stake_fixed_10"].sum())
        back_part = 10.0 * int((filt["entry_type"] == "back").sum())
        total_base_stake10 = lay_part + back_part
    total_pnl_stake10 = float(filt["pnl_stake_fixed_10"].sum()) if not filt.empty else 0.0
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
    if entry_type == "lay":
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

    # Graficos de evolucao (por dia ou por bet) com opcao de minimizar
    plot = filt.copy()
    if not plot.empty:
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        # Sempre calcula agregacao diaria para uso em graficos semanais, mesmo quando o eixo X e "Bet"
        plot["date_only"] = plot["ts"].dt.date
        daily = plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum().sort_values("date_only")
        daily["cum_stake"] = (daily["pnl_stake_fixed_10"] * scale_factor).cumsum()
        daily["cum_liab"] = (daily["pnl_liability_fixed_10"] * scale_factor).cumsum()

        if x_axis_mode == "Dia":
            plot["date_only"] = plot["ts"].dt.date
            daily = plot.groupby("date_only", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum().sort_values("date_only")
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
        else:
            # Evolucao por sequencia de apostas (ordem temporal)
            plot["bet_idx"] = range(1, len(plot) + 1)
            plot["cum_stake"] = (plot["pnl_stake_fixed_10"] * scale_factor).cumsum()
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

        # Desempenho agregado por dia da semana
        daily["weekday"] = pd.to_datetime(daily["date_only"]).dt.weekday
        wd_order = [0, 1, 2, 3, 4, 5, 6]
        wd_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
        wd_order_names = [wd_names[w] for w in wd_order]
        by_wd = daily.groupby("weekday", as_index=False)[["pnl_stake_fixed_10", "pnl_liability_fixed_10"]].sum()
        by_wd["weekday_name"] = by_wd["weekday"].map(wd_names)
        by_wd["weekday_name"] = pd.Categorical(by_wd["weekday_name"], categories=wd_order_names, ordered=True)
        by_wd["pnl_stake"] = by_wd["pnl_stake_fixed_10"] * scale_factor
        if entry_type == "lay":
            by_wd["pnl_liab"] = by_wd["pnl_liability_fixed_10"] * scale_factor

        with st.expander("Desempenho por dia da semana (PnL agregado)", expanded=False):
            bar_stake = (
                alt.Chart(by_wd)
                .mark_bar()
                .encode(
                    x=alt.X("weekday_name:N", sort=wd_order_names, title=""),
                    y=alt.Y("pnl_stake:Q", title="PnL"),
                )
                .properties(width=small_width * 2, height=small_height)
            )
            if entry_type == "lay":
                bar_liab = (
                    alt.Chart(by_wd)
                    .mark_bar(color="#8888FF")
                    .encode(
                        x=alt.X("weekday_name:N", sort=wd_order_names, title=""),
                        y=alt.Y("pnl_liab:Q", title="PnL"),
                    )
                    .properties(width=small_width * 2, height=small_height)
                )
                st.altair_chart(
                    alt.vconcat(
                        bar_stake.properties(title="Stake"),
                        bar_liab.properties(title="Liability"),
                    ).resolve_scale(y="independent").configure_view(stroke="#888", strokeWidth=1),
                    use_container_width=True,
                )
            else:
                st.altair_chart(
                    bar_stake.configure_view(stroke="#888", strokeWidth=1).properties(title="Stake"),
                    use_container_width=True,
                )

    # Tabela (evita duplicacao quando entry_type == "both", pois ja exibimos 2 tabelas acima)
    if entry_type != "both":
        show_cols = [
            "date", "track_name", "category_token", "race_time_iso",
            "num_runners",
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

        if entry_type == "back":
            show_cols = [
                "date", "track_name", "category_token", "race_time_iso",
                "num_runners",
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
                "num_runners",
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

    # Graficos pequenos: evolucao por pista e por categoria (um bloco por tipo de entrada)
    def _render_small_charts(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot2 = df_block.copy()
        if plot2.empty:
            return
        plot2["ts"] = pd.to_datetime(plot2["race_time_iso"], errors="coerce")
        plot2 = plot2.dropna(subset=["ts"]).sort_values("ts")
        plot2["date_only"] = plot2["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 10.0)) / 10.0

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
            # Evolucao por sequencia de apostas dentro de cada pista
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

        # Por categoria (apenas letras A/B/D/OR/IV...) sem subcategorias)
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
                    # Evolucao por sequencia de apostas dentro de cada categoria
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
                    gp = sp.groupby(["track_name", "category_token", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
                    gp = gp.merge(sp[["track_name", "track_title"]].drop_duplicates(), on="track_name", how="left")
                    # contador por celula (pista  subcategoria)
                    cell_counts = sp.groupby(["track_name", "category_token"], as_index=False).size().rename(columns={"size": "cell_count"})
                    gp = gp.merge(cell_counts, on=["track_name", "category_token"], how="left")
                    gp["cum"] = gp.groupby(["track_name", "category_token"])['pnl_stake_fixed_10'].cumsum() * local_scale
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
                    sp["cum"] = sp.groupby(["track_name", "category_token"])['pnl_stake_fixed_10'].cumsum() * local_scale
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
                    nd = nd.groupby(["num_runners", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
                    nd = nd.merge(nr_counts, on="num_runners", how="left")
                    nd["count"] = nd["count"].fillna(0).astype(int)
                    nd["cum"] = nd.groupby("num_runners")["pnl_stake_fixed_10"].cumsum() * local_scale
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
                    nd["cum"] = nd.groupby("num_runners")["pnl_stake_fixed_10"].cumsum() * local_scale
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

    if entry_type == "both":
        _render_small_charts(filt[filt["entry_type"] == "back"], "back")
        _render_small_charts(filt[filt["entry_type"] == "lay"], "lay")
    else:
        _render_small_charts(filt[filt["entry_type"] == entry_type], entry_type)

    # --- Graficos cruzados: Categoria  Numero de corredores (Evolucao) ---
    def _render_cross_nr_category(df_block: pd.DataFrame, entry_kind: str) -> None:
        plot = df_block.copy()
        if plot.empty or "num_runners" not in plot.columns or "category" not in plot.columns:
            return
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 10.0)) / 10.0
        # Titulos
        counts = plot.groupby(["category", "num_runners"], as_index=False).size().rename(columns={"size": "count"})
        if x_axis_mode == "Dia":
            agg = plot.groupby(["category", "num_runners", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["category", "num_runners"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["category", "num_runners"])['pnl_stake_fixed_10'].cumsum() * local_scale
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
            plot["cum"] = plot.groupby(["category", "num_runners"])['pnl_stake_fixed_10'].cumsum() * local_scale
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
        plot["ts"] = pd.to_datetime(plot["race_time_iso"], errors="coerce")
        plot = plot.dropna(subset=["ts"]).sort_values("ts")
        plot["date_only"] = plot["ts"].dt.date
        local_scale = float(st.session_state.get("base_amount", 10.0)) / 10.0
        # Seleciona Top K pistas por quantidade
        top_k = 12
        track_sizes = plot.groupby("track_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        top_tracks = set(track_sizes.head(top_k)["track_name"].tolist())
        plot = plot[plot["track_name"].isin(top_tracks)]
        counts = plot.groupby(["track_name", "num_runners"], as_index=False).size().rename(columns={"size": "count"})
        plot = plot.merge(track_sizes[["track_name", "count"]].rename(columns={"count": "track_total"}), on="track_name", how="left")
        if x_axis_mode == "Dia":
            agg = plot.groupby(["track_name", "num_runners", "date_only"], as_index=False)[["pnl_stake_fixed_10"]].sum().sort_values("date_only")
            agg = agg.merge(counts, on=["track_name", "num_runners"], how="left")
            agg["count"] = agg["count"].fillna(0).astype(int)
            agg["cum"] = agg.groupby(["track_name", "num_runners"])['pnl_stake_fixed_10'].cumsum() * local_scale
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
            plot["cum"] = plot.groupby(["track_name", "num_runners"])['pnl_stake_fixed_10'].cumsum() * local_scale
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

    # --- Heatmaps por Categoria/Pista  Numero de corredores ---
    def _render_heatmaps(df_block: pd.DataFrame, entry_kind: str) -> None:
        base_amount = float(st.session_state.get("base_amount", 10.0))
        scale = base_amount / 10.0
        metric = st.selectbox(
            f"Metrica do heatmap ({entry_kind.upper()})",
            ["ROI", "PnL", "Assertividade"],
            index=0,
            key=f"heat_metric_{entry_kind}",
        )
        def _agg(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=by_cols + ["roi", "pnl", "acc", "count"])
            pnl10 = df["pnl_stake_fixed_10"].sum()
            if entry_kind == "lay":
                base10 = df["liability_from_stake_fixed_10"].sum()
            else:
                base10 = 10.0 * len(df)
            pnl = pnl10 * scale
            base_val = base10 * scale
            roi = (pnl / base_val) if base_val > 0 else 0.0
            acc = float((df["is_green"] == True).sum()) / len(df) if len(df) > 0 else 0.0
            out = df.groupby(by_cols, as_index=False).size().rename(columns={"size": "count"})
            out["roi"] = roi
            out["pnl"] = pnl
            out["acc"] = acc
            return out
        # Categoria  N
        st.subheader(f"Heatmap por categoria  numero de corredores - {entry_kind.upper()}")
        by_cat = _agg(df_block, ["category", "num_runners"]).copy()
        if not by_cat.empty:
            value_col = {"ROI": "roi", "PnL": "pnl", "Assertividade": "acc"}[metric]
            by_cat["category"] = by_cat["category"].astype(str)
            cat_heat = (
                alt.Chart(by_cat)
                .mark_rect()
                .encode(
                    x=alt.X("category:N", title="Categoria"),
                    y=alt.Y("num_runners:O", title="N corredores", sort="ascending"),
                    color=alt.Color(f"{value_col}:Q", title=metric),
                    tooltip=["category", "num_runners", alt.Tooltip("count:Q", title="bets"), alt.Tooltip(f"{value_col}:Q", title=metric)],
                )
            )
            st.altair_chart(cat_heat.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)
        # Pista  N (Top K por contagem)
        st.subheader(f"Heatmap por pista  numero de corredores - {entry_kind.upper()}")
        top_k = 12
        track_sizes = df_block.groupby("track_name", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        top_tracks = set(track_sizes.head(top_k)["track_name"].tolist())
        df_top = df_block[df_block["track_name"].isin(top_tracks)].copy()
        by_track = _agg(df_top, ["track_name", "num_runners"]).copy()
        if not by_track.empty:
            value_col = {"ROI": "roi", "PnL": "pnl", "Assertividade": "acc"}[metric]
            by_track["track_name"] = by_track["track_name"].astype(str)
            tr_heat = (
                alt.Chart(by_track)
                .mark_rect()
                .encode(
                    x=alt.X("track_name:N", title="Pista"),
                    y=alt.Y("num_runners:O", title="N corredores", sort="ascending"),
                    color=alt.Color(f"{value_col}:Q", title=metric),
                    tooltip=["track_name", "num_runners", alt.Tooltip("count:Q", title="bets"), alt.Tooltip(f"{value_col}:Q", title=metric)],
                )
            )
            st.altair_chart(tr_heat.configure_view(stroke="#888", strokeWidth=1), use_container_width=True)

    # Renderizacao dos novos blocos para BACK/LAY
    if entry_type == "both":
        _render_cross_nr_category(filt[filt["entry_type"] == "back"], "back")
        _render_cross_nr_category(filt[filt["entry_type"] == "lay"], "lay")
        _render_cross_nr_track(filt[filt["entry_type"] == "back"], "back")
        _render_cross_nr_track(filt[filt["entry_type"] == "lay"], "lay")
        _render_heatmaps(filt[filt["entry_type"] == "back"], "back")
        _render_heatmaps(filt[filt["entry_type"] == "lay"], "lay")
    else:
        _render_cross_nr_category(filt[filt["entry_type"] == entry_type], entry_type)
        _render_cross_nr_track(filt[filt["entry_type"] == entry_type], entry_type)
        _render_heatmaps(filt[filt["entry_type"] == entry_type], entry_type)


if __name__ == "__main__":
    main()


