"""
Dashboard do cliente — porta 8503.

O que mostra:
  - Status do feed diario (sinais gerados hoje?)
  - Upload de CSV da Betfair → filtrado pelos sinais do sistema → historico acumulado
  - Metricas reais de performance (P&L, ROI, strike rate, drawdown)
  - URL do feed para configurar no BF Bot Manager

O que NAO mostra:
  - Estrategias, filtros, botoes de exportar/importar
  - Nenhuma informacao sobre como os sinais sao gerados
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="Sinais — Minha Performance",
    layout="wide",
    page_icon="📊",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.betfair_parser import filter_system_bets, parse_betfair_csv

# ---------------------------------------------------------------------------
# AUTENTICACAO
# ---------------------------------------------------------------------------
_AUTH_CONFIG_PATH = PROJECT_ROOT / "config" / "auth.yaml"

with open(_AUTH_CONFIG_PATH, encoding="utf-8") as _f:
    _auth_config = yaml.load(_f, Loader=SafeLoader)

_authenticator = stauth.Authenticate(
    _auth_config["credentials"],
    _auth_config["cookie"]["name"] + "_cliente",
    _auth_config["cookie"]["key"],
    _auth_config["cookie"]["expiry_days"],
)

_authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Utilizador ou senha incorretos.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Por favor, introduza as suas credenciais.")
    st.stop()

_username: str = st.session_state.get("username", "")

with st.sidebar:
    _authenticator.logout("Sair", "sidebar")
    st.markdown("---")
    st.caption(f"Sessao: **{_username}**")

# ---------------------------------------------------------------------------
# CAMINHOS
# ---------------------------------------------------------------------------
SIGNAL_INDEX_PATH = PROJECT_ROOT / "data" / "signal_index.parquet"
CLIENT_RESULTS_DIR = PROJECT_ROOT / "data" / "client_results" / _username
DAILY_TIPS_DIR = PROJECT_ROOT / "data" / "daily_tips"
CLIENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FUNCOES — cruzamento com indice de sinais
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_signal_index() -> pd.DataFrame | None:
    if not SIGNAL_INDEX_PATH.exists():
        return None
    return pd.read_parquet(SIGNAL_INDEX_PATH)


# ---------------------------------------------------------------------------
# FUNCOES — historico de uploads do cliente
# ---------------------------------------------------------------------------

def load_client_history() -> pd.DataFrame:
    """Carrega e combina todos os CSVs salvos do cliente."""
    files = sorted(CLIENT_RESULTS_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Remove duplicatas por bet_id se existir
    if "bet_id" in df.columns:
        df = df.drop_duplicates(subset=["bet_id"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


def save_upload(df: pd.DataFrame) -> Path:
    """Salva o upload filtrado como Parquet com timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = CLIENT_RESULTS_DIR / f"{ts}_upload.parquet"
    df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# FUNCOES — metricas
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pl = df["profit_loss"]
    total_pl = pl.sum()
    n = len(df)
    wins = (pl > 0).sum()
    strike = wins / n if n > 0 else 0.0

    # Exposicao = stake para back, liability para lay
    exposure = 0.0
    if "stake" in df.columns and "liability" in df.columns and "entry_type" in df.columns:
        back_exp = df.loc[df["entry_type"] == "back", "stake"].sum()
        lay_exp = df.loc[df["entry_type"] == "lay", "liability"].sum()
        exposure = back_exp + lay_exp
    roi = (total_pl / exposure * 100) if exposure > 0 else 0.0

    # Drawdown
    cumpl = pl.cumsum()
    running_max = cumpl.cummax()
    drawdown = (cumpl - running_max).min()

    return {
        "total_pl": total_pl,
        "n_bets": n,
        "wins": int(wins),
        "strike_rate": strike,
        "exposure": exposure,
        "roi": roi,
        "max_drawdown": drawdown,
    }


# ---------------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------------

st.title("Minha Performance")

# --- Secao 1: Status do feed ---
st.subheader("Status do Feed de Sinais")
today = datetime.date.today().isoformat()
feed_cols = st.columns(2)

for sport_label, sport_key in [("Galgos", "greyhounds"), ("Cavalos", "horses")]:
    tip_file = DAILY_TIPS_DIR / sport_key / f"{today}_CONSOLIDADO.csv"
    with feed_cols[0 if sport_key == "greyhounds" else 1]:
        if tip_file.exists():
            mtime = datetime.datetime.fromtimestamp(tip_file.stat().st_mtime)
            st.success(f"**{sport_label}** — sinais gerados hoje às {mtime.strftime('%H:%M')}")
        else:
            st.warning(f"**{sport_label}** — sinais ainda não gerados para hoje")

st.divider()

# --- Secao 2: Upload ---
st.subheader("Carregar Resultados da Betfair")
st.caption(
    "Aceda à Betfair → My Bets → Download → selecione o período → faça download do CSV. "
    "Apenas as apostas correspondentes aos sinais do sistema serão consideradas."
)

signal_index = load_signal_index()

if signal_index is None:
    st.error(
        "Índice de sinais não encontrado. Contacte o administrador para executar `build_signal_index.py`."
    )
else:
    uploaded_file = st.file_uploader(
        "Selecione o CSV de histórico de apostas da Betfair",
        type=["csv"],
        key="betfair_upload",
    )

    if uploaded_file is not None:
        with st.spinner("A processar o ficheiro..."):
            try:
                raw_df = parse_betfair_csv(uploaded_file.read())
                total_bets = len(raw_df)
                filtered = filter_system_bets(raw_df, signal_index)
                n_system = len(filtered)

                if n_system == 0:
                    st.warning(
                        f"O CSV continha {total_bets} apostas mas nenhuma correspondeu "
                        "aos sinais do sistema. Verifique se o período é correto."
                    )
                else:
                    save_upload(filtered)
                    st.success(
                        f"✅ {n_system} aposta(s) do sistema encontrada(s) "
                        f"(de {total_bets} no total). Histórico actualizado."
                    )
            except ValueError as e:
                st.error(f"Erro ao processar CSV: {e}")
            except Exception as e:
                st.error(f"Erro inesperado: {e}")

st.divider()

# --- Secao 3: Metricas ---
st.subheader("Performance Acumulada")

history = load_client_history()

if history.empty:
    st.info(
        "Ainda sem dados. Faça o upload do CSV da Betfair acima para ver as suas métricas."
    )
else:
    metrics = compute_metrics(history)

    m1, m2, m3, m4 = st.columns(4)
    pl_color = "normal" if metrics["total_pl"] >= 0 else "inverse"
    m1.metric("P&L Total", f"£{metrics['total_pl']:.2f}")
    m2.metric("ROI", f"{metrics['roi']:.1f}%")
    m3.metric("Strike Rate", f"{metrics['strike_rate']*100:.1f}%")
    m4.metric("Max Drawdown", f"£{metrics['max_drawdown']:.2f}")

    m5, m6 = st.columns(2)
    m5.metric("Total de Apostas", metrics["n_bets"])
    m6.metric("Apostas Ganhas", metrics["wins"])

    # Grafico P&L acumulado
    st.subheader("Evolução do P&L")
    chart_df = history[["date", "profit_loss"]].copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.sort_values("date")
    chart_df["pl_acumulado"] = chart_df["profit_loss"].cumsum()

    line = (
        alt.Chart(chart_df)
        .mark_line(color="#4CAF50", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Data"),
            y=alt.Y("pl_acumulado:Q", title="P&L Acumulado (£)"),
            tooltip=["date:T", alt.Tooltip("pl_acumulado:Q", format=".2f")],
        )
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    st.altair_chart((zero_line + line).properties(height=300), use_container_width=True)

    # Ultimas apostas
    st.subheader("Últimas Apostas")
    display_cols = [c for c in ["date", "track_raw", "selection", "entry_type", "avg_odds", "stake", "profit_loss"] if c in history.columns]
    last20 = history.sort_values("date", ascending=False).head(20)[display_cols]

    def _color_row(row: pd.Series) -> list[str]:
        color = "background-color: #1a3a1a" if row.get("profit_loss", 0) > 0 else "background-color: #3a1a1a"
        return [color] * len(row)

    st.dataframe(
        last20.style.apply(_color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --- Secao 4: URL do feed ---
st.subheader("Feed de Sinais — BF Bot Manager")

import os
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
_token = os.getenv("API_FEED_TOKEN", "TOKEN_NAO_CONFIGURADO")

st.info(
    "Configure o BF Bot Manager com as URLs abaixo para receber os sinais diários automaticamente."
)
col_g, col_h = st.columns(2)
col_g.code(f"http://SEU_IP:8000/feed/{_token}/galgos.csv", language=None)
col_h.code(f"http://SEU_IP:8000/feed/{_token}/cavalos.csv", language=None)
st.caption(
    "Substitua **SEU_IP** pelo endereço IP da VPS. "
    "Em caso de dúvida sobre a configuração, consulte o guia de configuração do BF Bot Manager."
)
