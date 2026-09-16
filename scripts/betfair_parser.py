"""
Funcoes utilitarias para parsear CSV de historico de apostas da Betfair
e cruzar com o indice de sinais do sistema.

Importavel independentemente do Streamlit.
"""
from __future__ import annotations

import datetime
import io
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.greyhounds.utils.text import normalize_track_name

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_BET_PLACED_RE = re.compile(
    r"(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{2,4})\s+(\d{2}:\d{2})",
)
_RACE_TIME_RE = re.compile(r"/\s*(\d{2}:\d{2})\b")
_TRACK_DATE_SUFFIX_RE = re.compile(
    r"\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3}.*$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_bet_date(bet_placed: str) -> str | None:
    """Extrai YYYY-MM-DD de 'Bet placed' (ex: '31-Aug-26 21:22')."""
    m = _BET_PLACED_RE.search(str(bet_placed))
    if not m:
        return None
    day, mon_str, year_str, _ = m.groups()
    month = _MONTH_MAP.get(mon_str.lower())
    if month is None:
        return None
    year = int(year_str)
    if year < 100:
        year += 2000
    try:
        return datetime.date(year, month, int(day)).isoformat()
    except ValueError:
        return None


def _extract_race_hhmm(market: str) -> str | None:
    """Extrai HH:MM da corrida da coluna Market."""
    m = _RACE_TIME_RE.search(str(market))
    return m.group(1) if m else None


def _extract_track(market: str) -> str:
    """Extrai nome da pista da coluna Market."""
    parts = str(market).split("/")
    if len(parts) >= 2:
        raw = parts[1].strip()
        clean = _TRACK_DATE_SUFFIX_RE.sub("", raw).strip()
        if clean:
            return clean
    return ""


def parse_betfair_csv(content: bytes) -> pd.DataFrame:
    """
    Parseia CSV de historico de apostas da Betfair.

    Parametros
    ----------
    content : bytes
        Conteudo binario do CSV.

    Retorna
    -------
    pd.DataFrame com colunas padronizadas incluindo campos de cruzamento.

    Raises
    ------
    ValueError
        Se colunas obrigatorias nao forem encontradas.
    """
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(content), encoding="latin-1")

    # Mapeia colunas para nomes canonicos
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "market":
            col_map[col] = "market"
        elif cl == "selection":
            col_map[col] = "selection"
        elif "bid type" in cl or "bid_type" in cl:
            col_map[col] = "bid_type"
        elif "bet id" in cl or "bet_id" in cl:
            col_map[col] = "bet_id"
        elif "bet placed" in cl or "bet_placed" in cl:
            col_map[col] = "bet_placed"
        elif "profit" in cl and "loss" in cl:
            col_map[col] = "profit_loss"
        elif "avg" in cl and "odds" in cl:
            col_map[col] = "avg_odds"
        elif "stake" in cl and "(" in col:
            col_map[col] = "stake"
        elif "liability" in cl and "(" in col:
            col_map[col] = "liability"
        elif "odds req" in cl:
            col_map[col] = "odds_req"
    df = df.rename(columns=col_map)

    required = {"market", "bid_type", "bet_placed", "profit_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas nao encontradas no CSV: {missing}")

    # Campos de cruzamento
    df["date"] = df["bet_placed"].apply(_parse_bet_date)
    df["race_hhmm"] = df["market"].apply(_extract_race_hhmm)
    df["track_raw"] = df["market"].apply(_extract_track)
    df["track_norm"] = df["track_raw"].apply(normalize_track_name)
    df["entry_type"] = (
        df["bid_type"].str.strip().str.lower()
        .map({"lay": "lay", "back": "back", "l": "lay", "b": "back"})
    )

    # P&L numerico
    df["profit_loss"] = (
        df["profit_loss"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0.0)

    for col in ("stake", "liability"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                errors="coerce",
            ).fillna(0.0)
        else:
            df[col] = 0.0

    return df


# ---------------------------------------------------------------------------
# Cruzamento com indice de sinais
# ---------------------------------------------------------------------------

def filter_system_bets(df_bets: pd.DataFrame, signal_index: pd.DataFrame) -> pd.DataFrame:
    """
    Mantem apenas apostas cujo (date, track_norm, race_hhmm) esta no indice de sinais.

    Parametros
    ----------
    df_bets : pd.DataFrame
        Resultado de parse_betfair_csv().
    signal_index : pd.DataFrame
        Indice carregado de data/signal_index.parquet.

    Retorna
    -------
    pd.DataFrame filtrado.
    """
    df = df_bets.dropna(subset=["date", "track_norm", "race_hhmm"]).copy()
    key_set = set(
        zip(
            signal_index["date"].astype(str),
            signal_index["track_norm"],
            signal_index["race_hhmm"],
        )
    )
    mask = [
        (str(d), t, h) in key_set
        for d, t, h in zip(df["date"], df["track_norm"], df["race_hhmm"])
    ]
    return df[mask].copy()
