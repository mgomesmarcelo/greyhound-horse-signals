"""
Stop loss por dia (sessão): zera stake/pnl das apostas após o PnL acumulado do dia atingir o limiar.

Função pura, sem dependência do Streamlit. Retorna o dataframe modificado e estatísticas
(ex.: dias_stopados).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_RACE_TIME_ISO_FORMAT = "%Y-%m-%dT%H:%M"

# Colunas de stake/pnl a zerar quando a aposta for "parada" (só as que existirem no df).
_STAKE_PNL_COLS = [
    "stake_ref",
    "pnl_stake_ref",
    "stake_fixed_10",
    "pnl_stake_fixed_10",
]
_LIABILITY_COLS = [
    "liability_ref",
    "pnl_liability_ref",
    "liability_from_stake_ref",
    "stake_for_liability_ref",
    "liability_fixed_10",
    "pnl_liability_fixed_10",
    "liability_from_stake_fixed_10",
    "stake_for_liability_10",
]
_COLS_TO_ZERO = _STAKE_PNL_COLS + _LIABILITY_COLS


def _get_pnl_series(df: pd.DataFrame):
    """Retorna a série de PnL stake (ref ou fixed_10). None se nenhuma existir."""
    if "pnl_stake_ref" in df.columns:
        return pd.to_numeric(df["pnl_stake_ref"], errors="coerce").fillna(0.0)
    if "pnl_stake_fixed_10" in df.columns:
        return pd.to_numeric(df["pnl_stake_fixed_10"], errors="coerce").fillna(0.0)
    return None


def apply_daily_stop_loss(
    df: pd.DataFrame,
    threshold_units: float,
    time_col: str = "race_time_iso",
    date_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Aplica stop loss por dia: quando o PnL acumulado do dia atinge <= -threshold_units,
    as apostas seguintes daquele dia são zeradas (stake/pnl e liability).

    Ordena por data e tempo; agrupa por dia; acumula PnL por dia. Para cada aposta,
    se o acumulado do dia (antes desta aposta) já for <= -threshold, esta aposta é
    considerada não realizada (colunas de stake/pnl/liability zeradas).

    Parâmetros
    ----------
    df : DataFrame
        DataFrame de sinais com colunas de PnL e opcionalmente liability.
    threshold_units : float
        Limiar em unidades (positivo). Ex.: 10 = parar quando prejuízo do dia >= 10.
        Se <= 0, retorna o dataframe inalterado.
    time_col : str
        Coluna de data/hora para ordenação (ex.: race_time_iso).
    date_col : str, opcional
        Coluna de data (dia). Se None, é derivada de time_col.

    Retorno
    -------
    (df_out, stats) : tuple
        df_out : DataFrame com as mesmas linhas; colunas de stake/pnl/liability zeradas
                 onde o stop foi aplicado; coluna extra "stop_applied" (True onde zerou).
        stats  : dict com "dias_stopados" = número de dias em que o stop foi acionado.
    """
    stats: dict[str, Any] = {"dias_stopados": 0}
    if df is None or df.empty:
        return (df.copy() if df is not None else pd.DataFrame(), stats)
    if threshold_units is None or float(threshold_units) <= 0:
        out = df.copy()
        if "stop_applied" not in out.columns:
            out["stop_applied"] = False
        return (out, stats)

    threshold = float(threshold_units)
    out = df.copy()

    pnl_series = _get_pnl_series(out)
    if pnl_series is None:
        if "stop_applied" not in out.columns:
            out["stop_applied"] = False
        return (out, stats)

    if time_col not in out.columns:
        if "stop_applied" not in out.columns:
            out["stop_applied"] = False
        return (out, stats)

    ts = pd.to_datetime(out[time_col], errors="coerce")
    out["_ts"] = ts
    if "date" in out.columns:
        out["_date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out["_date"] = out["_date"].fillna(ts.dt.date)
    else:
        out["_date"] = ts.dt.date
    out = out.sort_values(["_date", "_ts"]).reset_index(drop=True)

    cols_to_zero = [c for c in _COLS_TO_ZERO if c in out.columns]
    stop_applied = [False] * len(out)
    days_stopped: set[Any] = set()

    current_date = None
    running_cum = 0.0
    stopped_today = False

    for i in range(len(out)):
        row_date = out.iloc[i]["_date"]
        if row_date != current_date:
            current_date = row_date
            running_cum = 0.0
            stopped_today = False

        if stopped_today:
            stop_applied[i] = True
            days_stopped.add(current_date)
            idx = out.index[i]
            for c in cols_to_zero:
                out.loc[idx, c] = 0.0
            continue

        pnl_val = float(pnl_series.iloc[i]) if pnl_series is not None else 0.0
        running_cum += pnl_val
        if running_cum <= -threshold:
            stopped_today = True
            days_stopped.add(current_date)

    out["stop_applied"] = stop_applied
    out.drop(columns=["_ts", "_date"], inplace=True, errors="ignore")
    stats["dias_stopados"] = len(days_stopped)
    return (out, stats)
