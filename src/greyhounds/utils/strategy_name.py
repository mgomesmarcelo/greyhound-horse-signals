"""Formatação do nome de estratégia para export/import (sem data)."""
from __future__ import annotations

from typing import List, Optional


def format_strategy_name(
    rule_slug: str,
    source_slug: str,
    market: str,
    entry_type: str,
    tracks_ms: Optional[List[str]] = None,
    cats_ms: Optional[List[str]] = None,
    bsp_low: Optional[float] = None,
    bsp_high: Optional[float] = None,
    pnl: Optional[float] = None,
    roi: Optional[float] = None,
) -> str:
    """
    Gera strategy_name no formato:
    "{rule} • {source} • {market} • {entry_type} • pistas:{N/ALL} • cats:{.../ALL} • BSP:{low–high/-}"
    Sem data. Separador: " • ".
    PNL/ROI não são incluídos (calculados apenas no dashboard).
    """
    sep = " • "
    rule = (rule_slug or "").strip()
    source = (source_slug or "").strip()
    market_s = (market or "").strip().lower()
    entry = (entry_type or "").strip().lower()
    if entry in ("ambos", "both"):
        entry = "both"

    if tracks_ms is not None and len(tracks_ms) > 0:
        pistas_s = f"pistas:{len(tracks_ms)}"
    else:
        pistas_s = "pistas:ALL"

    if cats_ms is not None and len(cats_ms) > 0:
        cats_s = "cats:" + ",".join(str(c) for c in sorted(cats_ms))
    else:
        cats_s = "cats:ALL"

    if bsp_low is not None and bsp_high is not None:
        bsp_s = f"BSP:{bsp_low:.1f}\u2013{bsp_high:.1f}"
    else:
        bsp_s = "BSP:-"

    parts = [rule, source, market_s, entry, pistas_s, cats_s, bsp_s]
    return sep.join(parts)
