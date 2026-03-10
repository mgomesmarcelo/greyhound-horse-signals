"""
Checagens rapidas de invariancia de ROI e escala (stake/liability) para sinais de cavalos.

Uso:
  python scripts/horses/quick_checks.py --path <signals.parquet|csv>

Valida:
- ROI nao muda ao reescalar base_units de 1 para 10.
- PnL e base escalam em 10x entre base_units=1 e base_units=10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.horses.config import settings
from scripts.horses.units_helper import get_ref_factor, get_scale, get_col


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    read_kwargs = {
        "encoding": settings.CSV_ENCODING,
        "engine": "python",
        "on_bad_lines": "skip",
    }
    return pd.read_csv(path, **read_kwargs)


def _compute_metrics(df: pd.DataFrame, entry_kind: str, base_units: float) -> dict[str, float] | None:
    if "entry_type" not in df.columns:
        return None
    block = df[df["entry_type"] == entry_kind].copy()
    if block.empty:
        return None

    ref_factor = get_ref_factor(block)
    scale = get_scale(base_units, ref_factor)

    try:
        stake_series = get_col(block, "stake_ref", "stake_fixed_10")
        pnl_stake_series = get_col(block, "pnl_stake_ref", "pnl_stake_fixed_10")
    except KeyError:
        return None

    liability_series = None
    pnl_liab_series = None
    if entry_kind == "lay":
        if "liability_ref" in block.columns or "liability_fixed_10" in block.columns:
            try:
                liability_series = get_col(block, "liability_ref", "liability_fixed_10")
                pnl_liab_series = get_col(block, "pnl_liability_ref", "pnl_liability_fixed_10")
            except KeyError:
                pass

    base_stake = float(stake_series.sum()) * scale
    pnl_stake = float(pnl_stake_series.sum()) * scale
    roi_stake = pnl_stake / base_stake if base_stake > 0 else 0.0

    metrics = {
        "base_stake": base_stake,
        "pnl_stake": pnl_stake,
        "roi_stake": roi_stake,
    }

    if entry_kind == "lay" and liability_series is not None and pnl_liab_series is not None:
        base_liab = float(liability_series.sum()) * scale
        pnl_liab = float(pnl_liab_series.sum()) * scale
        roi_liab = pnl_liab / base_liab if base_liab > 0 else 0.0
        metrics["base_liab"] = base_liab
        metrics["pnl_liab"] = pnl_liab
        metrics["roi_liab"] = roi_liab
    elif entry_kind == "lay":
        metrics["base_liab"] = 0.0
        metrics["pnl_liab"] = 0.0
        metrics["roi_liab"] = 0.0

    return metrics


def _assert_close(a: float, b: float, tol: float, label: str) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{label} divergiu: {a} vs {b} (tol={tol})")


def _check_invariants(m1: dict[str, float], m10: dict[str, float], entry_kind: str, tol: float = 1e-9) -> None:
    _assert_close(m1["roi_stake"], m10["roi_stake"], tol, f"ROI stake ({entry_kind})")

    if m1["base_stake"] or m10["base_stake"]:
        _assert_close(m10["base_stake"], m1["base_stake"] * 10.0, 1e-6, f"base_stake scale ({entry_kind})")
    if m1["pnl_stake"] or m10["pnl_stake"]:
        _assert_close(m10["pnl_stake"], m1["pnl_stake"] * 10.0, 1e-6, f"pnl_stake scale ({entry_kind})")

    if entry_kind == "lay" and "roi_liab" in m1:
        if m1["base_liab"] or m10["base_liab"]:
            _assert_close(m10["base_liab"], m1["base_liab"] * 10.0, 1e-6, "base_liab scale (lay)")
        if m1["pnl_liab"] or m10["pnl_liab"]:
            _assert_close(m10["pnl_liab"], m1["pnl_liab"] * 10.0, 1e-6, "pnl_liab scale (lay)")
        _assert_close(m1["roi_liab"], m10["roi_liab"], tol, "ROI liability (lay)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checagens rapidas de ROI/escala para sinais de cavalos.")
    parser.add_argument("--path", required=True, help="Caminho para signals_*.parquet ou CSV.")
    parser.add_argument("--verbose", action="store_true", help="Mostra bases/PnL para base_units=1 e 10.")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Arquivo nao encontrado: {path}")

    df = _load_df(path)
    if df.empty:
        raise SystemExit("Dataset vazio; nada a validar.")

    results = {}
    for entry_kind in ("back", "lay"):
        m1 = _compute_metrics(df, entry_kind, base_units=1.0)
        m10 = _compute_metrics(df, entry_kind, base_units=10.0)
        if m1 is None or m10 is None:
            continue
        _check_invariants(m1, m10, entry_kind)
        if args.verbose:
            print(f"[{entry_kind}] base=1 -> stake={m1['base_stake']:.4f}, pnl_stake={m1['pnl_stake']:.4f}, roi_stake={m1['roi_stake']:.6f}")
            print(f"[{entry_kind}] base=10 -> stake={m10['base_stake']:.4f}, pnl_stake={m10['pnl_stake']:.4f}, roi_stake={m10['roi_stake']:.6f}")
            if entry_kind == "lay" and "base_liab" in m1:
                print(f"[{entry_kind}] base=1 liab={m1['base_liab']:.4f}, pnl_liab={m1['pnl_liab']:.4f}, roi_liab={m1['roi_liab']:.6f}")
                print(f"[{entry_kind}] base=10 liab={m10['base_liab']:.4f}, pnl_liab={m10['pnl_liab']:.4f}, roi_liab={m10['roi_liab']:.6f}")
        results[entry_kind] = {"base1": m1, "base10": m10}

    if not results:
        raise SystemExit("Nenhum bloco BACK/LAY com colunas stake encontrado para validar.")

    print("Quick checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
