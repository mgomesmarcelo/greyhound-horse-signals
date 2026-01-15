"""
Utilitários para lidar com colunas de unidades (ref = 1) e legado (_fixed_10).

Regra:
- Se o dataset já tem colunas *_ref, usamos fator de referência = 1.
- Caso contrário, assumimos o legado *_fixed_10 com fator = 10.
- get_scale(base_units, ref_factor) reescala valores para a unidade solicitada.
"""

from __future__ import annotations

import pandas as pd


def get_ref_factor(df: pd.DataFrame) -> float:
    """
    Retorna 1.0 se ao menos uma coluna canônica *_ref existir (stake_ref, pnl_stake_ref, liability_ref),
    caso contrário 10.0 (legado).
    """
    canonical_refs = {"stake_ref", "pnl_stake_ref", "liability_ref"}
    has_ref = any(col in canonical_refs for col in df.columns)
    return 1.0 if has_ref else 10.0


def get_col(df: pd.DataFrame, preferred: str, fallback: str):
    """
    Retorna a série preferida se existir, senão a série de fallback.
    Lança KeyError se nenhuma das duas colunas existir.
    """
    if preferred in df.columns:
        return df[preferred]
    if fallback in df.columns:
        return df[fallback]
    raise KeyError(f"Nenhuma das colunas foi encontrada: {preferred}, {fallback}")


def get_scale(base_units: float, ref_factor: float) -> float:
    """Retorna o fator de escala para reexpressar valores na unidade desejada."""
    return float(base_units) / float(ref_factor)


def to_bool_series(s: pd.Series) -> pd.Series:
    """
    Normaliza Series para bool: aceita bool, números (1/0), strings em {"true","1","t","yes","y","sim"}.
    NaN vira False.
    """
    if s is None:
        return pd.Series(dtype=bool)
    if s.dtype == bool:
        return s
    true_tokens = {"true", "1", "t", "yes", "y", "sim"}
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).apply(lambda v: bool(v == 1))
    return s.fillna("").astype(str).str.strip().str.lower().apply(lambda v: v in true_tokens)


if __name__ == "__main__":
    # Cobertura mínima: asserts simples
    df_ref = pd.DataFrame({"stake_ref": [1, 2], "pnl_stake_ref": [0.1, -0.2]})
    df_ref_partial = pd.DataFrame({"liability_ref": [1.5, 2.5]})
    df_legacy = pd.DataFrame({"stake_fixed_10": [10, 20], "pnl_stake_fixed_10": [1.0, -2.0]})

    assert get_ref_factor(df_ref) == 1.0
    assert get_ref_factor(df_ref_partial) == 1.0
    assert get_ref_factor(df_legacy) == 10.0

    assert get_col(df_ref, "stake_ref", "stake_fixed_10").tolist() == [1, 2]
    assert get_col(df_legacy, "stake_ref", "stake_fixed_10").tolist() == [10, 20]

    # to_bool_series
    assert to_bool_series(pd.Series([True, False])).tolist() == [True, False]
    assert to_bool_series(pd.Series([1, 0, 2])).tolist() == [True, False, False]
    assert to_bool_series(pd.Series(["true", "SIM", "no", None])).tolist() == [True, True, False, False]
    print("units_helper checks passed.")

    assert get_scale(1.0, 1.0) == 1.0
    assert get_scale(1.0, 10.0) == 0.1

    print("units_helper checks passed.")

