"""
Gera o indice de sinais historicos para cruzamento com CSV da Betfair.

Resultado: data/signal_index.parquet com colunas:
  date       : str YYYY-MM-DD
  track_norm : str nome da pista normalizado
  race_hhmm  : str HH:MM da hora da corrida
  sport      : str 'greyhounds' ou 'horses'

Uso:
  python scripts/build_signal_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.greyhounds.utils.text import normalize_track_name

COLS_NEEDED = ["date", "track_name", "race_time_iso"]

GREYHOUNDS_SIGNALS_DIR = PROJECT_ROOT / "data" / "greyhounds" / "signals"
HORSES_SIGNALS_DIR = PROJECT_ROOT / "data" / "horses" / "signals"
OUTPUT_PATH = PROJECT_ROOT / "data" / "signal_index.parquet"


def _extract_hhmm(race_time_iso: pd.Series) -> pd.Series:
    """Extrai HH:MM de coluna no formato 2025-09-06T17:25."""
    return race_time_iso.astype(str).str.extract(r"T(\d{2}:\d{2})")[0]


def _load_signals_dir(signals_dir: Path, sport: str) -> pd.DataFrame:
    frames = []
    csv_files = sorted(signals_dir.glob("*.csv"))
    print(f"  {sport}: {len(csv_files)} arquivo(s) encontrado(s)")
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=COLS_NEEDED, low_memory=False)
            df = df.dropna(subset=["date", "track_name", "race_time_iso"])
            df["sport"] = sport
            frames.append(df)
        except Exception as e:
            print(f"    Aviso: {f.name} ignorado — {e}")
    if not frames:
        return pd.DataFrame(columns=COLS_NEEDED + ["sport"])
    return pd.concat(frames, ignore_index=True)


def build_index() -> None:
    print("Construindo indice de sinais...")

    dfs = []
    for signals_dir, sport in [
        (GREYHOUNDS_SIGNALS_DIR, "greyhounds"),
        (HORSES_SIGNALS_DIR, "horses"),
    ]:
        if signals_dir.exists():
            dfs.append(_load_signals_dir(signals_dir, sport))
        else:
            print(f"  Aviso: {signals_dir} nao encontrado — ignorando")

    if not dfs:
        print("Nenhum dado encontrado. Abortando.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total de linhas brutas: {len(df):,}")

    # Normaliza pista usando a mesma funcao do projeto
    df["track_norm"] = df["track_name"].apply(normalize_track_name)

    # Extrai HH:MM da hora da corrida
    df["race_hhmm"] = _extract_hhmm(df["race_time_iso"])

    # Remove linhas sem hora valida
    df = df.dropna(subset=["track_norm", "race_hhmm"])
    df = df[df["track_norm"].str.strip() != ""]

    # Mantem so as colunas do indice
    df = df[["date", "track_norm", "race_hhmm", "sport"]].copy()

    # Deduplica — uma entrada por corrida sinalizada
    df = df.drop_duplicates()
    print(f"Total apos deduplicacao: {len(df):,}")

    # Salva
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Indice salvo em: {OUTPUT_PATH}")
    print(f"Tamanho: {size_kb:.1f} KB")
    print("Pronto.")


if __name__ == "__main__":
    build_index()
