"""
Checks rapidos no parquet bruto e (no Streamlit) no enriched.

Uso:
  python scripts/horses/check_parquet_and_enriched.py --path <arquivo.parquet>
  python scripts/horses/check_parquet_and_enriched.py --path <arquivo.parquet> --validate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")


def check_raw_parquet(path: Path) -> None:
    """Checks no parquet/CSV bruto (sem enriquecimento)."""
    if not path.exists():
        print("Arquivo nao encontrado:", path)
        return
    df = _load_df(path)
    print("=== Parquet/CSV bruto ===")
    print("path:", path)
    print("rows:", len(df))
    if df.empty:
        print("(dataset vazio)")
        return
    if "race_time_iso" in df.columns:
        iso = df["race_time_iso"].astype(str).str.strip()
        empty_iso = (iso == "") | iso.isna()
        print("race_time_iso empty %:", f"{empty_iso.mean() * 100:.2f}%")
    else:
        print("(coluna race_time_iso ausente)")
    print("colunas:", list(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Checks rapidos em parquet de sinais (e opcional validacao ROI/escala).")
    parser.add_argument("--path", required=True, help="Caminho para o parquet (ou CSV) de sinais.")
    parser.add_argument("--validate", action="store_true", help="Rodar validacao de ROI/escala (quick_checks) no arquivo.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Arquivo nao encontrado: {path}")

    check_raw_parquet(path)

    if args.validate:
        from scripts.horses.quick_checks import main as quick_checks_main
        quick_checks_main(["--path", str(path)])


if __name__ == "__main__":
    main()
