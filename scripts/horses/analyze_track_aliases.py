"""
Analisa CSVs de resultado de cavalos (data/horses/Result), extrai menu_hint,
normaliza nome da pista e gera relatorio JSON de aliases em
data/horses/reports/track_alias_report.json.

Uso:
  python scripts/horses/analyze_track_aliases.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.horses.config import settings


def extract_base_track(menu_hint: str) -> str:
    text = menu_hint or ""
    match = re.match(r"^([A-Za-z\s/\\-]+?)(?:\s*\d|$)", text)
    base = match.group(1) if match else text
    return base.strip()


def main() -> None:
    from src.horses.utils.text import normalize_track_name  # pylint: disable=import-outside-top-level

    result_dir = settings.RAW_RESULT_DIR
    if not result_dir.exists():
        raise SystemExit(f"Diretorio nao encontrado: {result_dir}")

    pattern = "*.csv"
    files = sorted(result_dir.glob(pattern))
    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado com padrao {pattern} em {result_dir}")

    norm_to_raw: dict[str, Counter[str]] = defaultdict(Counter)

    for csv_path in files:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=settings.CSV_ENCODING,
                engine="python",
                on_bad_lines="skip",
                usecols=["menu_hint"],
            )
        except Exception:
            continue

        for raw_hint in df["menu_hint"].dropna().astype(str):
            base = extract_base_track(raw_hint)
            normalized = normalize_track_name(base)
            norm_to_raw[normalized][base] += 1

    alias_report = {
        norm: {"variants": variants.most_common(), "total": sum(variants.values())}
        for norm, variants in sorted(norm_to_raw.items())
        if len(variants) > 1
    }

    reports_dir = settings.DATA_DIR / "reports"
    output_path = reports_dir / "track_alias_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(alias_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Relatorio escrito em: {output_path}")


if __name__ == "__main__":
    main()
