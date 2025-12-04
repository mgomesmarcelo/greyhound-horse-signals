from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def extract_base_track(menu_hint: str) -> str:
    text = menu_hint or ""
    match = re.match(r"^([A-Za-z\s/\\-]+?)(?:\s*\d|$)", text)
    base = match.group(1) if match else text
    return base.strip()


def main() -> None:
    root = project_root()
    sys.path.append(str(root / "src"))

    from src.greyhounds.utils.text import normalize_track_name  # pylint: disable=import-outside-top-level

    result_dir = root / "data" / "greyhounds" / "Result"
    if not result_dir.exists():
        raise SystemExit(f"Diretório não encontrado: {result_dir}")

    pattern = "dwbfgreyhound*.csv"
    files = sorted(result_dir.glob(pattern))
    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado com padrão {pattern}")

    norm_to_raw: dict[str, Counter[str]] = defaultdict(Counter)
    raw_only: Counter[str] = Counter()

    for csv_path in files:
        try:
            df = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                engine="python",
                on_bad_lines="skip",
                usecols=["menu_hint"],
            )
        except Exception:
            continue

        for raw_hint in df["menu_hint"].dropna().astype(str):
            raw_only[raw_hint] += 1
            base = extract_base_track(raw_hint)
            normalized = normalize_track_name(base)
            norm_to_raw[normalized][base] += 1

    alias_report = {
        norm: {"variants": variants.most_common(), "total": sum(variants.values())}
        for norm, variants in sorted(norm_to_raw.items())
        if len(variants) > 1
    }

    output_path = root / "scripts" / "track_alias_report.json"
    output_path.write_text(
        json.dumps(alias_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Relatório escrito em: {output_path}")


if __name__ == "__main__":
    main()

