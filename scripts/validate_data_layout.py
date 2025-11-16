from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "data"

PATTERNS = {
    # Greyhounds
    ("greyhounds", "race_links"): r"^race_links_\d{4}-\d{2}-\d{2}\.csv$",
    ("greyhounds", "TimeformForecast"): r"^TimeformForecast_\d{4}-\d{2}-\d{2}\.csv$",
    ("greyhounds", "timeform_top3"): r"^timeform_top3_\d{4}-\d{2}-\d{2}\.csv$",

    # Horses
    ("horses", "betfair_top3"): r"^betfair_top3_\d{4}-\d{2}-\d{2}\.csv$",
    ("horses", "TimeformForecast"): r"^TimeformForecast_\d{4}-\d{2}-\d{2}\.csv$",
}

# Pastas que não deveriam mais existir (sob greyhounds)
DEPRECATED_DIRS = [
    ("greyhounds", "2025-11-07"),
    ("greyhounds", "2025-11-08"),
    # adicione aqui outras datas/pastas antigas se aparecerem
]

def check_layout() -> int:
    issues = 0
    print(f"Base: {BASE}")

    # 1) Verifica padrões de nome
    for (sport, subdir), pattern in PATTERNS.items():
        d = BASE / sport / subdir
        if not d.exists():
            print(f"[WARN] Diretório ausente: {d}")
            continue

        reg = re.compile(pattern)
        for f in sorted(d.glob("*.csv")):
            if not reg.match(f.name):
                print(f"[NAME] {f} não bate com padrão {pattern}")
                issues += 1

    # 2) Sinaliza diretórios obsoletos em greyhounds
    for sport, subdir in DEPRECATED_DIRS:
        d = BASE / sport / subdir
        if d.exists():
            has_files = any(d.rglob("*"))
            print(f"[DEPRECATED] {d} ainda existe "
                  f"{'(com arquivos)' if has_files else '(vazio)'}")
            issues += 1

    # 3) Validação básica dos CSVs (não vazios e com header)
    for sport, subdir in {k for k in PATTERNS.keys()}:
        d = BASE / sport / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(f, nrows=5)  # leitura leve
                # header existe se len(columns) > 0
                if df.shape[1] == 0:
                    print(f"[CSV] {f} sem colunas.")
                    issues += 1
                # vamos checar se tem ao menos 1 linha além do header
                df_full = pd.read_csv(f)
                if df_full.shape[0] == 0:
                    print(f"[CSV] {f} sem linhas (apenas header).")
                    issues += 1
            except Exception as e:
                print(f"[CSV] Falha lendo {f}: {e}")
                issues += 1

    # 4) Checagem leve de colunas esperadas (tolerante)
    expected_cols = {
        ("greyhounds", "timeform_top3"): {"track_name", "race_time_iso"},
        ("greyhounds", "TimeformForecast"): {"track_name", "race_time_iso"},
        ("greyhounds", "race_links"): {"track_name", "race_time_iso"},
        ("horses", "betfair_top3"): {"track_name", "race_time_iso"},
        ("horses", "TimeformForecast"): {"track_name", "race_time_iso"},
    }
    for key, cols in expected_cols.items():
        sport, subdir = key
        d = BASE / sport / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(f, nrows=1)
                missing = cols - set(map(str, df.columns))
                if missing:
                    # só alerta: alguns arquivos antigos podem ter outra schema
                    print(f"[SCHEMA] {f} faltando colunas esperadas: {sorted(missing)}")
            except Exception as e:
                print(f"[SCHEMA] Falha lendo {f}: {e}")
                issues += 1

    print("\nResumo:",
          "OK sem problemas." if issues == 0 else f"{issues} ponto(s) a revisar.")
    return 0 if issues == 0 else 1

if __name__ == "__main__":
    raise SystemExit(check_layout())
