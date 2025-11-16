import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.horses.run_daily import main as run_horses_daily
from src.greyhounds.run_daily import main as run_greyhounds_daily


def main():
    run_horses_daily()
    run_greyhounds_daily()
    print("Pipeline diario concluido com sucesso.")


if __name__ == "__main__":
    main()
