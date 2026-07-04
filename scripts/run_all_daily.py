import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.greyhounds.run_daily import main as run_greyhounds_daily
from src.horses.run_daily import main as run_horses_daily

def run_script(script_path: str, *args):
    print(f"\n--- Executando {script_path} {' '.join(args)} ---")
    script_full_path = str(project_root / script_path)
    cmd = [sys.executable, script_full_path] + list(args)
    subprocess.run(cmd, check=True)

def main():
    print("--- 1. Raspagem e Geracao de Sinais do Dia ---")
    run_greyhounds_daily()
    run_horses_daily()
    
    print("\n--- 2. Download Oficial da Betfair ---")
    run_script("scripts/download_betfair_prices.py")

    print("\n--- 3. Limpeza dos Resultados Betfair ---")
    run_script("scripts/greyhounds/clean_greyhound_results.py")
    run_script("scripts/horses/clean_horse_results.py")
    
    print("\n--- 4. Conversao Raw Data para Parquet ---")
    run_script("scripts/greyhounds/convert_greyhound_history.py")
    run_script("scripts/horses/convert_horse_history.py")

    print("\n--- 5. Geracao de Sinais Historicos (BACKTESTING) ---")
    run_script("scripts/greyhounds/generate_greyhound_signals.py")
    run_script("scripts/horses/generate_horse_signals.py")

    print("\n--- 6. Conversao dos Sinais Gerados para Parquet ---")
    run_script("scripts/greyhounds/convert_greyhound_history.py", "--dataset", "signals")
    run_script("scripts/horses/convert_horse_history.py", "--dataset", "signals")
    
    print("\n--- 7. Atualizacao de Indices Consolidados ---")
    run_script("scripts/horses/build_horses_result_index.py")
    run_script("scripts/horses/build_horses_signals_enriched.py")

    print("\nPipeline diario concluido com sucesso!")

if __name__ == "__main__":
    main()
