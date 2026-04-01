import sys
import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.greyhounds.xbtips_scraper import XBTipsScraper

def main():
    print("Iniciando modo HISOTRICO (Jan 2025 ate Hoje)")
    # Headless falso para ver o navegador na primeira vez e logar
    scraper = XBTipsScraper(headless=False)
    
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date.today()
    
    out_dir = PROJECT_ROOT / "data" / "greyhounds" / "xbtips" / "raw"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    curr = start_date
    while curr <= end_date:
        date_str = curr.strftime("%Y-%m-%d")
        csv_path = out_dir / f"xbtips_raw_{date_str}.csv"
        
        if csv_path.exists():
            print(f"[{date_str}] Ja extraido. Pulando...")
        else:
            try:
                df = scraper.scrape_date(date_str)
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    print(f"[{date_str}] Salvo: {len(df)} registros.")
                else:
                    print(f"[{date_str}] Nenhum dado encontrado.")
            except Exception as e:
                print(f"[{date_str}] Erro ao extrair: {e}")
                
        curr += datetime.timedelta(days=1)
        
if __name__ == "__main__":
    main()
