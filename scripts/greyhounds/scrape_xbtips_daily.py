import sys
import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.greyhounds.xbtips_scraper import XBTipsScraper

def main():
    print("Iniciando modo DIARIO (Hoje e Amanha)")
    scraper = XBTipsScraper(headless=True)
    
    dates = [
        datetime.date.today(),
        datetime.date.today() + datetime.timedelta(days=1)
    ]
    
    out_dir = PROJECT_ROOT / "data" / "greyhounds" / "xbtips" / "raw"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        csv_path = out_dir / f"xbtips_raw_{date_str}.csv"
        
        print(f"[{date_str}] Extraindo...")
        try:
            df = scraper.scrape_date(date_str)
            if not df.empty:
                df.to_csv(csv_path, index=False)
                print(f"[{date_str}] Salvo: {len(df)} registros.")
            else:
                print(f"[{date_str}] Sem dados.")
        except Exception as e:
            print(f"[{date_str}] Erro ao extrair: {e}")

if __name__ == "__main__":
    main()
