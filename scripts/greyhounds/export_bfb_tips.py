import argparse
import csv
import datetime
from pathlib import Path
import re

import sys
from loguru import logger

# Make sure we can import src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.greyhounds.config import settings
from src.greyhounds.analysis.signals import _parse_forecast_all

def main():
    parser = argparse.ArgumentParser(description="Gera CSV de Entradas para o BF Bot Manager baseado no Timeform Forecast.")
    parser.add_argument("--date", type=str, help="Data no formato YYYY-MM-DD (Padrão: Hoje)")
    parser.add_argument("--type", type=str, choices=["back", "lay"], default="back", help="Tipo de aposta (back ou lay)")
    
    # Parâmetros da sua estratégia de Forecast
    parser.add_argument("--value_ratio_min", type=float, default=1.00, help="Value ratio mínimo para Back (Ex: 1.20)")
    parser.add_argument("--value_ratio_max", type=float, default=2.00, help="Limite do dashboard (ratio máximo, ex: 2.0)")
    parser.add_argument("--global_low", type=float, default=5.0, help="BSP Mínimo Absoluto (Ex: 5.0)")
    parser.add_argument("--global_high", type=float, default=12.0, help="BSP Máximo Absoluto (Ex: 12.0)")
    
    args = parser.parse_args()
    
    target_date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Busca o arquivo de Forecast gerado pelo scraper hoje
    raw_tf_dir = settings.RAW_TIMEFORM_FORECAST_DIR
    tf_file = raw_tf_dir / f"TimeformForecast_{target_date}.csv"
    
    if not tf_file.exists():
        logger.error(f"Arquivo não encontrado: {tf_file}. Você rodou o scraper do Timeform hoje?")
        sys.exit(1)
        
    out_dir = settings.DATA_DIR / "bfb_tips"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_csv = out_dir / f"bfb_tips_{args.type}_{target_date}.csv"
    
    headers = ["Date", "Time", "Course", "Selection", "BetType", "MinPrice", "MaxPrice"]
    rows_to_write = []
    
    with open(tf_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track = row.get("track_name", "").strip()
            race_iso = row.get("race_time_iso", "")
            forecast_raw = row.get("TimeformForecast", "")
            
            if not track or not race_iso or not forecast_raw:
                continue
                
            # Formata data e hora para o BF Bot Manager (Data: DD/MM/YYYY, Time: HH:MM)
            try:
                dt = datetime.datetime.fromisoformat(race_iso)
                uk_date = dt.strftime("%d/%m/%Y")
                uk_time = dt.strftime("%H:%M")
            except ValueError:
                continue
                
            items = _parse_forecast_all(forecast_raw)
            for item in items:
                f_odds = float(item["forecast_odds"])
                sel_clean = item["forecast_name_clean"]
                sel_raw = item["forecast_name_raw"]
                
                if args.type == "back":
                    # Cálculo exato conforme discutimos:
                    # Teto absoluto do mercado ou limite percentual (2x), o que for menor.
                    calculated_max = f_odds * args.value_ratio_max
                    final_max = min(args.global_high, calculated_max)
                    
                    # Piso absoluto ou mínimo exigido pelo ratio, o que for maior
                    calculated_min = f_odds * args.value_ratio_min
                    final_min = max(args.global_low, calculated_min)
                    
                    if final_min <= final_max:
                        rows_to_write.append({
                            "Date": uk_date,
                            "Time": uk_time,
                            "Course": track,
                            "Selection": sel_raw,
                            "BetType": "Back",
                            "MinPrice": round(final_min, 2),
                            "MaxPrice": round(final_max, 2)
                        })
                else:
                    # Lógica do Lay (Inversa)
                    # MinPrice do Lay (Para pegar lays bons, pagando pouco, a odd tem que ser BAIXA)
                    # Exemplo simplificado (você pode customizar o args.value_ratio_min depois)
                    calculated_max_lay = f_odds / args.value_ratio_min
                    calculated_min_lay = f_odds / args.value_ratio_max
                    
                    final_max = min(args.global_high, calculated_max_lay)
                    final_min = max(args.global_low, calculated_min_lay)
                    
                    if final_min <= final_max:
                        rows_to_write.append({
                            "Date": uk_date,
                            "Time": uk_time,
                            "Course": track,
                            "Selection": sel_raw,
                            "BetType": "Lay",
                            "MinPrice": round(final_min, 2),
                            "MaxPrice": round(final_max, 2)
                        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_to_write)
        
    logger.info(f"Gerado {len(rows_to_write)} dicas para {args.type.upper()} no arquivo: {out_csv}")

if __name__ == "__main__":
    main()
