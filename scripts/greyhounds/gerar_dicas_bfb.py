import os
import sys
import csv
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

# Adiciona o diretório raiz ao sys.path para importar utilitários do projeto
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.greyhounds.utils.text import normalize_track_name
from src.greyhounds.analysis.signals import _parse_forecast_all

def _parse_forecast_string(forecast_str):
    """
    Parse a TimeformForecast string into a list of dictionaries.
    Uses centralized parser from signals.py.
    """
    items = []
    if not isinstance(forecast_str, str):
        return items
        
    parsed = _parse_forecast_all(forecast_str)
    for p in parsed:
        items.append({
            "forecast_rank": p["forecast_rank"],
            "forecast_odds": p["forecast_odds"],
            "forecast_name_raw": p["forecast_name_raw"],
            "forecast_name_clean": p["forecast_name_clean"]
        })
    return items

def aplicar_filtros(item_candidato, filtros):
    """
    Verifica se o candidato passa nos filtros da estratégia.
    """
    # Filtro de Rank
    rank_ms = filtros.get("forecast_rank_ms", [])
    if rank_ms and item_candidato["forecast_rank"] not in rank_ms:
        return False
        
    # Filtro de Tracks
    tracks_ms = [t.lower() for t in filtros.get("tracks_ms", [])]
    if tracks_ms and item_candidato["course"].lower() not in tracks_ms:
        return False
        
    # Filtro de Runners (calculado com base na qtd de cães no forecast)
    num_runners_ms = filtros.get("num_runners_ms", [])
    if num_runners_ms and item_candidato["num_runners"] not in num_runners_ms:
        return False
        
    # Filtro de Weekdays
    weekdays_ms = [w.lower() for w in filtros.get("weekdays_ms", [])]
    if weekdays_ms:
        # datetime.weekday() -> 0=Seg, 1=Ter, ..., 6=Dom
        dias_semana_map = {0: "seg", 1: "ter", 2: "qua", 3: "qui", 4: "sex", 5: "sab", 6: "dom"}
        dia_atual = dias_semana_map[item_candidato["date_obj"].weekday()]
        if dia_atual not in weekdays_ms:
            return False
            
    # Filtro de Trap
    trap_ms = filtros.get("trap_ms", [])
    if trap_ms and item_candidato.get("trap") not in trap_ms:
        return False
        
    # Filtro de Category
    cats_ms = [c.lower() for c in filtros.get("cats_ms", [])]
    if cats_ms:
        cat_raw = item_candidato.get("category")
        cat = str(cat_raw).lower() if cat_raw else ""
        cat_letter = re.sub(r'[^a-z]', '', cat)
        if not cat or (cat not in cats_ms and cat_letter not in cats_ms):
            return False

    return True

def processar_estrategia(estrategia, item_candidato):
    """
    Calcula MinPrice e MaxPrice baseados na interseção de dois limites:
    1. Limite absoluto da estratégia (bsp_low e bsp_high)
    2. Limite dinâmico baseado no Forecast (O_f * vr_min e O_f * vr_max)
    """
    f_odds = item_candidato["forecast_odds"]
    filtros = estrategia.get("filtros", {})
    
    vr_min = filtros.get("value_ratio_min", 0.0)
    vr_max = filtros.get("value_ratio_max", 999.0)
    bsp_low = filtros.get("bsp_low", 1.01)
    bsp_high = filtros.get("bsp_high", 1000.0)
    
    # Range calculado a partir do Forecast e Value Ratio
    min_price_from_vr = f_odds * vr_min
    max_price_from_vr = f_odds * vr_max
        
    # Interseção com os limites absolutos de odd (bsp_low e bsp_high)
    final_min_price = max(bsp_low, round(min_price_from_vr, 2))
    final_max_price = min(bsp_high, round(max_price_from_vr, 2))
    
    # Se min > max, significa que as faixas não se cruzam (a aposta é impossível de satisfazer os dois critérios)
    if final_min_price > final_max_price:
        return None
        
    mercado_raw = estrategia.get("mercado", "win").lower()
    mercado_map = {"win": "WIN", "placed": "PLACE"}
    market_type = mercado_map.get(mercado_raw, "WIN")
    
    return {
        "StartTime": item_candidato["date_obj"].strftime("%d/%m/%Y %H:%M:%S"),
        "MarketId": item_candidato.get("market_id", ""),
        "MarketType": market_type,
        "EventName": item_candidato["course"],
        "SelectionName": f"{item_candidato.get('trap')}. {item_candidato['forecast_name_raw']}" if item_candidato.get("trap") else item_candidato["forecast_name_raw"],
        "BetType": estrategia.get("tipo_entrada", "back").capitalize(),
        "MinPrice": final_min_price,
        "MaxPrice": final_max_price,
        "Provider": estrategia.get("nome", "Estrategia")
    }

def main():
    # 1. Carregar Configurações
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "bfb_tips_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    gerais = config.get("configuracoes_gerais", {})
    estrategias = config.get("estrategias", [])
    
    dir_forecast = gerais.get("diretorio_forecast")
    dir_output = gerais.get("diretorio_output_bfb")
    dir_logs = gerais.get("diretorio_logs_auditoria")
    dir_output_fixed = gerais.get("diretorio_output_bfb_fixed", os.path.join(os.path.dirname(dir_output), "bfb_tips_fixed"))
    
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_logs, exist_ok=True)
    os.makedirs(dir_output_fixed, exist_ok=True)
    
    # 2. Descobrir arquivo Forecast de Hoje
    hoje_str = datetime.now().strftime("%Y-%m-%d")
    
    # Para teste, vamos tentar ler o arquivo do dia (ou o mais recente como 2026-04-22)
    # Se rodar hoje, hoje é 2026-04-25, mas os arquivos disponíveis podem ser outros.
    # Vamos usar uma data fixa ou o último modificado para simulação.
    
    # Aqui, tentamos carregar o arquivo específico do dia
    arquivo_hoje = os.path.join(dir_forecast, f"TimeformForecast_{hoje_str}.csv")
    
    # FALLBACK para testes manuais (pegar o mais recente caso o de hoje não exista)
    if not os.path.exists(arquivo_hoje):
        arquivos = [f for f in os.listdir(dir_forecast) if f.startswith("TimeformForecast_") and f.endswith(".csv")]
        if not arquivos:
            print(f"Nenhum arquivo Forecast encontrado em {dir_forecast}.")
            return
        arquivos.sort()
        arquivo_hoje = os.path.join(dir_forecast, arquivos[-1])
        print(f"Arquivo de hoje não encontrado. Usando o mais recente: {arquivo_hoje}")
        
    # 2.5 Carregar Cartões (TimeformCards) para Traps e Categorias
    dir_cards = os.path.join(dir_forecast, "..", "timeform_cards")
    arquivo_cards = os.path.join(dir_cards, f"timeform_cards_{hoje_str}.csv")
    
    lookup_cards = {}
    if not os.path.exists(arquivo_cards):
        # Tenta fallback se o de hoje não existir
        arquivos_cards = [f for f in os.listdir(dir_cards) if f.startswith("timeform_cards_") and f.endswith(".csv")] if os.path.exists(dir_cards) else []
        if arquivos_cards:
            arquivos_cards.sort()
            arquivo_cards = os.path.join(dir_cards, arquivos_cards[-1])
            
    if os.path.exists(arquivo_cards):
        try:
            df_cards = pd.read_csv(arquivo_cards)
            for _, r in df_cards.iterrows():
                # course, time, clean_name -> trap, category
                c_name = re.sub(r'[^a-zA-Z0-9]', '', str(r["greyhound_name"])).lower()
                key = (str(r["track_name"]).strip().lower(), str(r["race_time_iso"]).strip(), c_name)
                lookup_cards[key] = {
                    "trap": int(r["trap"]) if pd.notnull(r["trap"]) else None,
                    "category": str(r["category"]).strip() if pd.notnull(r["category"]) else ""
                }
        except Exception as e:
            print(f"Aviso: Falha ao ler {arquivo_cards}: {e}")
            
    # 2.6 Carregar Links da Betfair (MarketId)
    dir_links = os.path.join(dir_forecast, "..", "race_links")
    arquivo_links = os.path.join(dir_links, f"race_links_{hoje_str}.csv")
    
    lookup_links = {}
    if not os.path.exists(arquivo_links):
        arquivos_links = [f for f in os.listdir(dir_links) if f.startswith("race_links_") and f.endswith(".csv")] if os.path.exists(dir_links) else []
        if arquivos_links:
            arquivos_links.sort()
            arquivo_links = os.path.join(dir_links, arquivos_links[-1])
            
    if os.path.exists(arquivo_links):
        try:
            df_links = pd.read_csv(arquivo_links)
            for _, r in df_links.iterrows():
                url = str(r.get("race_url", ""))
                if url:
                    m_id = url.split("/")[-1]
                    # track_name, race_time_iso
                    t_name = normalize_track_name(str(r["track_name"]))
                    r_time = str(r["race_time_iso"]).strip()
                    lookup_links[(t_name, r_time)] = m_id
        except Exception as e:
            print(f"Aviso: Falha ao ler {arquivo_links}: {e}")

    # 3. Ler arquivo CSV do Forecast
    df = pd.read_csv(arquivo_hoje, names=["Course", "RaceTime", "ForecastStr"])
    
    tips_geradas = []
    
    for _, row in df.iterrows():
        course = row["Course"].strip()
        race_time_str = row["RaceTime"].strip()
        forecast_str = row["ForecastStr"]
        
        try:
            date_obj = datetime.strptime(race_time_str, "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
            
        caes = _parse_forecast_string(forecast_str)
        num_runners = len(caes)
        
        for cao in caes:
            c_name = cao["forecast_name_clean"]
            # lookup_key para cards
            lookup_key_cards = (course.lower(), race_time_str, c_name)
            card_info = lookup_cards.get(lookup_key_cards, {})
            
            # lookup_key para links (MarketId)
            norm_course = normalize_track_name(course)
            lookup_key_links = (norm_course, race_time_str)
            market_id = lookup_links.get(lookup_key_links, "")
            
            item_candidato = {
                "course": course,
                "date_obj": date_obj,
                "market_id": market_id,
                "num_runners": num_runners,
                "forecast_rank": cao["forecast_rank"],
                "forecast_odds": cao["forecast_odds"],
                "forecast_name_raw": cao["forecast_name_raw"],
                "forecast_name_clean": c_name,
                "trap": card_info.get("trap"),
                "category": card_info.get("category")
            }
            
            # 4. Avaliar contra todas as estratégias
            for est in estrategias:
                if aplicar_filtros(item_candidato, est.get("filtros", {})):
                    tip = processar_estrategia(est, item_candidato)
                    if tip:
                        tips_geradas.append(tip)
                        
    # 5. Salvar CSV para o BFB
    if tips_geradas:
        output_file = os.path.join(dir_output, f"bfb_tips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        fixed_output_file = os.path.join(dir_output_fixed, "bfb_tips.csv")
        
        df_tips = pd.DataFrame(tips_geradas)
        # O BFB espera colunas oficiais. StartTime com tempo universal, EventName, SelectionName, etc.
        # Provider é opcional (Tipster), mas vamos colocar
        cols_order = ["StartTime", "MarketId", "MarketType", "EventName", "SelectionName", "BetType", "MinPrice", "MaxPrice", "Provider"]
        
        df_tips[cols_order].to_csv(output_file, index=False)
        print(f"{len(tips_geradas)} dicas geradas e salvas em {output_file}")
        
        df_tips[cols_order].to_csv(fixed_output_file, index=False)
        print(f"{len(tips_geradas)} dicas salvas com nome fixo em {fixed_output_file}")
        
        # 6. Salvar log de auditoria
        log_file = os.path.join(dir_logs, f"auditoria_{datetime.now().strftime('%Y%m%d')}.csv")
        # Se já existe append, senão w
        header = not os.path.exists(log_file)
        df_tips["Timestamp_Geracao"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        df_tips.to_csv(log_file, mode='a', header=header, index=False)
    else:
        print("Nenhuma dica gerada com as estratégias atuais.")

if __name__ == "__main__":
    main()
