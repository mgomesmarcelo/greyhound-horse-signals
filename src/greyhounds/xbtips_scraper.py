import os
import re
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from playwright.sync_api import sync_playwright, Page, BrowserContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_DATA_DIR = PROJECT_ROOT / ".playwright_user_data"

class XBTipsScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.base_url = "https://www.xbtips.com/uk/basic"

    def _parse_text_for_signals(self, text: str, date_str: str) -> pd.DataFrame:
        lines = text.split("\n")
        
        mode = None # "lay" or "back"
        records = []
        
        current_race_time_uk = None
        current_category = None
        current_track = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "2) Corridas boas para lay" in line:
                mode = "lay"
                continue
            elif "3) Corridas boas para back" in line:
                mode = "back"
                continue
            
            # Match race header: 11:27 (08:27 BR) - A5 Romford
            m_race = re.match(r"^(\d{2}:\d{2})\s+\(\d{2}:\d{2}\s+BR\)\s+-\s+([A-Z0-9]+)\s+(.+)$", line)
            if m_race and mode:
                current_race_time_uk = m_race.group(1)
                current_category = m_race.group(2)
                current_track = m_race.group(3)
                continue
                
            # Match selection: Lay: Trap 3 - Rough Ronan | Score: 0.47 | Gap: 0.13
            # ou Back: Trap 1 - Jogon Flo Joe | Score: 0.16 | Gap: 0.13
            if mode:
                m_sel = re.match(rf"^({mode.capitalize()}):?\s+Trap\s+(\d)\s+-\s+([^\|]+?)\s+\|", line, re.IGNORECASE)
                if m_sel and current_race_time_uk:
                    trap_num = m_sel.group(2)
                    dog_name = m_sel.group(3).strip()
                    
                    score = 0.0
                    gap = 0.0
                    m_score = re.search(r"Score:\s*([0-9\.]+)", line)
                    if m_score:
                        score = float(m_score.group(1))
                    m_gap = re.search(r"Gap:\s*([0-9\.]+)", line)
                    if m_gap:
                        gap = float(m_gap.group(1))
                        
                    records.append({
                        "date": date_str,
                        "time_uk": current_race_time_uk,
                        "category": current_category,
                        "track_name": current_track,
                        "recommendation": mode.lower(),
                        "trap_number": trap_num,
                        "greyhound_name": dog_name,
                        "score": score,
                        "gap": gap
                    })
        
        return pd.DataFrame(records)

    def scrape_date(self, date_str: str) -> pd.DataFrame:
        url = f"{self.base_url}?lang=pt&date={date_str}"
        
        df = pd.DataFrame()
        USER_DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        print(f"[{date_str}] Inicializando Playwright na url: {url}")
        
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                headless=self.headless,
                # Evita detecção basica de bots
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = context.new_page()
            
            # Timeout longo para a primeira rede
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            try:
                # Ajusta para aceitar os cookies se aparecer
                try:
                    cookie_btn = page.locator("button:has-text('Accept')").first
                    if cookie_btn.is_visible(timeout=2000):
                        cookie_btn.click()
                except Exception:
                    pass
                
                # Se não estamos logados, o form de email pode aparecer ou "Acessar plataforma"
                if page.locator("input[type='password']").count() > 0 or "login" in page.url:
                    print("=========================================")
                    print("ATENÇÃO: ReCAPTCHA ou Login detectado.")
                    print("Faça o login M-A-N-U-A-L-M-E-N-T-E na janela do navegador.")
                    print("Aguardando carregamento da plataforma (timeout 5 min)...")
                    print("=========================================")
                    page.wait_for_selector("text=1) Corridas", timeout=300000)
                
                # Se logado mas preso numa landing page
                acessar_btn = page.locator("text='Acessar plataforma'")
                if acessar_btn.count() > 0:
                    acessar_btn.first.click()
                    page.wait_for_selector("text=1) Corridas", timeout=30000)
                    
                # Esperar até as corridas carregarem
                page.wait_for_selector("text=2) Corridas boas para lay", timeout=15000)
                
                # Extrai o texto limpo direto do body e joga na Regex
                text_content = page.locator("body").inner_text()
                df = self._parse_text_for_signals(text_content, date_str)
                
            except Exception as e:
                print(f"[{date_str}] Falha ao processar pagina: {e}")
            finally:
                context.close()
            
        return df
