from __future__ import annotations

import re
import time
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from src.greyhounds.config import settings
from src.greyhounds.utils.selenium_driver import build_chrome_driver

_cookies_accepted: bool = False
_cookies_attempted: bool = False

def _accept_cookies(driver) -> None:
    global _cookies_accepted, _cookies_attempted
    if _cookies_accepted:
        return

    timeout_sec = 6 if not _cookies_attempted else 1.5
    clicked = False
    try:
        wait = WebDriverWait(driver, timeout_sec)
        try:
            btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            try:
                btn.click()
            except Exception:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                except Exception:
                    pass
            else:
                clicked = True
        except Exception:
            try:
                candidates = driver.find_elements(
                    By.XPATH,
                    "//button[contains(., 'Accept') or contains(., 'Aceitar') or contains(., 'Allow all cookies')]"
                )
                for element in candidates:
                    if element.is_displayed() and element.is_enabled():
                        try:
                            element.click()
                            clicked = True
                            break
                        except Exception:
                            try:
                                driver.execute_script("arguments[0].click();", element)
                                clicked = True
                                break
                            except Exception:
                                continue
            except Exception:
                pass
    finally:
        _cookies_attempted = True
        if clicked:
            _cookies_accepted = True


def scrape_betfair_race(race_url: str, track_name: str, race_time_iso: str, driver=None) -> Dict[str, object]:
    """
    Coleta informacoes da corrida (Runners, Traps, Category, Timeform)
    diretamente da pagina da corrida na Betfair.
    """
    own_driver = False
    if driver is None:
        driver = build_chrome_driver()
        own_driver = True

    try:
        for attempt in range(2):
            if attempt == 0:
                logger.debug(f"Abrindo corrida Betfair para {track_name} as {race_time_iso}: {race_url}")
            else:
                logger.debug(f"Atualizando a pagina (Tentativa 2) para {track_name} as {race_time_iso}...")
                
            driver.get(race_url)
            _accept_cookies(driver)

            # Wait for the main elements to load (runners and timeform block)
            try:
                wait = WebDriverWait(driver, settings.SELENIUM_EXPLICIT_WAIT_SEC)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tr.runner-line, .runner-name")))
            except TimeoutException:
                logger.warning(f"Timeout ao carregar corredores em {race_url}. Pagina pode estar vazia.")
                pass # Tenta extrair mesmo assim

            # Wait specifically for Betting Forecast to appear (it sometimes takes a few extra seconds)
            for _ in range(5):
                html = driver.page_source
                if "Betting Forecast" in html:
                    break
                time.sleep(1)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # 1. CATEGORIA (A1, D2, OR, etc)
            category = ""
            title_text = soup.title.get_text(strip=True) if soup.title else ""
            cat_match = re.search(r'\b(A\d|D\d|S\d|B\d|HP|OR\d*|IT\d*|IV|HC)\b', title_text)
            if cat_match:
                category = cat_match.group(1)

            # 2. RUNNERS e TRAPS
            runners_data = []
            # Usando explicitamente tr.runner-line para ignorar containers mobile (como divs ou lis)
            runner_nodes = soup.select("tr.runner-line")
            for node in runner_nodes:
                name_el = node.select_one(".runner-name, .name")
                if not name_el:
                    continue
                name = name_el.get_text(strip=True)
                
                trap = ""
                cloth_el = node.select_one(".runner-cloth-number, .cloth-number, .trap-number, .greyhound-silk")
                if cloth_el:
                    trap_text = cloth_el.get_text(strip=True)
                    if trap_text:
                        # Extrai apenas os digitos do trap_text (caso venha com letras)
                        trap_digits = re.search(r'\d+', trap_text)
                        if trap_digits:
                            trap = trap_digits.group(0)
                    else:
                        # Na Betfair, a trap frequentemente vem apenas como classe (ex: cloth-number-2 ou greyhound-silk-2)
                        classes = cloth_el.get('class', [])
                        for c in classes:
                            match = re.match(r'(?:cloth-number-|trap-|greyhound-silk-)(\d+)', c)
                            if match:
                                trap = match.group(1)
                                break
                else:
                    text = node.get_text(" ", strip=True)
                    match = re.match(r'^(\d+)\s+', text)
                    if match:
                        trap = match.group(1)

                if name:
                    trap_val = None
                    if trap.isdigit():
                        trap_val = int(trap)
                    elif trap:
                        trap_val = trap

                    existing = next((r for r in runners_data if r['name'] == name), None)
                    if existing:
                        # Se ja existe mas estava sem trap e agora achou, atualiza
                        if existing['trap'] is None and trap_val is not None:
                            existing['trap'] = trap_val
                    else:
                        runners_data.append({
                            "trap": trap_val,
                            "name": name
                        })

            # 3. TIMEFORM TOP 3
            top3 = []
            tf_section = soup.select_one(".timeform-section-123, .runner-rating-list, .timeform-verdict")
            if tf_section:
                names = tf_section.select(".runner-name")
                for n in names[:3]:
                    top3.append(n.get_text(strip=True))

            # 4. TIMEFORM FORECAST
            forecast_text = ""
            forecast_node = soup.find(string=lambda text: text and "Betting Forecast" in text)
            if forecast_node and forecast_node.parent:
                sibling = forecast_node.parent.find_next_sibling()
                if sibling:
                    forecast_text = sibling.get_text(" ", strip=True)
                    
            forecast_full = f"TimeformForecast : {forecast_text}" if forecast_text else ""

            # Condicao de sucesso: temos o forecast, runners e pelo menos 1 do top 3.
            # Se conseguimos, paramos de tentar. Se nao, fazemos a 2a tentativa.
            if forecast_full and runners_data and top3:
                break
                
            if attempt == 0:
                logger.debug(f"Dados incompletos em {track_name}. Realizando tentativa 2 (refresh)...")

        return {
            "track_name": track_name,
            "race_time_iso": race_time_iso,
            "TimeformForecast": forecast_full,
            "TimeformTop1": top3[0] if len(top3) > 0 else "",
            "TimeformTop2": top3[1] if len(top3) > 1 else "",
            "TimeformTop3": top3[2] if len(top3) > 2 else "",
            "RaceCategory": category,
            "Runners": runners_data,
        }

    finally:
        if own_driver:
            driver.quit()
