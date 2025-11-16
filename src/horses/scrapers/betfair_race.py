from __future__ import annotations

import time
from typing import Dict, List, Optional

from loguru import logger
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.horses.config import settings
from src.horses.utils.selenium_driver import build_chrome_driver
from src.horses.utils.text import clean_horse_name

_cookies_accepted: bool = False
_cookies_attempted: bool = False


def _accept_cookies(driver: WebDriver) -> None:
    """Aceita cookies na primeira pagina da sessao."""
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
                    "//button[contains(., 'Accept') or contains(., 'Aceitar') or contains(., 'Agree') or contains(., 'Concordo')]",
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


def scrape_betfair_race(race_url: str, driver: WebDriver | None = None) -> Optional[Dict[str, object]]:
    """Coleta informacoes de Timeform 1-2-3 para uma corrida da Betfair."""
    logger.debug("Abrindo corrida Betfair: %s", race_url)

    own_driver = False
    driver_instance: WebDriver
    if driver is None:
        driver_instance = build_chrome_driver()
        own_driver = True
    else:
        driver_instance = driver

    try:
        driver_instance.get(race_url)
        _accept_cookies(driver_instance)

        wait = WebDriverWait(driver_instance, settings.SELENIUM_EXPLICIT_WAIT_SEC + 3)
        try:
            try:
                wait.until(EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Timeform 1-2-3')]")))
            except Exception:
                logger.debug("Titulo 'Timeform 1-2-3' nao encontrado imediatamente. Tentando localizar container...")
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".timeform-section-123, .runner-rating-list")))
        except TimeoutException:
            logger.warning("Bloco Timeform nao disponivel. Recarregando pagina e tentando novamente...")
            driver_instance.refresh()
            time.sleep(1.0)
            _accept_cookies(driver_instance)
            wait = WebDriverWait(driver_instance, settings.SELENIUM_EXPLICIT_WAIT_SEC + 3)
            try:
                try:
                    wait.until(EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Timeform 1-2-3')]")))
                except Exception:
                    logger.debug("Apos recarregar, tentando localizar container Timeform...")
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".timeform-section-123, .runner-rating-list")))
            except TimeoutException:
                logger.warning("Informacoes Timeform indisponiveis apos recarregar. Ignorando corrida: %s", race_url)
                return None

        runner_elems = driver_instance.find_elements(By.CSS_SELECTOR, ".runner-rating-list li.runner-rating-item p.runner-name")
        raw_names: List[str] = [element.text.strip() for element in runner_elems[:3]]
        normalized = [clean_horse_name(name) for name in raw_names if name]

        return {
            "source": "betfair",
            "TimeformPrev": "; ".join(normalized) if normalized else "",
            "TimeformPrev_list": normalized,
        }
    finally:
        if own_driver:
            driver_instance.quit()