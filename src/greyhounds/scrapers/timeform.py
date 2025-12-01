from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from urllib.parse import urljoin

from src.greyhounds.config import settings
from src.greyhounds.utils.files import write_dataframe_snapshots
from src.greyhounds.utils.dates import iso_to_hhmm
from src.greyhounds.utils.selenium_driver import build_chrome_driver
from src.greyhounds.utils.text import clean_horse_name, normalize_track_name


_TIMEFORM_HOME = settings.TIMEFORM_BASE_URL
_TIMEFORM_BASE = "https://www.timeform.com/greyhound-racing"


def _sleep_jitter(label: str = "") -> None:
    low = max(0.0, settings.TIMEFORM_MIN_DELAY_SEC)
    high = max(low, settings.TIMEFORM_MAX_DELAY_SEC)
    delay = random.uniform(low, high)
    logger.debug("Delay{}: {:.2f}s", f" {label}" if label else "", delay)
    time.sleep(delay)


def _accept_cookies(driver) -> None:
    try:
        wait = WebDriverWait(driver, settings.SELENIUM_EXPLICIT_WAIT_SEC)
        banner = None
        try:
            banner = wait.until(EC.presence_of_element_located((By.ID, "onetrust-banner-sdk")))
        except Exception:
            banner = None

        if banner and banner.is_displayed():
            try:
                btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
                btn.click()
            except Exception:
                try:
                    driver.execute_script("document.getElementById('onetrust-accept-btn-handler')?.click();")
                except Exception:
                    try:
                        btn2 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@id='onetrust-accept-btn-handler' or contains(., 'Allow All Cookies')]") ))
                        driver.execute_script("arguments[0].click();", btn2)
                    except Exception:
                        logger.debug("Falha ao clicar no botao de cookies do Timeform.")

            try:
                WebDriverWait(driver, 5).until(EC.invisibility_of_element_located((By.ID, "onetrust-banner-sdk")))
            except Exception:
                logger.debug("Banner de cookies ainda visivel apos clique.")
            _sleep_jitter("cookies")
        else:
            logger.debug("Banner de cookies (Timeform) nao presente.")
    except Exception:
        logger.debug("Botao/banner de cookies (Timeform) nao encontrado ou ja aceito.")


def _list_cards(driver) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []

    try:
        container_list = driver.find_elements(By.CSS_SELECTOR, ".wfr-bytrack-content")
        for container in container_list:
            meetings = container.find_elements(By.CSS_SELECTOR, ".wfr-meeting")
            for section in meetings:
                try:
                    track_name = section.find_element(By.CSS_SELECTOR, "b.wfr-track").text.strip()
                    links = section.find_elements(By.CSS_SELECTOR, "ul li a.wfr-race")
                    for anchor in links:
                        hhmm = anchor.text.strip()
                        link = anchor.get_attribute("href") or anchor.get_attribute("ng-href")
                        if link and not link.startswith("http"):
                            link = urljoin(_TIMEFORM_BASE, link)
                        cards.append({
                            "track_name": track_name,
                            "track_key": normalize_track_name(track_name),
                            "hhmm": hhmm,
                            "url": link,
                        })
                except Exception:
                    continue
    except Exception:
        pass

    if not cards:
        sections = driver.find_elements(By.CSS_SELECTOR, ".w-cards-results section")
        for section in sections:
            try:
                track_name = section.find_element(By.TAG_NAME, "h3").text.strip()
                links = section.find_elements(By.CSS_SELECTOR, "li a")
                for anchor in links:
                    hhmm = anchor.text.strip()
                    link = anchor.get_attribute("href") or anchor.get_attribute("ng-href")
                    if link and not link.startswith("http"):
                        link = urljoin(_TIMEFORM_BASE, link)
                    cards.append({
                        "track_name": track_name,
                        "track_key": normalize_track_name(track_name),
                        "hhmm": hhmm,
                        "url": link,
                    })
            except Exception:
                continue

    return cards


def _extract_forecast(driver) -> str:
    try:
        paragraph = driver.find_element(By.XPATH, "//p[b[contains(., 'Betting Forecast')]]")
        text = paragraph.text.strip()
        if text.lower().startswith("betting forecast"):
            text = text.split(":", 1)[-1].strip()
            return f"TimeformForecast : {_convert_forecast_to_decimal(text)}"
        return ""
    except Exception:
        return ""


def _extract_top3(driver) -> List[str]:
    try:
        container = driver.find_element(By.CSS_SELECTOR, ".rpf-verdict-container")
        selections = container.find_elements(By.CSS_SELECTOR, ".rpf-verdict-selection")
        top_names: List[str] = []
        for selection in selections[:3]:
            try:
                name_el = selection.find_element(By.CSS_SELECTOR, ".rpf-verdict-selection-name a")
                name = name_el.text.strip()
                if name:
                    top_names.append(clean_horse_name(name))
            except Exception:
                continue
        return top_names
    except Exception:
        return []


def _convert_forecast_to_decimal(raw: str) -> str:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    converted: List[str] = []
    frac_re = re.compile(r"^(\d+)\s*/\s*(\d+)(?:\b|\s)(.*)$")
    evens_re = re.compile(r"^(?:evs|evens)\b\s*(.*)$", re.IGNORECASE)
    for item in items:
        match = frac_re.match(item)
        if match:
            num = int(match.group(1))
            den = int(match.group(2)) if int(match.group(2)) != 0 else 1
            name = match.group(3).strip()
            value = (num / den) + 1.0
            converted.append(f"{value:.2f} {name}" if name else f"{value:.2f}")
            continue
        evens = evens_re.match(item)
        if evens:
            name = evens.group(1).strip()
            converted.append(f"2.00 {name}" if name else "2.00")
            continue
        converted.append(item)
    return ", ".join(converted)


def scrape_timeform_for_races(race_rows: Iterable[Dict[str, str]]) -> Iterable[Dict[str, object]]:
    logger.info("Iniciando raspagem Timeform para corridas filtradas pelo Betfair race_links.csv")
    driver = build_chrome_driver()
    try:
        driver.get(_TIMEFORM_HOME)
        _accept_cookies(driver)
        _sleep_jitter("home")

        cards = _list_cards(driver)
        logger.debug("Total de cards Timeform capturados: {}", len(cards))

        index: Dict[tuple, str] = {}
        for card in cards:
            track_key = card.get("track_key") or normalize_track_name(card.get("track_name", ""))
            hhmm = card.get("hhmm", "")
            url = card.get("url", "")
            if track_key and hhmm and url:
                index[(track_key, hhmm)] = url

        count = 0

        for row in race_rows:
            track = row.get("track_name", "")
            race_time_iso = row.get("race_time_iso", "")
            match_key = (normalize_track_name(track), iso_to_hhmm(race_time_iso))
            url = index.get(match_key)
            if not url:
                continue

            driver.get(url)
            _sleep_jitter("race")
            forecast = _extract_forecast(driver)
            if forecast:
                out_row: Dict[str, object] = {
                    "track_name": track,
                    "race_time_iso": race_time_iso,
                    "TimeformForecast": forecast,
                }
                top3 = _extract_top3(driver)
                if len(top3) > 0:
                    out_row["TimeformTop1"] = top3[0]
                if len(top3) > 1:
                    out_row["TimeformTop2"] = top3[1]
                if len(top3) > 2:
                    out_row["TimeformTop3"] = top3[2]
                logger.debug(f"TimeformForecast coletado: {track} {race_time_iso}")
                count += 1
                yield out_row
            _sleep_jitter("post-race")

        logger.info(f"Raspagem Timeform concluida. Corridas com forecast: {count}")
        return
    finally:
        driver.quit()


def _build_timeform_forecast_df(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    data = [
        {
            "track_name": row.get("track_name"),
            "race_time_iso": row.get("race_time_iso"),
            "TimeformForecast": row.get("TimeformForecast", ""),
        }
        for row in rows
        if row.get("TimeformForecast")
    ]
    if not data:
        return pd.DataFrame([], columns=["track_name", "race_time_iso", "TimeformForecast"])
    return pd.DataFrame(data)


def _build_timeform_top3_df(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    data = [
        {
            "track_name": row.get("track_name"),
            "race_time_iso": row.get("race_time_iso"),
            "TimeformTop1": row.get("TimeformTop1"),
            "TimeformTop2": row.get("TimeformTop2"),
            "TimeformTop3": row.get("TimeformTop3"),
        }
        for row in rows
        if row.get("TimeformTop1") or row.get("TimeformTop2") or row.get("TimeformTop3")
    ]
    if not data:
        return pd.DataFrame([], columns=["track_name", "race_time_iso", "TimeformTop1", "TimeformTop2", "TimeformTop3"])
    return pd.DataFrame(data)


def save_timeform_forecast(
    rows: Iterable[Dict[str, object]],
    raw_path: Path,
    parquet_path: Path,
) -> pd.DataFrame:
    df = _build_timeform_forecast_df(rows)
    write_dataframe_snapshots(df, raw_path=raw_path, parquet_path=parquet_path)
    return df


def save_timeform_top3(
    rows: Iterable[Dict[str, object]],
    raw_path: Path,
    parquet_path: Path,
) -> pd.DataFrame:
    df = _build_timeform_top3_df(rows)
    write_dataframe_snapshots(df, raw_path=raw_path, parquet_path=parquet_path)
    return df