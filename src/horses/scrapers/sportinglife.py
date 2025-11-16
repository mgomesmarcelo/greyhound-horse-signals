from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
from loguru import logger
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.horses.config import settings as horses_settings
from src.horses.utils.selenium_driver import build_chrome_driver
from src.horses.utils.text import clean_horse_name, normalize_track_name


def _sleep_jitter(label: str = "") -> None:
    low = max(0.0, float(horses_settings.SPORTINGLIFE_MIN_DELAY_SEC))
    high = max(low, float(horses_settings.SPORTINGLIFE_MAX_DELAY_SEC))
    delay = random.uniform(low, high)
    logger.debug("Delay Sporting Life{}: {:.2f}s", f" ({label})" if label else "", delay)
    time.sleep(delay)


def _build_day_urls(date_str: str) -> List[str]:
    base = horses_settings.SPORTINGLIFE_BASE_URL.rstrip("/")
    return [
        f"{base}/{date_str}",
        f"{base}?date={date_str}",
    ]


def _accept_cookies(driver) -> None:
    selectors = [
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
        (By.CSS_SELECTOR, "button[data-testid='uc-accept-all-button']"),
        (By.XPATH, "//button[contains(., 'Accept') and contains(., 'Cookies')]"),
    ]
    for by, value in selectors:
        try:
            button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, value)))
            driver.execute_script("arguments[0].click();", button)
            _sleep_jitter("cookies")
            return
        except Exception:
            continue


def _looks_like_race_href(href: str | None, text: str | None) -> bool:
    if not isinstance(href, str):
        return False
    if "/racing/results" not in href:
        return False
    has_time_in_text = bool(re.search(r"\b\d{1,2}:\d{2}\b", str(text or "")))
    has_time_in_href = bool(re.search(r"\b\d{1,2}:\d{2}\b", href))
    return has_time_in_text or has_time_in_href


def _append_unique(seq: List[str], seen: set[str], value: str) -> None:
    if value not in seen:
        seq.append(value)
        seen.add(value)


def _collect_day_links(driver, date_str: str) -> List[str]:
    wait = WebDriverWait(driver, horses_settings.SELENIUM_EXPLICIT_WAIT_SEC)
    links: List[str] = []
    seen: set[str] = set()

    for url in _build_day_urls(date_str):
        try:
            driver.get(url)
        except WebDriverException as exc:
            logger.debug("Falha ao abrir {}: {}", url, exc)
            continue

        _sleep_jitter("day-page")
        _accept_cookies(driver)

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='meeting-summary']")))
        except TimeoutException:
            logger.debug("Sumários não carregaram para {}", url)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        candidates = soup.select("div[data-testid='meeting-summary'] div[data-test-id='race-container'] a[href]")

        for anchor in candidates:
            href = anchor.get("href") or ""
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.sportinglife.com" + href
            href_norm = href.split("#", 1)[0].split("?", 1)[0]
            if "/racing/results/" in href_norm:
                _append_unique(links, seen, href_norm)

        if not links:
            for anchor in soup.find_all("a"):
                href = anchor.get("href")
                txt = anchor.get_text(strip=True)
                if not href or not _looks_like_race_href(href, txt):
                    continue
                if href.startswith("/"):
                    href = "https://www.sportinglife.com" + href
                href_norm = href.split("#", 1)[0].split("?", 1)[0]
                _append_unique(links, seen, href_norm)

        if links:
            break

    logger.info("{} links de corridas Sporting Life encontrados para {}", len(links), date_str)
    return links


def _load_race_soup(driver, url: str) -> BeautifulSoup:
    try:
        driver.get(url)
    except WebDriverException as exc:
        raise RuntimeError(f"Falha ao carregar {url}: {exc}") from exc

    _sleep_jitter("race-page")
    _accept_cookies(driver)

    try:
        WebDriverWait(driver, horses_settings.SELENIUM_EXPLICIT_WAIT_SEC).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "main"))
        )
    except TimeoutException:
        logger.debug("Main não detectado após timeout em {}", url)

    html = driver.page_source
    return BeautifulSoup(html, "html.parser")


_HORSE_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z '\-\.&()]+[A-Za-z\)]$")
_DISCARD = {"html", "body", "div", "sporting", "life", "results", "cookies", "privacy", "policy", "terms"}


def _valid_name(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    base = re.sub(r"\s+", " ", s)
    if base.lower() in _DISCARD:
        return False
    if len(base) < 2 or len(base) > 60:
        return False
    if not _HORSE_NAME_OK.match(base):
        return False
    if not re.search(r"[aeiouAEIOU]", base):
        return False
    return True


def _sl_forecast_to_decimal(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    items = [p.strip() for p in text.split(",") if p.strip()]
    out: List[str] = []
    re_paren_frac = re.compile(r"^(.*)\((\d+)\s*/\s*(\d+)\)\s*$")
    re_lead_frac = re.compile(r"^(\d+)\s*/\s*(\d+)\s+(.+)$")
    re_evens = re.compile(r"^(.*)\b(evs|evens)\b(.*)$", re.IGNORECASE)
    for item in items:
        s = item.strip()
        m1 = re_paren_frac.match(s)
        if m1:
            name_left = m1.group(1).strip()
            num = int(m1.group(2))
            den = int(m1.group(3)) or 1
            dec = (num / den) + 1.0
            label = clean_horse_name(name_left)
            out.append(f"{dec:.2f} {label}" if label else f"{dec:.2f}")
            continue
        m2 = re_lead_frac.match(s)
        if m2:
            num = int(m2.group(1))
            den = int(m2.group(2)) or 1
            name = m2.group(3).strip()
            dec = (num / den) + 1.0
            label = clean_horse_name(name)
            out.append(f"{dec:.2f} {label}" if label else f"{dec:.2f}")
            continue
        e = re_evens.match(s)
        if e:
            name = (e.group(1) + " " + e.group(3)).strip()
            label = clean_horse_name(name)
            out.append(f"2.00 {label}" if label else "2.00")
            continue
        s2 = re.sub(r"\s*\([^\)]*\)\s*", "", s).strip()
        out.append(s2)
    return ", ".join(out)


def _parse_forecast_top3_from_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    s = re.sub(r"(?i)(betting\s+)?forecast\s*:\s*", "", text.strip())
    parts = [p.strip() for p in s.split(",") if p and isinstance(p, str)]
    names: List[str] = []
    for part in parts:
        m1 = re.match(r"^\s*\d+(?:[\./]\d+|(?:\.\d+)?)\s+(.+)$", part)
        if m1:
            candidate = m1.group(1).strip()
        else:
            candidate = re.sub(r"\s*\([^\)]*\)\s*$", "", part).strip()
        candidate = re.sub(r"^\s*\d+\.\s*", "", candidate)
        cleaned = clean_horse_name(candidate)
        if cleaned and _valid_name(cleaned) and cleaned not in names:
            names.append(cleaned)
        if len(names) >= 3:
            break
    return names


def _extract_track_and_time(soup: BeautifulSoup, fallback_date: str, url: str) -> Tuple[str, str]:
    title = soup.title.get_text(strip=True) if soup.title else ""
    mtime = re.search(r"\b(\d{1,2}:\d{2})\b", title)
    track = ""
    mtrack = re.search(r"([A-Za-z][A-Za-z\s]+?)\s*-\s*\d{1,2}:\d{2}", title)
    if mtrack:
        track = normalize_track_name(mtrack.group(1))
    if not track:
        for heading in soup.find_all(["h1", "h2", "h3"]):
            txt = heading.get_text(" ", strip=True)
            if not txt:
                continue
            m = re.search(r"^([A-Za-z][A-Za-z\s]+)\s+\d{1,2}:\d{2}\b", txt)
            if m:
                track = normalize_track_name(m.group(1))
                break
    if not track:
        murl = re.search(r"/racing/results/\d{4}-\d{2}-\d{2}/([^/]+)/", url)
        if murl:
            slug = murl.group(1).replace("-", " ")
            track = normalize_track_name(slug)
    hhmm = mtime.group(1) if mtime else "00:00"
    race_iso = f"{fallback_date}T{hhmm}"
    return track, race_iso


def _extract_betting_forecast(soup: BeautifulSoup) -> str:
    node = soup.select_one("section#forecasts [data-test-id='forecasts-body'] p")
    if node:
        return node.get_text(" ", strip=True)
    for tag in soup.find_all(True):
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        m = re.search(r"(?i)(betting\s+forecast|forecast)\s*:\s*(.+)$", txt)
        if m:
            return m.group(0)
        if re.search(r"(?i)(betting\s+forecast|forecast)\s*:\s*$", txt):
            sibling = tag.find_next(string=True)
            if sibling and isinstance(sibling, str):
                return f"Forecast: {sibling.strip()}"
    return ""


def _extract_timeform_top3(soup: BeautifulSoup) -> List[str]:
    items = soup.select("section#verdict [data-test-id='verdict-body'] ol li")
    names: List[str] = []
    for li in items:
        bold = li.find("b")
        txt = bold.get_text(" ", strip=True) if bold else li.get_text(" ", strip=True)
        txt = re.sub(r"\s*\([^\)]*\)\s*$", "", txt)
        cleaned = clean_horse_name(txt)
        if cleaned and _valid_name(cleaned) and cleaned not in names:
            names.append(cleaned)
        if len(names) >= 3:
            break
    if len(names) >= 3:
        return names[:3]

    texts: List[str] = []
    for tag in soup.find_all(True):
        txt = tag.get_text("\n", strip=True)
        if not txt:
            continue
        if re.search(r"(?i)timeform|verdict", txt) and re.search(r"\b1\.|\b2\.|\b3\.", txt):
            texts.append(txt)

    for text in texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        out: List[str] = []
        for line in lines:
            match = re.match(r"^\s*([123])[\).\-\:]?\s+(.+)$", line)
            if not match:
                continue
            candidate = re.sub(r"\s*\([^\)]*\)\s*$", "", match.group(2)).strip()
            cleaned = clean_horse_name(candidate)
            if cleaned and _valid_name(cleaned) and cleaned not in out:
                out.append(cleaned)
            if len(out) >= 3:
                break
        if len(out) >= 3:
            return out[:3]
    return []


def _scrape_race_with_driver(driver, url: str, date_str: str) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    soup = _load_race_soup(driver, url)
    track, race_iso = _extract_track_and_time(soup, date_str, url)

    forecast_text = _extract_betting_forecast(soup)
    forecast_text = _sl_forecast_to_decimal(forecast_text)
    top3 = _extract_timeform_top3(soup)
    forecast_top3 = _parse_forecast_top3_from_text(forecast_text)

    if len(top3) < 3 and forecast_top3:
        top3 = forecast_top3[:3]

    top3 = [cleaned for cleaned in top3 if cleaned and _valid_name(cleaned)]
    while len(top3) < 3:
        top3.append("")

    row_top3: Optional[Dict[str, str]] = None
    if any(name for name in top3):
        row_top3 = {
            "track_name": track,
            "race_time_iso": race_iso,
            "TimeformTop1": top3[0] if len(top3) > 0 else "",
            "TimeformTop2": top3[1] if len(top3) > 1 else "",
            "TimeformTop3": top3[2] if len(top3) > 2 else "",
        }

    row_fc: Optional[Dict[str, str]] = None
    if forecast_text:
        row_fc = {
            "track_name": track,
            "race_time_iso": race_iso,
            "SportingLifeForecast": forecast_text,
        }

    logger.debug(
        "Corrida Sporting Life {} {} - top3 extraídos: {} - forecast: {}",
        track or "?",
        race_iso,
        ", ".join([name for name in top3 if name]) or "nenhum",
        "ok" if forecast_text else "vazio",
    )

    return row_top3, row_fc


def _scrape_races_concurrent(links: List[str], date_str: str, max_workers: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    collected_top3: List[Tuple[int, Dict[str, str]]] = []
    collected_fc: List[Tuple[int, Dict[str, str]]] = []

    def _job(payload: Tuple[int, str]) -> Tuple[int, Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        idx, race_url = payload
        driver = build_chrome_driver()
        try:
            top3_row, fc_row = _scrape_race_with_driver(driver, race_url, date_str)
            return idx, top3_row, fc_row
        finally:
            driver.quit()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_job, (idx, url)): (idx, url) for idx, url in enumerate(links)}
        for future in as_completed(futures):
            idx, url = futures[future]
            try:
                result_idx, top3_row, fc_row = future.result()
            except Exception as exc:
                logger.error("Erro ao raspar corrida {} ({}): {}", idx + 1, url, exc)
                continue
            if top3_row:
                collected_top3.append((result_idx, top3_row))
            if fc_row:
                collected_fc.append((result_idx, fc_row))

    collected_top3.sort(key=lambda item: item[0])
    collected_fc.sort(key=lambda item: item[0])
    return [item[1] for item in collected_top3], [item[1] for item in collected_fc]


def scrape_day(date_str: str, max_workers: int = 1) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Retorna:
      rows_top3: List[Dict] com colunas: track_name, race_time_iso, TimeformTop1, TimeformTop2, TimeformTop3
      rows_fc:   List[Dict] com colunas: track_name, race_time_iso, SportingLifeForecast
      links:     List[str] com URLs visitadas/descobertas
    """
    logger.info("Iniciando Sporting Life para {}", date_str)

    rows_top3: List[Dict] = []
    rows_fc: List[Dict] = []
    links: List[str] = []

    driver = build_chrome_driver()
    try:
        links = _collect_day_links(driver, date_str)
        if not links:
            logger.info("Sporting Life sem dados para {}", date_str)
            return rows_top3, rows_fc, links

        workers = max(1, int(max_workers or 1))
        if workers == 1:
            for idx, url in enumerate(links):
                try:
                    top3_row, fc_row = _scrape_race_with_driver(driver, url, date_str)
                except Exception as exc:
                    logger.error("Erro ao raspar corrida {} ({}): {}", idx + 1, url, exc)
                    continue
                if top3_row:
                    rows_top3.append(top3_row)
                if fc_row:
                    rows_fc.append(fc_row)
                _sleep_jitter("entre-corridas")
            _sleep_jitter("fim-dia")
        else:
            driver.quit()
            driver = None
            rows_top3, rows_fc = _scrape_races_concurrent(links, date_str, workers)
    finally:
        if driver is not None:
            driver.quit()

    logger.info(
        "Concluído Sporting Life para {}. Corridas com top3: {}, forecasts: {}",
        date_str,
        len(rows_top3),
        len(rows_fc),
    )
    return rows_top3, rows_fc, links

