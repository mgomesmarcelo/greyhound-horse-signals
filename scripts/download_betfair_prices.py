from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urljoin
import re

import requests
from bs4 import BeautifulSoup  # type: ignore[import]
from loguru import logger


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.core.config import settings


BETFAIR_INDEX_URLS = (
    "https://promo.betfair.com/betfairsp/prices",
    "https://promo.betfair.com/betfairsp/prices/",
)

BETFAIR_FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-z0-9_]+?)(?P<day>\d{2})(?P<month>\d{2})(?P<year>\d{4})\.csv$",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(
    r"(?P<year>20\d{2})[-_](?P<month>\d{2})[-_](?P<day>\d{2})"
)


@dataclass
class TargetConfig:
    name: str
    data_root: Path
    result_dir: Path
    start_date: date
    prefixes: Sequence[str]
    existing_files: set[str] = field(default_factory=set)

    def needs_file(self, filename: str, file_date: date, prefix: str) -> bool:
        if file_date < self.start_date:
            return False
        if prefix not in self.prefixes:
            return False
        return filename not in self.existing_files


def parse_betfair_filename(filename: str) -> tuple[Optional[str], Optional[date]]:
    match = BETFAIR_FILENAME_RE.match(filename)
    if not match:
        return None, None
    try:
        day = int(match.group("day"))
        month = int(match.group("month"))
        year = int(match.group("year"))
        return match.group("prefix"), date(year, month, day)
    except ValueError:
        return None, None


def extract_any_date(filename: str) -> Optional[date]:
    _, betfair_date = parse_betfair_filename(filename)
    if betfair_date:
        return betfair_date
    iso_match = ISO_DATE_RE.search(filename)
    if iso_match:
        try:
            year = int(iso_match.group("year"))
            month = int(iso_match.group("month"))
            day = int(iso_match.group("day"))
            return date(year, month, day)
        except ValueError:
            return None
    return None


def discover_earliest_date(data_root: Path) -> Optional[date]:
    if not data_root.exists():
        return None
    min_date: Optional[date] = None
    for csv_path in data_root.rglob("*.csv"):
        file_date = extract_any_date(csv_path.name)
        if not file_date:
            continue
        if min_date is None or file_date < min_date:
            min_date = file_date
    return min_date


def detect_prefixes(result_dir: Path) -> List[str]:
    prefixes: Dict[str, None] = {}
    if not result_dir.exists():
        return []
    for csv_path in result_dir.glob("*.csv"):
        prefix, fdate = parse_betfair_filename(csv_path.name)
        if prefix and fdate:
            prefixes[prefix] = None
    return list(prefixes.keys())


def ensure_target_config(
    name: str,
    subdir: str,
    start_date_override: Optional[str],
) -> Optional[TargetConfig]:
    data_root = settings.DATA_DIR / subdir
    result_dir = data_root / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)

    if start_date_override:
        start_date = datetime.strptime(start_date_override, "%Y-%m-%d").date()
    else:
        start_date = discover_earliest_date(data_root) or date.today()

    prefixes = detect_prefixes(result_dir)
    if not prefixes:
        logger.warning(
            f"Nenhum prefixo Betfair identificado para {name} em {result_dir}. Nada será baixado."
        )
        return None

    existing = {p.name for p in result_dir.glob("*.csv")}
    start_date_iso = start_date.isoformat()
    logger.info(
        f"[{name}] Prefixos={', '.join(sorted(prefixes))} | "
        f"Arquivos locais={len(existing)} | Data mínima={start_date_iso}",
    )
    return TargetConfig(
        name=name,
        data_root=data_root,
        result_dir=result_dir,
        start_date=start_date,
        prefixes=prefixes,
        existing_files=existing,
    )


def fetch_index(session: requests.Session) -> tuple[str, str]:
    last_exc: Optional[Exception] = None
    for url in BETFAIR_INDEX_URLS:
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            base_url = resp.url
            if not base_url.endswith("/"):
                base_url = base_url + "/"
            return resp.text, base_url
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha ao carregar índice Betfair SP")


def iter_remote_rows(
    html: str,
    base_url: str,
) -> Iterator[tuple[str, str, Optional[str], Optional[date]]]:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href.lower().endswith(".csv"):
            continue
        filename = Path(href).name
        prefix, fdate = parse_betfair_filename(filename)
        yield filename, urljoin(base_url, href), prefix, fdate


def build_download_plan(
    targets: Sequence[TargetConfig],
    remote_rows: Iterable[tuple[str, str, Optional[str], Optional[date]]],
) -> Dict[str, List[tuple[date, str, str]]]:
    plan: Dict[str, List[tuple[date, str, str]]] = {t.name: [] for t in targets}
    min_required_date = min((t.start_date for t in targets), default=date.today())

    for filename, url, prefix, file_date in remote_rows:
        if not prefix or not file_date:
            continue
        if file_date < min_required_date:
            break
        for target in targets:
            if target.needs_file(filename, file_date, prefix):
                plan[target.name].append((file_date, filename, url))
    for bucket in plan.values():
        bucket.sort(key=lambda item: item[0])
    return plan


def download_files(
    session: requests.Session,
    target: TargetConfig,
    downloads: List[tuple[date, str, str]],
    delay_seconds: float,
    dry_run: bool,
    max_downloads: Optional[int],
) -> int:
    if not downloads:
        logger.success(f"[{target.name}] Nenhum arquivo faltante.")
        return 0
    total = len(downloads)
    logger.info(f"[{target.name}] {total} arquivos faltantes.")
    completed = 0
    for idx, (fdate, filename, url) in enumerate(downloads, start=1):
        dest = target.result_dir / filename
        logger.info(
            f"[{target.name}] ({idx}/{total}) Baixando {filename} "
            f"({fdate.isoformat()}) -> {dest}"
        )
        if not dry_run:
            with session.get(url, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            time.sleep(max(delay_seconds, 0.0))
        completed += 1
        if max_downloads is not None and completed >= max_downloads:
            logger.warning(
                f"[{target.name}] Limite de downloads ({max_downloads}) atingido."
            )
            break
    return completed


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa automaticamente arquivos Betfair SP faltantes "
            "para horses e greyhounds."
        )
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Espera (segundos) entre downloads. Padrão: 1.5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os arquivos necessários, sem baixar.",
    )
    parser.add_argument(
        "--horses-start-date",
        help="Sobrescreve a data mínima detectada para cavalos (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--greyhounds-start-date",
        help="Sobrescreve a data mínima detectada para galgos (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        help="Limita a quantidade total de downloads (útil para testes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    horses = ensure_target_config(
        "horses", "horses", args.horses_start_date
    )
    greyhounds = ensure_target_config(
        "greyhounds", "greyhounds", args.greyhounds_start_date
    )
    targets = [t for t in (horses, greyhounds) if t]
    if not targets:
        logger.warning("Nenhum alvo configurado. Encerrando.")
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            )
        }
    )

    logger.info("Carregando índice público Betfair SP...")
    html, base_url = fetch_index(session)

    plan = build_download_plan(
        targets=targets,
        remote_rows=iter_remote_rows(html, base_url),
    )

    total_downloaded = 0
    for target in targets:
        downloads = plan.get(target.name, [])
        total_downloaded += download_files(
            session=session,
            target=target,
            downloads=downloads,
            delay_seconds=args.delay,
            dry_run=args.dry_run,
            max_downloads=args.max_downloads,
        )

    dry_suffix = " (dry-run)" if args.dry_run else ""
    logger.success(
        f"Processo finalizado. Arquivos baixados: {total_downloaded}{dry_suffix}"
    )


if __name__ == "__main__":
    main()

