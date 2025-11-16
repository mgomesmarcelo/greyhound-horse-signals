from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ast
import pandas as pd
from dateutil import parser as date_parser
from loguru import logger

from ..config import settings
from ..utils.text import clean_horse_name, normalize_track_name


_TRAP_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _strip_trap_prefix(name: str) -> str:
    return _TRAP_PREFIX_RE.sub("", name or "").strip()


def _extract_track_from_menu_hint(menu_hint: str) -> str:
    # Normaliza prefixos regionais e separadores (ex.: "IRE / Punchestown 31st Dec" -> "Punchestown 31st Dec")
    text = (menu_hint or "").strip()
    text = re.sub(r"^\s*(?:UK|IRE|IRL|GB|UK\s*&\s*IRE)\s*/\s*", "", text, flags=re.IGNORECASE)
    # Troca separadores incomuns por espaco
    text = re.sub(r"[/|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Extrai apenas letras e espacos ate o primeiro digito (ex.: "Punchestown 31st Dec" -> "Punchestown")
    m = re.match(r"^([A-Za-z][A-Za-z\s]+?)(?:\s*\d|$)", text)
    base = m.group(1) if m else text
    return normalize_track_name(base)


def _to_iso_yyyy_mm_dd_thh_mm(value: str) -> str:
    # event_dt exemplo: "19-09-2025 20:01"
    try:
        dt = date_parser.parse(value, dayfirst=True)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _to_iso_series(values: pd.Series) -> pd.Series:
    """Converte uma Series de datas variadas para ISO (YYYY-MM-DDTHH:MM) de forma vetorizada.

    Usa pandas.to_datetime com dayfirst=True e errors='coerce' para robustez e performance,
    evitando chamadas por linha ao dateutil (que sao lentas e podem travar em strings ruins).
    """
    try:
        s = values.astype(str).str.strip()
        # Normaliza separador
        s = s.str.replace("/", "-", regex=False)
        # Tentativa 1: dd-mm-YYYY HH:MM (mais comum nos arquivos)
        dt = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
        mask = dt.isna()
        if mask.any():
            # Tentativa 2: YYYY-mm-dd HH:MM
            dt2 = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M", errors="coerce")
            dt.loc[mask] = dt2
            mask = dt.isna()
        if mask.any():
            # Fallback pontual usando dateutil apenas nos remanescentes
            def _safe_parse(v: str) -> pd.Timestamp | None:
                try:
                    return pd.Timestamp(date_parser.parse(v, dayfirst=True))
                except Exception:
                    return pd.NaT
            dt3 = s[mask].map(_safe_parse)
            dt.loc[mask] = pd.to_datetime(dt3, errors="coerce")
        out = dt.dt.strftime("%Y-%m-%dT%H:%M")
        return out.fillna("")
    except Exception:
        # Fallback final: funcao escalar
        return values.astype(str).map(_to_iso_yyyy_mm_dd_thh_mm)


def _parse_forecast_top3(text: str) -> List[str]:
    """Extrai apenas os 3 primeiros nomes previstos da string TimeformForecast.

    Suporta formatos como:
    - "TimeformForecast : 2.50 Nome A, 3.50 Nome B, 4.50 Nome C, ..."
    - "Nome A (2/1), Nome B (5/2), Nome C (4/1), ..."
    - "2.50 Nome A, 3.50 Nome B, 4.50 Nome C"
    """
    if not isinstance(text, str):
        return []
    s = re.sub(r"(?i)\btimeformforecast\s*:\s*", "", text.strip())
    parts = [p.strip() for p in s.split(",") if p and isinstance(p, str)]
    names: List[str] = []
    for p in parts:
        # remove odds no inicio (ex.: 2.50 Nome A)
        m1 = re.match(r"^\s*\d+(?:\.\d+)?\s+(.+)$", p)
        if m1:
            candidate = m1.group(1).strip()
        else:
            # remove odds entre parenteses no final (ex.: Nome A (2/1))
            candidate = re.sub(r"\s*\([^\)]*\)\s*$", "", p).strip()
        candidate = _strip_trap_prefix(candidate)
        cleaned = clean_horse_name(candidate)
        if cleaned and cleaned not in names:
            names.append(cleaned)
        if len(names) >= 3:
            break
    return names


@dataclass
class RunnerBF:
    selection_name_raw: str
    selection_name_clean: str
    pptradedvol: float
    bsp: float
    win_lose: int


def load_betfair_win() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    """Carrega todos os CSVs de resultados WIN (UK/IRE) e indexa por (track_key, race_iso).

    Compativel com padroes como:
    - dwbfpricesukwin*.csv
    - dwbfpricesirewin*.csv
    - ou, genericamente, dwbfprices*win*.csv
    """
    result_dir = settings.DATA_DIR / "Result"
    all_files = sorted(result_dir.glob("dwbfprices*win*.csv"))
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for csv_path in all_files:
        try:
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING)
            # Normaliza cabecalhos para minusculas (compativel com arquivos antigos em CAIXA ALTA)
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception as e:
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        # Garante colunas
        for col in ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]:
            if col not in df.columns:
                df[col] = ""

        # Limpeza e normalizacao
        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = _to_iso_series(df["event_dt"].astype(str))
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), grp in df.groupby(["track_key", "race_iso" ], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, r in grp.iterrows():
                name_clean = r["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                runners[name_clean] = RunnerBF(
                    selection_name_raw=r["selection_name_raw"],
                    selection_name_clean=name_clean,
                    pptradedvol=float(r["pptradedvol"]),
                    bsp=float(r["bsp"]) if pd.notna(r["bsp"]) else float("nan"),
                    win_lose=int(r["win_lose"]),
                )

    logger.info("Betfair WIN index criado: {} corridas", len(index))
    return index


def load_betfair_place() -> Dict[Tuple[str, str], Dict[str, RunnerBF]]:
    """Carrega todos os CSVs de resultados PLACE (UK/IRE) e indexa por (track_key, race_iso).

    Compativel com padroes como:
    - dwbfpricesukplace*.csv
    - dwbfpricesireplace*.csv
    - ou, genericamente, dwbfprices*place*.csv
    """
    result_dir = settings.DATA_DIR / "Result"
    all_files = sorted(result_dir.glob("dwbfprices*place*.csv"))
    index: Dict[Tuple[str, str], Dict[str, RunnerBF]] = {}

    for csv_path in all_files:
        try:
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING)
            # Normaliza cabecalhos para minusculas (compativel com arquivos antigos em CAIXA ALTA)
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception as e:
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        # Garante colunas
        for col in ["menu_hint", "event_dt", "selection_name", "pptradedvol", "bsp", "win_lose"]:
            if col not in df.columns:
                df[col] = ""

        # Limpeza e normalizacao
        df["track_key"] = df["menu_hint"].astype(str).map(_extract_track_from_menu_hint)
        df["race_iso"] = _to_iso_series(df["event_dt"].astype(str))
        df["selection_name_raw"] = df["selection_name"].astype(str)
        df["selection_name_clean"] = df["selection_name_raw"].map(_strip_trap_prefix).map(clean_horse_name)
        df["pptradedvol"] = pd.to_numeric(df["pptradedvol"], errors="coerce").fillna(0.0)
        df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
        df["win_lose"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype(int)

        for (track_key, race_iso), grp in df.groupby(["track_key", "race_iso" ], dropna=False):
            if not track_key or not race_iso:
                continue
            runners: Dict[str, RunnerBF] = index.setdefault((track_key, race_iso), {})
            for _, r in grp.iterrows():
                name_clean = r["selection_name_clean"]
                if not isinstance(name_clean, str) or not name_clean:
                    continue
                runners[name_clean] = RunnerBF(
                    selection_name_raw=r["selection_name_raw"],
                    selection_name_clean=name_clean,
                    pptradedvol=float(r["pptradedvol"]),
                    bsp=float(r["bsp"]) if pd.notna(r["bsp"]) else float("nan"),
                    win_lose=int(r["win_lose"]),
                )

    logger.info("Betfair PLACE index criado: {} corridas", len(index))
    return index

def load_timeform_top3() -> List[dict]:
    """Carrega todos os CSVs timeform_top3_*.csv e retorna linhas normalizadas.

    Suporta dois esquemas:
    - Esquema antigo: colunas TimeformTop1/2/3
    - Esquema novo: colunas TimeformPrev_list (string representando lista) ou TimeformPrev (string com nomes separados por ';')
    """
    tf_dir = settings.DATA_DIR / "timeform_top3"
    rows: List[dict] = []
    for csv_path in sorted(tf_dir.glob("timeform_top3_*.csv")):
        try:
            # Usa engine=python e on_bad_lines='skip' para tolerar linhas malformadas
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, engine="python", on_bad_lines="skip")
        except Exception as e:
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        # Normaliza colunas basicas
        for col in ["track_name", "race_time_iso"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = str(r.get("race_time_iso", ""))
            names: List[str] = []
            if {"TimeformTop1", "TimeformTop2", "TimeformTop3"}.issubset(set(df.columns)):
                names = [clean_horse_name(str(r.get(c, ""))) for c in ["TimeformTop1", "TimeformTop2", "TimeformTop3"]]
            else:
                # Tenta esquema novo
                raw_list = r.get("TimeformPrev_list")
                raw_text = r.get("TimeformPrev")
                parsed: List[str] = []
                if isinstance(raw_list, str) and raw_list.strip().startswith("["):
                    try:
                        tmp = ast.literal_eval(raw_list)
                        if isinstance(tmp, list):
                            parsed = [str(x) for x in tmp]
                    except Exception:
                        parsed = []
                if not parsed and isinstance(raw_text, str):
                    # separa por ';'
                    parsed = [p.strip() for p in raw_text.split(";") if isinstance(p, str) and p.strip()]
                names = [clean_horse_name(x) for x in parsed if isinstance(x, str) and x]
                # garante tamanho 3
                while len(names) < 3:
                    names.append("")
                names = names[:3]
            if not track or not race_iso or not any(names):
                continue
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": {
                    "track_name": track,
                    "race_time_iso": race_iso,
                    "TimeformTop1": names[0] if len(names) > 0 else "",
                    "TimeformTop2": names[1] if len(names) > 1 else "",
                    "TimeformTop3": names[2] if len(names) > 2 else "",
                },
            })
    logger.info("Timeform Top3 carregado: {} corridas", len(rows))
    return rows


def load_timeform_forecast_top3() -> List[dict]:
    """Carrega TimeformForecast_*.csv e retorna linhas com apenas os 3 primeiros previstos.

    Mantem o mesmo formato de saida de load_timeform_top3, preenchendo
    os campos TimeformTop1/2/3 com os nomes extraidos.
    """
    tf_dir = settings.DATA_DIR / "TimeformForecast"
    rows: List[dict] = []
    for csv_path in sorted(tf_dir.glob("TimeformForecast_*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, engine="python", on_bad_lines="skip")
        except Exception as e:
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        for col in ["track_name", "race_time_iso", "TimeformForecast"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = str(r.get("race_time_iso", ""))
            names = _parse_forecast_top3(str(r.get("TimeformForecast", "")))
            if not track or not race_iso or not names:
                continue
            # garante 3 posicoes
            while len(names) < 3:
                names.append("")
            raw_like = {
                "track_name": track,
                "race_time_iso": race_iso,
                "TimeformTop1": names[0],
                "TimeformTop2": names[1],
                "TimeformTop3": names[2],
            }
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": raw_like,
            })

    logger.info("Timeform Forecast(Top3) carregado: {} corridas", len(rows))
    return rows


def load_sportinglife_top3() -> List[dict]:
    """Carrega sportinglife_top3_*.csv e retorna linhas normalizadas no mesmo formato do Timeform.

    Colunas esperadas: track_name, race_time_iso, TimeformTop1/2/3
    """
    sl_dir = settings.DATA_DIR / "sportinglife_top3"
    rows: List[dict] = []
    for csv_path in sorted(sl_dir.glob("sportinglife_top3_*.csv")):
        try:
            # Ignora arquivos vazios rapidamente
            try:
                if csv_path.stat().st_size == 0:
                    logger.warning("Arquivo vazio ignorado: {}", csv_path.name)
                    continue
            except Exception:
                pass
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, engine="python", on_bad_lines="skip")
            if df is None or (hasattr(df, "empty") and df.empty and len(df.columns) == 0):
                logger.warning("Sem colunas/linhas em {}  ignorado", csv_path.name)
                continue
        except Exception as e:
            # Muitos arquivos antigos estao vazios ou com BOM estranho; trate como ignoravel
            if "No columns to parse from file" in str(e):
                logger.warning("Sem colunas em {}  ignorado", csv_path.name)
                continue
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        for col in ["track_name", "race_time_iso", "TimeformTop1", "TimeformTop2", "TimeformTop3"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = str(r.get("race_time_iso", ""))
            names = [clean_horse_name(str(r.get(c, ""))) for c in ["TimeformTop1", "TimeformTop2", "TimeformTop3"]]
            if not track or not race_iso or not any(names):
                continue
            while len(names) < 3:
                names.append("")
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": {
                    "track_name": track,
                    "race_time_iso": race_iso,
                    "TimeformTop1": names[0],
                    "TimeformTop2": names[1],
                    "TimeformTop3": names[2],
                },
            })
    logger.info("Sporting Life Top3 carregado: {} corridas", len(rows))
    return rows


def load_sportinglife_forecast_top3() -> List[dict]:
    """Carrega SportingLifeForecast_*.csv e retorna os 3 primeiros do Forecast.

    Mantem o mesmo formato de saida dos loaders de Top3.
    """
    sl_dir = settings.DATA_DIR / "SportingLifeForecast"
    rows: List[dict] = []
    for csv_path in sorted(sl_dir.glob("SportingLifeForecast_*.csv")):
        try:
            try:
                if csv_path.stat().st_size == 0:
                    logger.warning("Arquivo vazio ignorado: {}", csv_path.name)
                    continue
            except Exception:
                pass
            df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING, engine="python", on_bad_lines="skip")
            if df is None or (hasattr(df, "empty") and df.empty and len(df.columns) == 0):
                logger.warning("Sem colunas/linhas em {}  ignorado", csv_path.name)
                continue
        except Exception as e:
            if "No columns to parse from file" in str(e):
                logger.warning("Sem colunas em {}  ignorado", csv_path.name)
                continue
            logger.error("Falha ao ler {}: {}", csv_path.name, e)
            continue

        for col in ["track_name", "race_time_iso", "TimeformForecast"]:
            if col not in df.columns:
                df[col] = pd.NA

        for _, r in df.iterrows():
            track = normalize_track_name(str(r.get("track_name", "")))
            race_iso = str(r.get("race_time_iso", ""))
            names = _parse_forecast_top3(str(r.get("TimeformForecast", "")))
            if not track or not race_iso or not names:
                continue
            while len(names) < 3:
                names.append("")
            raw_like = {
                "track_name": track,
                "race_time_iso": race_iso,
                "TimeformTop1": names[0],
                "TimeformTop2": names[1],
                "TimeformTop3": names[2],
            }
            rows.append({
                "track_key": track,
                "race_iso": race_iso,
                "top_names": names[:3],
                "raw": raw_like,
            })
    logger.info("Sporting Life Forecast(Top3) carregado: {} corridas", len(rows))
    return rows


def _calc_signals_for_race(
    tf_row: dict,
    bf_win_index: Dict[Tuple[str, str], Dict[str, RunnerBF]],
    bf_place_index: Dict[Tuple[str, str], Dict[str, RunnerBF]] | None = None,
    market: str = "win",
    strategy: str = "lay",
    leader_share_min: float = 0.5,
) -> dict | None:
    track_key = tf_row["track_key"]
    race_iso = tf_row["race_iso"]
    top_names = [n for n in tf_row["top_names"] if isinstance(n, str) and n]
    # Selecao por volume sempre no mercado WIN
    group = bf_win_index.get((track_key, race_iso))
    if not group:
        return None

    # Coleta volumes e BSP para os tres de referencia
    triples: List[Tuple[str, float, float]] = []  # (name_clean, vol, bsp)
    for name in top_names:
        r = group.get(name)
        if not r or pd.isna(r.bsp):
            return None
        triples.append((name, max(0.0, float(r.pptradedvol)), float(r.bsp)))

    if len(triples) < 3:
        return None

    # Ordena por volume desc entre os Top3 de referencia
    triples_sorted = sorted(triples, key=lambda t: t[1], reverse=True)
    first, second, third = triples_sorted[0], triples_sorted[1], triples_sorted[2]

    # Metricas auxiliares entre 2o e 3o (usadas pela estrategia LAY)
    vol2, vol3 = second[1], third[1]
    pct_diff = (vol2 - vol3) / vol2 if vol2 > 0 else float("inf")
    ratio = (vol2 / vol3) if vol3 > 0 else float("inf")

    # Para LAY, manter a regra de queda > 50% relativa ao segundo
    if strategy == "lay":
        if vol3 <= 0:
            return None
        if pct_diff <= 0.5:
            return None

    # Para BACK, a selecao e o lider por volume (entre os Top3),
    # com checagem de participacao sobre o total da corrida (no mercado WIN)
    # Total de volume no WIN para toda a corrida
    total_vol_race = 0.0
    for _name, r in bf_win_index.get((track_key, race_iso), {}).items():
        total_vol_race += max(0.0, float(r.pptradedvol))
    leader_share = (first[1] / total_vol_race) if total_vol_race > 0 else 0.0
    if strategy == "back" and leader_share < float(leader_share_min):
        return None

    # Define alvo conforme estrategia
    if strategy == "back":
        # Alvo e o lider por volume entre os Top3
        target_name_clean = first[0]
        target_bsp_win = first[2]
    else:
        # LAY: alvo e o 3o por volume entre os Top3
        target_name_clean = third[0]
        target_bsp_win = third[2]

    # Recupera runner conforme mercado (win/place) para obter BSP/label corretos
    if market == "place" and bf_place_index is not None:
        target_runner = bf_place_index.get((track_key, race_iso), {}).get(target_name_clean)
    else:
        target_runner = bf_win_index.get((track_key, race_iso), {}).get(target_name_clean)
    if not target_runner:
        return None
    target_win_lose = int(target_runner.win_lose)

    # Para place, usar BSP do mercado PLACE; para win, BSP do WIN do terceiro por volume
    odd = float(target_runner.bsp) if market == "place" and target_runner is not None else target_bsp_win

    # Stake base
    stake_fix10 = 10.00

    if strategy == "back":
        # Back: lucro quando win_lose==1, PnL = stake*(odd-1) ou -stake
        if target_win_lose == 1:
            pnl_stake10 = stake_fix10 * max(0.0, odd - 1.0)
            is_green = True
        else:
            pnl_stake10 = -stake_fix10
            is_green = False
        # Campos nao aplicaveis ao back, manter como 0 para compatibilidade
        liability_from_stake10 = 0.0
        liability_fix10 = 0.0
        stake_from_liab10 = 0.0
        pnl_liab10 = 0.0
    else:
        # Lay: perde quando win_lose==1
        liability_from_stake10 = stake_fix10 * max(0.0, odd - 1.0)
        liability_fix10 = 10.00
        stake_from_liab10 = (liability_fix10 / max(0.001, odd - 1.0))
        commission_rate = 0.065
        if target_win_lose == 1:
            pnl_stake10 = -liability_from_stake10
            pnl_liab10 = -liability_fix10
            is_green = False
        else:
            gross_gain_stake10 = stake_fix10
            gross_gain_liab10 = stake_from_liab10
            pnl_stake10 = gross_gain_stake10 * (1.0 - commission_rate)
            pnl_liab10 = gross_gain_liab10 * (1.0 - commission_rate)
            is_green = True

    raw = tf_row["raw"]
    # Helpers seguros para obter volumes dos Top1/2/3
    def _vol_for(name_raw: object) -> float:
        name = clean_horse_name(str(name_raw)) if isinstance(name_raw, (str,)) else ""
        return next((v for n, v, _ in triples if n == name), 0.0)

    out = {
        "date": race_iso.split("T")[0],
        "track_name": raw.get("track_name", ""),
        "race_time_iso": race_iso,
        "tf_top1": raw.get("TimeformTop1", ""),
        "tf_top2": raw.get("TimeformTop2", ""),
        "tf_top3": raw.get("TimeformTop3", ""),
        "vol_top1": _vol_for(raw.get("TimeformTop1")),
        "vol_top2": _vol_for(raw.get("TimeformTop2")),
        "vol_top3": _vol_for(raw.get("TimeformTop3")),
        "second_name_by_volume": second[0],
        "third_name_by_volume": third[0],
        "ratio_second_over_third": round(ratio, 2),
        # Percentual agora relativo ao volume do 2o colocado por volume
        "pct_diff_second_vs_third": round(pct_diff * 100.0, 2),
        # Campos de alvo, diferenciando por estrategia
        "lay_target_name": target_name_clean if strategy == "lay" else "",
        "lay_target_bsp": round(odd, 2) if strategy == "lay" else float("nan"),
        "back_target_name": target_name_clean if strategy == "back" else "",
        "back_target_bsp": round(odd, 2) if strategy == "back" else float("nan"),
        # Participacao do lider (sempre calculada com base no WIN)
        "leader_name_by_volume": first[0],
        "leader_volume_share_pct": round(leader_share * 100.0, 2),
        "stake_fixed_10": round(stake_fix10, 2),
        "liability_from_stake_fixed_10": round(liability_from_stake10, 2),
        "stake_for_liability_10": round(stake_from_liab10, 2),
        "liability_fixed_10": round(liability_fix10, 2),
        "win_lose": target_win_lose,
        "is_green": is_green,
        "pnl_stake_fixed_10": round(pnl_stake10, 2),
        "pnl_liability_fixed_10": round(pnl_liab10, 2),
        # ROI: para lay usa liability; para back usa stake
        "roi_row_stake_fixed_10": round((pnl_stake10 / (liability_from_stake10 if strategy == "lay" else stake_fix10)) if (liability_from_stake10 if strategy == "lay" else stake_fix10) > 0 else 0.0, 4),
        "roi_row_liability_fixed_10": round((pnl_liab10 / liability_fix10) if liability_fix10 > 0 else 0.0, 4),
        "market": market,
        "strategy": strategy,
    }
    return out


def generate_signals(source: str = "top3", market: str = "win", strategy: str = "lay", leader_share_min: float = 0.5, provider: str = "timeform") -> pd.DataFrame:
    bf_win_index = load_betfair_win()
    bf_place_index = load_betfair_place() if market == "place" else None
    if provider == "sportinglife":
        if source == "forecast":
            tf_rows = load_sportinglife_forecast_top3()
        else:
            tf_rows = load_sportinglife_top3()
    else:
        if source == "forecast":
            tf_rows = load_timeform_forecast_top3()
        else:
            tf_rows = load_timeform_top3()

    signals: List[dict] = []
    for row in tf_rows:
        result = _calc_signals_for_race(row, bf_win_index, bf_place_index, market=market, strategy=strategy, leader_share_min=leader_share_min)
        if result:
            signals.append(result)

    df = pd.DataFrame(signals)
    logger.info("Sinais encontrados (provider={}, source={}, market={}, strategy={}, leader_share_min={}): {}", provider, source, market, strategy, leader_share_min, len(df))
    return df


def write_signals_csv(df: pd.DataFrame, source: str = "top3", market: str = "win", strategy: str = "lay", provider: str = "timeform") -> Path:
    base_dir = settings.DATA_DIR / "signals"
    if provider == "sportinglife":
        out_dir = base_dir / "sportinglife"
    else:
        out_dir = base_dir
    _ensure_dir(out_dir)
    if strategy == "back":
        if source == "forecast" and market == "place":
            name = "back_signals_forecast_place.csv"
        elif source == "forecast":
            name = "back_signals_forecast.csv"
        elif market == "place":
            name = "back_signals_place.csv"
        else:
            name = "back_signals.csv"
    else:
        if source == "forecast" and market == "place":
            name = "lay_signals_forecast_place.csv"
        elif source == "forecast":
            name = "lay_signals_forecast.csv"
        elif market == "place":
            name = "lay_signals_place.csv"
        else:
            name = "lay_signals.csv"
    out_path = out_dir / name
    df = df.copy()
    df["source"] = source
    df["market"] = market
    df["strategy"] = strategy
    df["provider"] = provider
    if df.empty:
        # cria CSV vazio com cabecalhos padrao
        df_sorted = pd.DataFrame([], columns=[
            "date","track_name","race_time_iso",
            "tf_top1","tf_top2","tf_top3",
            "vol_top1","vol_top2","vol_top3",
            "second_name_by_volume","third_name_by_volume",
            "ratio_second_over_third","pct_diff_second_vs_third",
            "lay_target_name","lay_target_bsp",
            "back_target_name","back_target_bsp",
            "leader_name_by_volume","leader_volume_share_pct",
            "stake_fixed_10","liability_from_stake_fixed_10",
            "stake_for_liability_10","liability_fixed_10",
            "win_lose","is_green","pnl_stake_fixed_10","pnl_liability_fixed_10",
            "roi_row_stake_fixed_10","roi_row_liability_fixed_10",
            "source","market","strategy",
        ])
    else:
        df_sorted = df.sort_values(["date", "track_name", "race_time_iso"]).reset_index(drop=True)
    df_sorted.to_csv(out_path, index=False, encoding=settings.CSV_ENCODING)
    logger.info("Gerado: {} ({} linhas)", out_path, len(df_sorted))
    return out_path


