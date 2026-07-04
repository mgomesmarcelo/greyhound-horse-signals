import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.core.config import ensure_data_dir
from src.greyhounds.config import settings as gh_settings
from src.horses.config import settings as ho_settings
from scripts.greyhounds.streamlit_greyhounds_app import parse_strategies_csv, normalize_track_name

def _build_daily_mask(df: pd.DataFrame, strategy: dict) -> pd.Series:
    """Cria a mascara de filtros para os dados de HOJE.
    Ignora bsp e value_ratio na filtragem porque eles serao passados como MinPrice/MaxPrice para o bot.
    """
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)

    def _get_list(key: str) -> list:
        v = strategy.get(key)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                pass
        return v if isinstance(v, list) else []

    tracks = _get_list("tracks_ms")
    if tracks and "track_name" in df.columns:
        mask &= df["track_name"].astype(str).isin([str(t) for t in tracks])

    traps = _get_list("trap_ms")
    if traps and "trap_number" in df.columns:
        trap_series = pd.to_numeric(df["trap_number"], errors="coerce").astype("Int64")
        mask &= trap_series.isin([int(t) for t in traps]).fillna(False)

    cats = _get_list("cats_ms")
    if cats and "category" in df.columns:
        cat_series = df["category"].astype(str).str.upper()
        cat_letters = cat_series.str.replace(r'[^A-Z]', '', regex=True)
        target_cats = [str(c).upper() for c in cats]
        mask &= (cat_series.isin(target_cats)) | (cat_letters.isin(target_cats))

    subcats = _get_list("subcats_ms")
    if subcats and "category_token" in df.columns:
        mask &= df["category_token"].astype(str).isin([str(s) for s in subcats])

    nr = _get_list("num_runners_ms")
    if nr and "num_runners" in df.columns:
        nr_series = pd.to_numeric(df["num_runners"], errors="coerce")
        mask &= nr_series.isin([int(x) for x in nr]).fillna(False)

    rule_slug = str(strategy.get("rule_select_label", "")).lower()
    if "forecast" in rule_slug or "forecast_odds" in rule_slug:
        ranks = _get_list("forecast_rank_ms")
        if ranks and "forecast_rank" in df.columns:
            rank_series = pd.to_numeric(df["forecast_rank"], errors="coerce")
            mask &= rank_series.isin([int(x) for x in ranks if x is not None]).fillna(False)

    return mask

def _format_bsp(val: Any) -> str:
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return ""

def _format_start_time(race_iso: str) -> str:
    """Converte 2026-06-11T14:23 para 11/06/2026 14:23:00 (formato BFB)."""
    try:
        dt = datetime.strptime(str(race_iso), "%Y-%m-%dT%H:%M")
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except (ValueError, TypeError):
        return str(race_iso)

def process_strategies_for_sport(sport: str, today_str: str):
    logger.info(f"Gerando entradas de HOJE para {sport} ({today_str})...")
    
    project_root = Path(__file__).resolve().parents[1]
    bot_strategies_dir = project_root / "config" / "bot_strategies" / sport
    out_dir = project_root / "data" / "daily_tips" / sport
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not bot_strategies_dir.exists():
        logger.info(f"Pasta {bot_strategies_dir} nao existe. Nenhuma estrategia ativa.")
        return
        
    strategy_files = list(bot_strategies_dir.glob("*.csv"))
    if not strategy_files:
        logger.info(f"Nenhum arquivo CSV encontrado em {bot_strategies_dir}.")
        return

    # Tenta carregar as corridas de hoje
    # No caso de galgos, a raspagem gera TimeformForecast_YYYY-MM-DD.parquet e timeform_top3...
    if sport == "greyhounds":
        forecast_path = gh_settings.PROCESSED_TIMEFORM_FORECAST_DIR / f"TimeformForecast_{today_str}.parquet"
        top3_path = gh_settings.PROCESSED_TIMEFORM_TOP3_DIR / f"timeform_top3_{today_str}.parquet"
        cards_path = gh_settings.DATA_DIR / "timeform_cards" / f"timeform_cards_{today_str}.parquet"
        race_links_path = gh_settings.DATA_DIR / "race_links" / f"race_links_{today_str}.csv"
    else:
        # Horses
        forecast_path = ho_settings.PROCESSED_TIMEFORM_FORECAST_DIR / f"TimeformForecast_{today_str}.parquet"
        top3_path = ho_settings.PROCESSED_TIMEFORM_TOP3_DIR / f"timeform_top3_{today_str}.parquet"
        cards_path = ho_settings.DATA_DIR / "timeform_cards" / f"timeform_cards_{today_str}.parquet"
        race_links_path = ho_settings.DATA_DIR / "race_links" / f"race_links_{today_str}.csv"

    if sport == "greyhounds":
        from src.greyhounds.analysis.signals import _parse_forecast_all
        from src.greyhounds.utils.text import clean_greyhound_name as clean_name
    else:
        from src.horses.analysis.signals import _parse_forecast_all
        from src.horses.utils.text import clean_horse_name as clean_name

    df_forecast = pd.read_parquet(forecast_path) if forecast_path.exists() else pd.DataFrame()
    df_top3 = pd.read_parquet(top3_path) if top3_path.exists() else pd.DataFrame()
    df_cards = pd.read_parquet(cards_path) if cards_path.exists() else pd.DataFrame()
    if not df_cards.empty and "track_name" in df_cards.columns and "race_time_iso" in df_cards.columns:
        if "num_runners" not in df_cards.columns:
            # Calcula total de corredores por corrida
            runner_counts = df_cards.groupby(["track_name", "race_time_iso"]).size().reset_index(name="num_runners")
            df_cards = df_cards.merge(runner_counts, on=["track_name", "race_time_iso"], how="left")

    # Carregar race_links para MarketId e RaceUrl
    lookup_links = {}
    lookup_urls = {}
    if race_links_path.exists():
        try:
            df_links = pd.read_csv(race_links_path)
            for _, r_link in df_links.iterrows():
                url = str(r_link.get("race_url", ""))
                if url:
                    m_id = url.split("/")[-1]
                    t_name = normalize_track_name(str(r_link["track_name"]))
                    r_time = str(r_link["race_time_iso"]).strip()
                    lookup_links[(t_name, r_time)] = m_id
                    lookup_urls[(t_name, r_time)] = url
        except Exception as e:
            logger.warning(f"Falha ao ler race_links: {e}")

    if not df_forecast.empty:
        parsed_rows = []
        for _, r in df_forecast.iterrows():
            items = _parse_forecast_all(str(r.get("TimeformForecast", "")))
            for item in items:
                new_r = r.to_dict()
                new_r.update(item)
                parsed_rows.append(new_r)
        df_forecast = pd.DataFrame(parsed_rows)
        
    if not df_top3.empty:
        parsed_rows = []
        for _, r in df_top3.iterrows():
            for rank in [1, 2, 3]:
                col = f"TimeformTop{rank}"
                if col in r and pd.notna(r[col]) and r[col]:
                    new_r = r.to_dict()
                    cname = clean_name(str(r[col]))
                    if cname:
                        new_r["forecast_rank"] = rank
                        new_r["forecast_name_clean"] = cname
                        parsed_rows.append(new_r)
        df_top3 = pd.DataFrame(parsed_rows)

    all_tips = []

    for strat_file in strategy_files:
        strats = parse_strategies_csv(open(strat_file, "rb"))
        if not strats:
            continue
            
        strat = strats[0]
        strategy_name = strat.get("strategy_name", strat_file.stem)
        safe_strategy_name = strat_file.stem
        rule_label = str(strat.get("rule_select_label", "")).lower()
        source_label = str(strat.get("source_select_label", "")).lower()
        entry_type = str(strat.get("entry_type", "")).lower()
        mercado_raw = str(strat.get("market", "win")).lower()
        mercado_map = {"win": "WIN", "placed": "PLACE"}
        market_type = mercado_map.get(mercado_raw, mercado_raw.upper())
        
        # Decide qual df usar base na fonte
        use_forecast = "forecast" in source_label or "forecast" in rule_label
        df_base = df_forecast.copy() if use_forecast else df_top3.copy()
        
        if df_base.empty:
            logger.warning(f"Base de dados de hoje {'forecast' if use_forecast else 'top3'} vazia/nao encontrada para {sport}.")
            continue

        # Normaliza colunas necessarias (simulando o enriquecimento que so aconteceria apos corrida)
        if "track_name" not in df_base.columns and "track_key" in df_base.columns:
            df_base["track_name"] = df_base["track_key"].apply(normalize_track_name)
        if "track_key" not in df_base.columns and "track_name" in df_base.columns:
            df_base["track_key"] = df_base["track_name"]
        if "race_iso" not in df_base.columns and "race_time_iso" in df_base.columns:
            df_base["race_iso"] = df_base["race_time_iso"]
        if "forecast_rank" not in df_base.columns and "rank" in df_base.columns:
            df_base["forecast_rank"] = df_base["rank"]
        
        if not df_cards.empty and "track_key" in df_base.columns and "race_iso" in df_base.columns:
            if "track_key" not in df_cards.columns and "track_name" in df_cards.columns:
                df_cards["track_key"] = df_cards["track_name"]
            if "race_iso" not in df_cards.columns and "race_time_iso" in df_cards.columns:
                df_cards["race_iso"] = df_cards["race_time_iso"]
            # Enriquecer num_runners, category
            df_cards_clean = df_cards.drop_duplicates(subset=["track_key", "race_iso"])
            cols_to_merge = ["track_key", "race_iso"]
            for col in ["num_runners", "category", "category_token"]:
                if col in df_cards_clean.columns:
                    cols_to_merge.append(col)
            df_base = df_base.merge(df_cards_clean[cols_to_merge], on=["track_key", "race_iso"], how="left")

        mask = _build_daily_mask(df_base, strat)
        df_filtered = df_base[mask].copy()
        
        if df_filtered.empty:
            logger.info(f"Nenhuma entrada para {strategy_name} hoje.")
            continue

        # Construir o CSV
        # StartTime,MarketId,MarketType,EventName,SelectionName,BetType,MinPrice,MaxPrice,Provider
        tips_for_strat = []
        for _, row in df_filtered.iterrows():
            dog_name = row.get("forecast_name_raw", row.get("name_raw", row.get("selection_name", "")))
            if not dog_name:
                dog_name = row.get("forecast_name_clean", row.get("name_clean", ""))
            if not dog_name:
                continue
                
            trap = row.get("trap_number", "")
            if pd.notna(trap) and trap != "":
                selection_name = f"{int(trap)}. {dog_name}"
            else:
                selection_name = dog_name

            bet_type = "Back" if entry_type == "back" else ("Lay" if entry_type == "lay" else "Back") # Fallback
            
            min_p = ""
            max_p = ""
            if "forecast" in rule_label:
                fo = float(row.get("forecast_odds", 0))
                vmin = float(strat.get("value_ratio_min", 1.0))
                vmax = float(strat.get("value_ratio_max", 999.0))
                bsp_lo = float(strat.get("bsp_low", 1.01))
                bsp_hi = float(strat.get("bsp_high", 1000.0))
                if fo > 0:
                    min_from_vr = round(fo * vmin, 2)
                    max_from_vr = round(fo * vmax, 2)
                    final_min = max(bsp_lo, min_from_vr)
                    final_max = min(bsp_hi, max_from_vr)
                    if final_min > final_max:
                        continue
                    min_p = f"{final_min:.2f}"
                    max_p = f"{final_max:.2f}"
            else:
                low = strat.get("bsp_low")
                high = strat.get("bsp_high")
                min_p = _format_bsp(low) if low is not None else ""
                max_p = _format_bsp(high) if high is not None else ""

            tips_for_strat.append({
                "StartTime": _format_start_time(row.get("race_iso", "")),
                "MarketId": lookup_links.get(
                    (normalize_track_name(str(row.get("track_name", ""))),
                     str(row.get("race_iso", ""))),
                    ""
                ),
                "RaceUrl": lookup_urls.get(
                    (normalize_track_name(str(row.get("track_name", ""))),
                     str(row.get("race_iso", ""))),
                    ""
                ),
                "MarketType": market_type,
                "EventName": row.get("track_name", ""),
                "SelectionName": selection_name,
                "BetType": bet_type,
                "MinPrice": min_p,
                "MaxPrice": max_p,
                "Provider": strategy_name
            })
            
        if tips_for_strat:
            df_tips = pd.DataFrame(tips_for_strat)
            indiv_path = out_dir / f"{today_str}_{safe_strategy_name}.csv"
            df_tips.to_csv(indiv_path, index=False)
            logger.info(f"Gerado: {indiv_path.name} com {len(df_tips)} sinais.")
            all_tips.extend(tips_for_strat)

    if all_tips:
        df_all = pd.DataFrame(all_tips)
        cons_path = out_dir / f"{today_str}_CONSOLIDADO.csv"
        df_all.to_csv(cons_path, index=False)
        logger.info(f"CONSOLIDADO gerado: {cons_path.name} com {len(df_all)} sinais totaal.")
    else:
        logger.info("Nenhuma estrategia gerou sinais hoje.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=["greyhounds", "horses", "all"], default="all")
    args = parser.parse_args()

    today_str = date.today().isoformat()
    
    if args.sport in ["greyhounds", "all"]:
        process_strategies_for_sport("greyhounds", today_str)
    if args.sport in ["horses", "all"]:
        process_strategies_for_sport("horses", today_str)

if __name__ == "__main__":
    main()
