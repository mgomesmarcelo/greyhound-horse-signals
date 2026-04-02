import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.greyhounds.utils.text import normalize_track_name, clean_greyhound_name
from src.greyhounds.analysis.signals import load_betfair_win
from src.greyhounds.utils.files import write_dataframe_snapshots

def process_xbtips_to_signals():
    raw_dir = PROJECT_ROOT / "data" / "greyhounds" / "xbtips" / "raw"
    out_dir = PROJECT_ROOT / "data" / "greyhounds" / "signals"
    out_dir.mkdir(exist_ok=True, parents=True)
    processed_dir = PROJECT_ROOT / "data" / "greyhounds" / "processed" / "signals"
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    if not raw_dir.exists():
        print("Pasta raw nao encontrada.")
        return
        
    all_data = []
    
    for f in sorted(raw_dir.glob("xbtips_raw_*.csv")):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                all_data.append(df)
        except Exception:
            pass
            
    if not all_data:
        print("Nenhum CSV raw xbttips encontrado.")
        return
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    print("Carregando index Betfair WIN para cruzamento de métricas PNL/Odds...")
    bf_win_index = load_betfair_win()
    
    # Process rows
    records = []
    for _, row in final_df.iterrows():
        entry_type = row["recommendation"]
        race_iso = str(row["date"]) + "T" + str(row["time_uk"])
        track_raw = str(row["track_name"]) if pd.notna(row["track_name"]) else ""
        track_key = normalize_track_name(track_raw)
        dog_raw = str(row["greyhound_name"])
        dog_clean = clean_greyhound_name(dog_raw)
        trap_number = int(row["trap_number"]) if pd.notna(row["trap_number"]) else 0
        
        # Base record
        rec = {
            "date": row["date"],
            "market": "win",
            "entry_type": entry_type,
            "race_time_iso": race_iso,
            "track_name": track_key,
            "category_token": row["category"],
            "trap_number": trap_number,
            "greyhound_name": dog_raw,
            "score": row.get("score", 0.0),
            "gap": row.get("gap", 0.0),
            
            # Init empty metric
            "lay_target_bsp": pd.NA,
            "back_target_bsp": pd.NA,
            "lay_target_name": "",
            "back_target_name": "",
            "total_matched_volume": 0.0,
            "win_lose": pd.NA,
            "is_green": False,
            "stake_ref": 1.0,
            "pnl_stake_ref": 0.0,
            "liability_ref": 1.0,
            "pnl_liability_ref": 0.0,
            "liability_from_stake_ref": 0.0,
            "stake_for_liability_ref": 0.0,
            "roi_row_stake_ref": 0.0,
            "roi_row_liability_ref": 0.0,
            "roi_row_exposure_ref": 0.0,
            "stake_fixed_10": 10.0,
            "liability_from_stake_fixed_10": 0.0,
            "stake_for_liability_10": 0.0,
            "liability_fixed_10": 0.0,
            "pnl_stake_fixed_10": 0.0,
            "pnl_liability_fixed_10": 0.0,
            "roi_row_stake_fixed_10": 0.0,
            "roi_row_liability_fixed_10": 0.0,
            "roi_row_exposure_fixed_10": 0.0,
            # Enriquecimento (evita build_num_runners_index no dashboard)
            "num_runners": pd.NA,
            "category": row["category"] if "category" in row.index else "",
        }
        
        group = bf_win_index.get((track_key, race_iso))
        if group:
            # Total volume da corrida
            total_vol = sum(r.pptradedvol for r in group.values())
            rec["total_matched_volume"] = total_vol
            # num_runners direto do grupo (evita _build_num_runners_index no dashboard)
            rec["num_runners"] = len(group)
            
            # Try to match by name
            runner = group.get(dog_clean)
            if not runner:
                # Try to fall back by trap number if name mismatches or missing
                for r_clean, r_bf in group.items():
                    if r_bf.trap_number == trap_number:
                        runner = r_bf
                        break
                        
            if runner:
                bsp = float(runner.bsp)
                win_lose = int(runner.win_lose)
                rec["win_lose"] = win_lose
                
                commission_rate = 0.02
                legacy_scale = 10.0
                stake_ref = 1.0
                
                if entry_type == "lay":
                    rec["lay_target_name"] = runner.selection_name_raw
                    rec["lay_target_bsp"] = bsp
                    rec["is_green"] = (win_lose == 0)
                    
                    liab_ref = 1.0
                    liab_from_stake = stake_ref * max(0.0, bsp - 1.0)
                    stake_from_liab = liab_ref / max(0.0, bsp - 1.0) if bsp > 1.0 else 0.0
                    
                    if win_lose == 0:
                        pnl_stake = stake_ref * (1.0 - commission_rate)
                        pnl_liab = stake_from_liab * (1.0 - commission_rate)
                    else:
                        pnl_stake = -liab_from_stake
                        pnl_liab = -liab_ref
                        
                    rec["liability_ref"] = liab_ref
                    rec["liability_from_stake_ref"] = liab_from_stake
                    rec["stake_for_liability_ref"] = stake_from_liab
                    rec["pnl_stake_ref"] = pnl_stake
                    rec["pnl_liability_ref"] = pnl_liab
                    rec["roi_row_stake_ref"] = pnl_stake / stake_ref if stake_ref > 0 else 0.0
                    rec["roi_row_liability_ref"] = pnl_liab / liab_ref if liab_ref > 0 else 0.0
                    rec["roi_row_exposure_ref"] = pnl_stake / liab_from_stake if liab_from_stake > 0 else 0.0
                    
                    rec["liability_from_stake_fixed_10"] = liab_from_stake * legacy_scale
                    rec["stake_for_liability_10"] = stake_from_liab * legacy_scale
                    rec["liability_fixed_10"] = liab_ref * legacy_scale
                    rec["pnl_stake_fixed_10"] = pnl_stake * legacy_scale
                    rec["pnl_liability_fixed_10"] = pnl_liab * legacy_scale
                    rec["roi_row_stake_fixed_10"] = rec["roi_row_stake_ref"]
                    rec["roi_row_liability_fixed_10"] = rec["roi_row_liability_ref"]
                    rec["roi_row_exposure_fixed_10"] = rec["roi_row_exposure_ref"]
                        
                elif entry_type == "back":
                    rec["back_target_name"] = runner.selection_name_raw
                    rec["back_target_bsp"] = bsp
                    rec["is_green"] = (win_lose == 1)
                    
                    if win_lose == 1:
                        pnl_stake = stake_ref * (bsp - 1.0) * (1.0 - commission_rate)
                    else:
                        pnl_stake = -stake_ref
                        
                    rec["pnl_stake_ref"] = pnl_stake
                    rec["pnl_liability_ref"] = 0.0
                    rec["liability_ref"] = 0.0
                    rec["roi_row_stake_ref"] = pnl_stake / stake_ref if stake_ref > 0 else 0.0
                    
                    rec["pnl_stake_fixed_10"] = pnl_stake * legacy_scale
                    rec["roi_row_stake_fixed_10"] = rec["roi_row_stake_ref"]
                        
        records.append(rec)
        
    sig_df = pd.DataFrame(records)
    
    # Save lay recommendations
    df_lay = sig_df[sig_df["entry_type"] == "lay"]
    if not df_lay.empty:
        df_lay = df_lay.dropna(subset=["lay_target_bsp"])
        base_lay = "signals_xbtips_win_lay_recommendation"
        write_dataframe_snapshots(
            df_lay,
            raw_path=out_dir / f"{base_lay}.csv",
            parquet_path=processed_dir / f"{base_lay}.parquet",
        )
        print(f"Gerado {base_lay}.csv ({len(df_lay)} linhas) + .parquet")

    # Save back recommendations
    df_back = sig_df[sig_df["entry_type"] == "back"]
    if not df_back.empty:
        df_back = df_back.dropna(subset=["back_target_bsp"])
        base_back = "signals_xbtips_win_back_recommendation"
        write_dataframe_snapshots(
            df_back,
            raw_path=out_dir / f"{base_back}.csv",
            parquet_path=processed_dir / f"{base_back}.parquet",
        )
        print(f"Gerado {base_back}.csv ({len(df_back)} linhas) + .parquet")

if __name__ == "__main__":
    process_xbtips_to_signals()
