from __future__ import annotations

import argparse
import sys

from loguru import logger
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.greyhounds.analysis.signals import (
    generate_signals,
    load_betfair_place,
    load_betfair_win,
    write_signals_csv,
)
from src.greyhounds.config import RULE_LABELS, settings


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Gerar sinais para greyhounds combinando fonte/mercado/regra."
    )
    parser.add_argument("--source", choices=["top3", "forecast", "betfair_resultado", "iggy", "both", "all"], default="all")
    parser.add_argument("--market", choices=["win", "place", "both"], default="both")
    parser.add_argument(
        "--rule",
        choices=["lider_volume_total", "terceiro_queda50", "forecast_odds", "both"],
        default="both",
    )
    parser.add_argument("--entry_type", choices=["back", "lay", "both"], default="both")
    parser.add_argument(
        "--leader_share_min",
        type=float,
        default=0.5,
        help="Participação mínima do líder para a regra lider_volume_total.",
    )
    parser.add_argument("--region", default="auto", help="Regiao-alvo para gerar sinais (UK, AUS, auto)")
    args = parser.parse_args(argv)

    markets = [args.market] if args.market != "both" else ["win", "place"]
    
    # Se for auto, carregamos todos os dados sem filtro na memória para cruzar com as diferentes fontes
    bf_region_filter = None if args.region == "auto" else args.region
    bf_win_index = load_betfair_win(region_filter=bf_region_filter)
    bf_place_index = load_betfair_place(region_filter=bf_region_filter) if "place" in markets else None

    SOURCE_TO_REGION = {
        "top3": "UK",
        "forecast": "UK",
        "betfair_resultado": "UK",
        "iggy": "AUS",
        "xbtips": "UK",
        "gemini": "UK",
    }

    def _run_for(
        source_value: str,
        market_value: str,
        rule_value: str,
        bf_win: "dict",
        bf_place: "dict | None",
    ) -> None:
        active_region = args.region if args.region != "auto" else SOURCE_TO_REGION.get(source_value, "UK")
        df = generate_signals(
            source=source_value,
            market=market_value,
            rule=rule_value,
            leader_share_min=args.leader_share_min,
            entry_type=args.entry_type,
            bf_win_index=bf_win,
            bf_place_index=bf_place,
            region=active_region,
        )
        out_path = write_signals_csv(df, source=source_value, market=market_value, rule=rule_value, region=active_region)
        rule_label = RULE_LABELS.get(rule_value, rule_value)
        logger.info(
            "Concluído {} ({} / {}) [Região: {}]. Total de sinais: {}",
            source_value,
            market_value,
            rule_label,
            active_region,
            len(df),
        )
        print(out_path)

    if args.source == "both":
        sources = ["top3", "forecast"]
    elif args.source == "all":
        sources = ["top3", "forecast", "betfair_resultado", "iggy"]
    else:
        sources = [args.source]

    rules = (
        [args.rule]
        if args.rule != "both"
        else ["lider_volume_total", "terceiro_queda50", "forecast_odds"]
    )

    for source_value in sources:
        for market_value in markets:
            for rule_value in rules:
                if rule_value == "forecast_odds" and source_value not in ("forecast", "iggy"):
                    logger.info(
                        "Pulando combinação inválida: source={} market={} rule={}",
                        source_value,
                        market_value,
                        rule_value,
                    )
                    continue
                _run_for(source_value, market_value, rule_value, bf_win_index, bf_place_index)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

