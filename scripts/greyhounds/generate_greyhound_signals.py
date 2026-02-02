from __future__ import annotations

import argparse
import sys

from loguru import logger
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.greyhounds.analysis.signals import generate_signals, write_signals_csv
from src.greyhounds.config import RULE_LABELS, settings


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Gerar sinais para greyhounds combinando fonte/mercado/regra."
    )
    parser.add_argument("--source", choices=["top3", "forecast", "betfair_resultado", "both", "all"], default="all")
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
    args = parser.parse_args(argv)

    def _run_for(source_value: str, market_value: str, rule_value: str) -> None:
        df = generate_signals(
            source=source_value,
            market=market_value,
            rule=rule_value,
            leader_share_min=args.leader_share_min,
            entry_type=args.entry_type,
        )
        out_path = write_signals_csv(df, source=source_value, market=market_value, rule=rule_value)
        rule_label = RULE_LABELS.get(rule_value, rule_value)
        logger.info(
            "Concluído {} ({} / {}). Total de sinais: {}",
            source_value,
            market_value,
            rule_label,
            len(df),
        )
        print(out_path)

    if args.source == "both":
        sources = ["top3", "forecast"]
    elif args.source == "all":
        sources = ["top3", "forecast", "betfair_resultado"]
    else:
        sources = [args.source]

    markets = [args.market] if args.market != "both" else ["win", "place"]
    rules = (
        [args.rule]
        if args.rule != "both"
        else ["lider_volume_total", "terceiro_queda50", "forecast_odds"]
    )

    for source_value in sources:
        for market_value in markets:
            for rule_value in rules:
                if rule_value == "forecast_odds" and source_value != "forecast":
                    logger.info(
                        "Pulando combinação inválida: source={} market={} rule={}",
                        source_value,
                        market_value,
                        rule_value,
                    )
                    continue
                _run_for(source_value, market_value, rule_value)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

