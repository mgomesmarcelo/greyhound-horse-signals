from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.horses.analysis.signals import generate_signals, write_signals_csv
from src.horses.config import settings


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)

    parser = argparse.ArgumentParser(description="Gerar sinais para horses (Timeform/Sporting Life).")
    parser.add_argument("--source", choices=["top3", "forecast", "both"], default="both")
    parser.add_argument("--market", choices=["win", "place", "both"], default="both")
    parser.add_argument("--strategy", choices=["lay", "back", "both"], default="both")
    parser.add_argument("--provider", choices=["timeform", "sportinglife"], default="timeform")
    parser.add_argument(
        "--leader_share_min",
        type=float,
        default=0.5,
        help="Participacao minima do lider (0-1) para estrategia BACK.",
    )
    args = parser.parse_args(argv)

    def _run_for(source_val: str, market_val: str, strategy_val: str) -> None:
        df = generate_signals(
            source=source_val,
            market=market_val,
            strategy=strategy_val,
            leader_share_min=args.leader_share_min,
            provider=args.provider,
        )
        out_path = write_signals_csv(
            df,
            source=source_val,
            market=market_val,
            strategy=strategy_val,
            provider=args.provider,
        )
        logger.info(
            "Concluido {} / {} / {} [{}] -> {} sinais",
            source_val,
            market_val,
            strategy_val,
            args.provider,
            len(df),
        )
        print(out_path)

    sources = [args.source] if args.source != "both" else ["top3", "forecast"]
    markets = [args.market] if args.market != "both" else ["win", "place"]
    strategies = [args.strategy] if args.strategy != "both" else ["lay", "back"]

    for source_val in sources:
        for market_val in markets:
            for strategy_val in strategies:
                _run_for(source_val, market_val, strategy_val)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

