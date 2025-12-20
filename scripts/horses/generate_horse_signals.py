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
    parser.add_argument(
        "--rule",
        choices=["lider_volume_total", "terceiro_queda50", "both"],
        default="both",
        help="Regra de seleção do alvo: líder por volume ou terceiro com queda de 50%.",
    )
    parser.add_argument("--entry_type", choices=["lay", "back", "both"], default="both", help="Tipo de entrada (gera um único arquivo quando 'both').")
    parser.add_argument(
        "--strategy",
        choices=["lay", "back", "both"],
        default=None,
        help="DEPRECATED: use --entry_type. Se informado, sobrescreve entry_type.",
    )
    parser.add_argument("--provider", choices=["timeform", "sportinglife", "both"], default="both")
    parser.add_argument(
        "--leader_share_min",
        type=float,
        default=0.5,
        help="Participacao minima do lider (0-1) para estrategia BACK.",
    )
    args = parser.parse_args(argv)

    def _run_for(source_val: str, market_val: str, rule_val: str, entry_type_val: str, provider_val: str) -> None:
        df = generate_signals(
            source=source_val,
            market=market_val,
            rule=rule_val,
            entry_type=entry_type_val,
            leader_share_min=args.leader_share_min,
            provider=provider_val,
            strategy=args.strategy,
        )
        out_path = write_signals_csv(
            df,
            source=source_val,
            market=market_val,
            rule=rule_val,
            provider=provider_val,
        )
        logger.info(
            "Concluido {} / {} / {} / {} [{}] -> {} sinais",
            source_val,
            market_val,
            rule_val,
            entry_type_val,
            provider_val,
            len(df),
        )
        print(out_path)

    sources = [args.source] if args.source != "both" else ["top3", "forecast"]
    markets = [args.market] if args.market != "both" else ["win", "place"]
    entry_type_cli = args.entry_type
    if args.strategy:
        entry_type_cli = args.strategy  # compatibilidade legada
    rules = [args.rule] if args.rule != "both" else ["lider_volume_total", "terceiro_queda50"]
    providers = [args.provider] if args.provider != "both" else ["timeform", "sportinglife"]

    for source_val in sources:
        for market_val in markets:
            for rule_val in rules:
                for provider_val in providers:
                    _run_for(source_val, market_val, rule_val, entry_type_cli, provider_val)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

