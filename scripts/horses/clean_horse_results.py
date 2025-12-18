from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

if __package__ in (None, ""):
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))

from src.horses.config import settings  # type: ignore[reportMissingImports]


TARGET_COLUMNS = [
	"event_id",
	"menu_hint",
	"event_name",
	"event_dt",
	"selection_id",
	"selection_name",
	"win_lose",
	"bsp",
	"pptradedvol",
]


_BSP_TWO_DEC_REGEX = re.compile(r"^\d+\.\d{2}$")


def _canonicalize_column(name: str) -> str | None:
	raw = (name or "").strip().lower()
	normalized = re.sub(r"[^a-z0-9]+", "", raw)
	mapping = {
		"eventid": "event_id",
		"_eventid": "event_id",
		"menuhint": "menu_hint",
		"eventname": "event_name",
		"eventdt": "event_dt",
		"selectionid": "selection_id",
		"selectionname": "selection_name",
		"winlose": "win_lose",
		"win_lose": "win_lose",
		"bsp": "bsp",
		"ppwap": "ppwap",
		"morningwap": "morningwap",
		"ppmax": "ppmax",
		"ppmin": "ppmin",
		"ipmax": "ipmax",
		"ipmin": "ipmin",
		"morningtradedvol": "morningtradedvol",
		"pptradedvol": "pptradedvol",
		"iptradedvol": "iptradedvol",
	}
	return mapping.get(normalized, None)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	rename_map: dict[str, str] = {}
	for col in df.columns:
		canonical = _canonicalize_column(str(col))
		if canonical:
			rename_map[col] = canonical
	return df.rename(columns=rename_map)


def is_already_clean(df: pd.DataFrame) -> bool:
	df_norm = normalize_columns(df)
	# precisa conter todas as colunas essenciais
	for col in TARGET_COLUMNS:
		if col not in df_norm.columns:
			return False
	if "bsp" not in df_norm.columns:
		return False
	series = df_norm["bsp"]
	if len(series) == 0:
		return True
	mask_notna = series.notna()
	if not mask_notna.any():
		return True
	bsp_as_str = series.astype(str)
	return _BSP_TWO_DEC_REGEX.fullmatch("0.00") is not None and bsp_as_str[mask_notna].map(
		lambda text: bool(_BSP_TWO_DEC_REGEX.fullmatch(text))
	).all()


def format_bsp_to_two_decimals(value) -> str:
	if pd.isna(value):
		return ""
	try:
		num = float(value)
		return f"{num:.2f}"
	except Exception:
		return ""


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	out = normalize_columns(df.copy())

	# Garante colunas essenciais
	for col in TARGET_COLUMNS:
		if col not in out.columns:
			out[col] = ""

	# Reordena para ter essenciais primeiro e preserva extras
	extra_cols = [c for c in out.columns if c not in TARGET_COLUMNS]
	out = out[TARGET_COLUMNS + extra_cols]

	out["bsp"] = out["bsp"].map(format_bsp_to_two_decimals)
	return out


def clean_results_dir(result_dir: Path, force: bool = False) -> int:
	changed = 0
	for csv_path in sorted(result_dir.glob("*.csv")):
		try:
			df = pd.read_csv(csv_path, encoding=settings.CSV_ENCODING)
		except Exception as exc:
			logger.error("Falha ao ler {}: {}", csv_path.name, exc)
			continue

		if not force and is_already_clean(df):
			logger.debug("Pulado (já limpo): {}", csv_path.name)
			continue

		clean_df = clean_dataframe(df)
		try:
			clean_df.to_csv(csv_path, index=False, encoding=settings.CSV_ENCODING)
			changed += 1
			logger.info("Arquivo limpo: {} ({} linhas)", csv_path.name, len(clean_df))
		except Exception as exc:
			logger.error("Falha ao escrever {}: {}", csv_path.name, exc)

	return changed


def main(argv: list[str] | None = None) -> int:
	argv = argv or sys.argv[1:]
	force = False
	if "--force" in argv:
		force = True

	logger.remove()
	logger.add(sys.stderr, level=settings.LOG_LEVEL)

	result_dir = settings.DATA_DIR / "Result"
	if not result_dir.exists():
		logger.error("Diretório não encontrado: {}", result_dir)
		return 1

	logger.info("Limpando CSVs em: {} (force={})", result_dir, force)
	changed = clean_results_dir(result_dir, force=force)
	logger.info("Concluído. Arquivos alterados: {}", changed)
	return 0


if __name__ == "__main__":
	sys.exit(main())

