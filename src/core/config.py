from dataclasses import dataclass
from pathlib import Path
from typing import Union


# raiz do projeto = pasta 2 níveis acima de src/core
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"


def ensure_data_dir(*subdirs: Union[Path, str]) -> Path:
    """
    Garante a existência de DATA_DIR ou de um subdiretório específico.
    Retorna o caminho criado.
    """
    if subdirs:
        target = DATA_DIR.joinpath(*[str(part) for part in subdirs])
    else:
        target = DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = DATA_DIR
    CSV_ENCODING: str = "utf-8-sig"
    LOG_LEVEL: str = "INFO"


settings = Settings()