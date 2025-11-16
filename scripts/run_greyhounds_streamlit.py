from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o app Streamlit dos greyhounds.")
    parser.add_argument("--port", type=int, default=8501, help="Porta do servidor (default: 8501)")
    parser.add_argument("--address", type=str, default="localhost", help="Endereço do servidor (default: localhost)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    script_path = project_root / "scripts" / "streamlit_greyhounds_app.py"
    if not script_path.exists():
        print(f"Arquivo não encontrado: {script_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

