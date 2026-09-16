from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o dashboard do cliente (porta 8503).")
    parser.add_argument("--port", type=int, default=8503, help="Porta do servidor (default: 8503)")
    parser.add_argument("--address", type=str, default="localhost", help="Endereco do servidor (default: localhost)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    script_path = project_root / "scripts" / "dashboard_cliente.py"
    if not script_path.exists():
        print(f"Arquivo nao encontrado: {script_path}", file=sys.stderr)
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
