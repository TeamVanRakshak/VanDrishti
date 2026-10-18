"""Run the API:  python -m backend  (then open http://127.0.0.1:8000/docs)."""
import argparse
from pathlib import Path

import uvicorn

from backend.api import create_app
from core import ROOT

parser = argparse.ArgumentParser(description="Run the VanDrishti API.")
parser.add_argument("--host", default="127.0.0.1", help="keep on loopback: the API has no authentication yet")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
args = parser.parse_args()
uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
