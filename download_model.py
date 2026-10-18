"""Download standard Ultralytics weights once for offline inference."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
(root / ".runtime" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".runtime" / "ultralytics"))
(root / "models").mkdir(exist_ok=True)
os.chdir(root / "models")
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
print(f"Model ready: {root / 'models' / 'yolo11n.pt'} ({len(model.names)} supported categories)")
if not (root / "models" / "vanrakshak-wildlife.pt").exists():
    subprocess.check_call([sys.executable, str(root / "prepare_wildlife.py")])
subprocess.check_call([sys.executable, str(root / "download_namer.py")])
