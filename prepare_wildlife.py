"""Prepare YOLOE named targets once for offline inference."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
(ROOT / ".runtime" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "false"
from vocabulary import CLASSES
from ultralytics import YOLOE

(ROOT / "models").mkdir(exist_ok=True)
os.chdir(ROOT / "models")
model = YOLOE("yoloe-11m-seg.pt")
model.set_classes(CLASSES, model.get_text_pe(CLASSES))
if hasattr(model.model, "clip_model"):
    del model.model.clip_model
model.save(str(ROOT / "models" / "vanrakshak-wildlife.pt"))
print(f"Prepared offline YOLOE detector with {len(CLASSES)} named targets.")
