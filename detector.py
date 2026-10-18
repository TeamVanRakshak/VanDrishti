"""Local YOLO model loading and inference."""
import os
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parent
(ROOT / ".runtime" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
_INFERENCE_LOCK = RLock()


def load_model(profile="Wildlife"):
    from ultralytics import YOLO, YOLOE
    weights = ROOT / "models" / ("vanrakshak-wildlife.pt" if profile == "Wildlife" else "yolo11n.pt")
    if not weights.exists():
        raise FileNotFoundError("YOLO weights are missing. Run setup.ps1 or python download_model.py while online.")
    model = YOLOE(str(weights)) if profile == "Wildlife" else YOLO(str(weights))
    model._vanrakshak_profile = profile
    if profile == "Wildlife":
        from naming import load_namer
        model._vanrakshak_namer = load_namer()
    return model


def predict(model, image, threshold):
    profile = getattr(model, "_vanrakshak_profile", "Standard")
    with _INFERENCE_LOCK:
        result = model.predict(source=image, conf=threshold, imgsz=960 if profile == "Wildlife" else 640,
                               device="cpu", agnostic_nms=True, iou=.5, verbose=False)[0]
        detections = [
        {"label": result.names[int(box.cls.item())], "confidence": round(float(box.conf.item()), 4),
         "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
         "model_label": result.names[int(box.cls.item())], "verified": False,
         "species_candidate": profile == "Wildlife", "detector": "YOLOE-11m + CLIP / 960px" if profile == "Wildlife" else "YOLO11n / 640px"}
        for box in result.boxes
        ]
        if profile == "Wildlife":
            from naming import refine_names
            detections = refine_names(image, detections, model._vanrakshak_namer)
        return detections
