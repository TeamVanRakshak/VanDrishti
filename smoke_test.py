"""Verify real YOLO inference against the bundled standard sample image."""
import json
from pathlib import Path
from PIL import Image
from core import annotate
from detector import load_model, predict

root = Path(__file__).resolve().parent
model = load_model()
out = root / "samples" / "verified"
out.mkdir(exist_ok=True)
for filename, expected in [("bus.jpg", {"person", "bus"}), ("giraffes.jpg", {"giraffe"}), ("zebra.jpg", {"zebra"}), ("gorilla.jpg", {"gorilla"})]:
    image = Image.open(root / "samples" / filename).convert("RGB")
    objects = predict(model, image, .35)
    labels = {d["label"] for d in objects}
    assert expected <= labels, f"Unexpected {filename} detections: {labels}"
    stem = Path(filename).stem
    annotate(image, objects).save(out / f"{stem}-detected.jpg")
    (out / f"{stem}-detections.json").write_text(json.dumps(objects, indent=2), encoding="utf-8")
    print(f"REAL YOLO TEST PASSED: {filename}; {len(objects)} objects; labels={sorted(labels)}")
