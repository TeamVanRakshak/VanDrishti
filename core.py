"""Detection, alert rules and local persistence for VanRakshak."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from vocabulary import WILDLIFE

ROOT = Path(__file__).resolve().parent
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
ANIMALS.update({"animal (unverified)", "gorilla", "primate", "deer", "tiger", "rhino", "leopard"})
ANIMALS.update(WILDLIFE)
VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck", "boat", "airplane", "train"}
CAMERAS = {
    "CAM-01 · River watch": (26.659, 93.300),
    "CAM-02 · Forest trail": (26.629, 93.337),
    "CAM-03 · Service road": (26.606, 93.290),
}


def category(label):
    if label == "person":
        return "Human"
    if label in ANIMALS:
        return "Animal"
    if label in VEHICLES:
        return "Vehicle"
    return "Other"


def assess(detections, restricted, authorized):
    """Flag observations requiring review; never infer criminal intent."""
    relevant = [d for d in detections if category(d["label"]) in {"Human", "Vehicle"}]
    if relevant and restricted and not authorized:
        return "Review required", "Person or vehicle detected at a camera marked restricted. Verify access with a ranger."
    if relevant and authorized:
        return "Authorized activity", "Activity marked as an authorized patrol or service visit by the operator."
    if relevant:
        return "Activity observed", "Person or vehicle observed outside a restricted location."
    if detections:
        return "No rule triggered", "No person or vehicle triggered the configured intrusion rule. This does not prove the area is safe."
    return "No detections", "Nothing exceeded the confidence threshold. Inspect the image manually if visibility is poor."


def verification_labels(detections):
    """Treat closed-set animal predictions as suggestions, never verified species."""
    output = []
    for detection in detections:
        item = dict(detection)
        if category(item["label"]) == "Animal" and not item.get("verified", False) and not item.get("species_candidate", False):
            item.setdefault("model_label", item["label"])
            item["label"] = "animal (unverified)"
        output.append(item)
    return output


def analyze(raw_detections, restricted, authorized):
    """Shared by the dashboard and the API: apply the labelling policy, then the access rule."""
    detections = verification_labels(raw_detections)
    status, reason = assess(detections, restricted, authorized)
    return detections, status, reason


MAX_PIXELS = 24_000_000


class ImageTooLarge(ValueError):
    pass


def load_image(file):
    """Open an upload the way every entry point must: size limit, EXIF rotation, RGB, at most 1920 px.

    Raises ImageTooLarge, or PIL.UnidentifiedImageError / OSError / Image.DecompressionBombError for unreadable files.
    """
    with Image.open(file) as opened:
        if opened.width * opened.height > MAX_PIXELS:
            raise ImageTooLarge("Please use an image below 24 megapixels.")
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((1920, 1920))
    return image


def correct_detections(detections, labels):
    corrected = []
    for item, label in zip(detections, labels):
        if label != "False detection":
            corrected.append(dict(item) | {"label": label, "model_label": item.get("model_label", item["label"]),
                                           "verified": label != "animal (unverified)", "verification_source": "Operator review"})
    return corrected


def update_detections(record_id, detections, status, reason, db_path=None):
    conn = connect(db_path)
    try:
        conn.execute("UPDATE observations SET detections = ?, status = ?, reason = ?, review = 'Unreviewed' WHERE id = ?",
                     (json.dumps(detections), status, reason, record_id))
        conn.commit()
    finally:
        conn.close()


def annotate(image, detections):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 260)
    for d in detections:
        box = tuple(d["box"])
        color = {"Animal": "#9be7b1", "Human": "#ffbf69", "Vehicle": "#ff877b", "Other": "#94bcff"}[category(d["label"])]
        draw.rectangle(box, outline=color, width=line)
        suffix = "?" if d.get("species_candidate") and category(d["label"]) == "Animal" and not d.get("verified") else ""
        label = f'{d["label"]}{suffix} (AI estimate)' if d.get("naming_model") and suffix else f'{d["label"]}{suffix}  {d["confidence"]:.0%}'
        x, y = box[:2]
        y = max(0, y - 20)
        text_box = draw.textbbox((x + 5, y + 3), label)
        draw.rectangle((x, y, text_box[2] + 5, text_box[3] + 3), fill=color)
        draw.text((x + 5, y + 3), label, fill="#10221b")
    return canvas



def connect(db_path=None):
    path = Path(db_path) if db_path else ROOT / "data" / "observations.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, mode TEXT NOT NULL,
        camera TEXT NOT NULL, lat REAL, lon REAL,
        source TEXT NOT NULL, restricted INTEGER NOT NULL, authorized INTEGER NOT NULL,
        threshold REAL NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
        detections TEXT NOT NULL, review TEXT NOT NULL DEFAULT 'Unreviewed')""")
    # Upgrade existing databases without losing their observations.
    if any(col[1] in {"lat", "lon"} and col[3] for col in conn.execute("PRAGMA table_info(observations)")):
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'observations'").fetchone()[0]
        schema = schema.replace("lat REAL NOT NULL", "lat REAL").replace("lon REAL NOT NULL", "lon REAL")
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            conn.execute("ALTER TABLE observations RENAME TO observations_legacy")
            conn.execute(schema)
            conn.execute("INSERT INTO observations SELECT * FROM observations_legacy")
            conn.execute("DROP TABLE observations_legacy")
    conn.commit()
    return conn


def save_observation(record, db_path=None):
    record = dict(record)
    record["id"] = uuid.uuid4().hex[:12]
    record["captured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["detections"] = json.dumps(record["detections"])
    keys = list(record)
    conn = connect(db_path)
    try:
        conn.execute(f"INSERT INTO observations ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(record.values()))
        conn.commit()
    finally:
        conn.close()
    return record["id"]


def history(mode=None, db_path=None):
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM observations WHERE mode = ? ORDER BY captured_at DESC, rowid DESC LIMIT 500", (mode,)).fetchall() if mode else conn.execute("SELECT * FROM observations ORDER BY captured_at DESC, rowid DESC LIMIT 500").fetchall()
        return [dict(r) | {"detections": json.loads(r["detections"])} for r in rows]
    finally:
        conn.close()


def update_review(record_id, review, db_path=None):
    if review not in {"Unreviewed", "Investigating", "Resolved", "False alarm"}:
        raise ValueError("Invalid review status")
    conn = connect(db_path)
    try:
        conn.execute("UPDATE observations SET review = ? WHERE id = ?", (review, record_id))
        conn.commit()
    finally:
        conn.close()
