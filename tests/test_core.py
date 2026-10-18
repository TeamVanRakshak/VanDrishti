import tempfile
import unittest
import sqlite3
from pathlib import Path

from PIL import Image
from core import annotate, assess, category, history, save_observation, update_review, verification_labels, correct_detections, update_detections, connect


class RulesAndStorage(unittest.TestCase):
    def test_legacy_database_preserved_when_location_becomes_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            new = connect(Path(directory) / "new.sqlite3")
            schema = new.execute("SELECT sql FROM sqlite_master WHERE name = 'observations'").fetchone()[0]
            new.close()
            db = Path(directory) / "legacy.sqlite3"
            legacy = sqlite3.connect(db)
            legacy.execute(schema.replace("lat REAL", "lat REAL NOT NULL").replace("lon REAL", "lon REAL NOT NULL"))
            legacy.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("retained", "2026-10-18", "YOLO detection", "Camera", 1., 2., "old.jpg", 0, 0, .35, "No detections", "test", "[]", "Unreviewed"))
            legacy.commit()
            legacy.close()
            self.assertEqual(history(db_path=db)[0]["id"], "retained")
            record = dict(mode="YOLO detection", camera="Location not assigned", lat=None, lon=None, source="new.jpg", restricted=0, authorized=0, threshold=.35, status="No detections", reason="test", detections=[])
            save_observation(record, db)
            self.assertEqual(len(history(db_path=db)), 2)
    def test_context_changes_intrusion_outcome(self):
        objects = [{"label": "person"}, {"label": "car"}]
        self.assertEqual(assess(objects, True, False)[0], "Review required")
        self.assertEqual(assess(objects, True, True)[0], "Authorized activity")
        self.assertEqual(assess(objects, False, False)[0], "Activity observed")
        self.assertEqual(assess([{"label": "elephant"}], True, False)[0], "No rule triggered")
        self.assertEqual(assess([], True, False)[0], "No detections")
        self.assertEqual(category("deer"), "Animal")

    def test_species_suggestions_require_verification(self):
        image = Image.new("RGB", (100, 100))
        detections = verification_labels([{"label": "bear", "confidence": .99, "box": [1, 1, 90, 90]}])
        self.assertEqual(detections[0]["label"], "animal (unverified)")
        self.assertEqual(detections[0]["model_label"], "bear")
        corrected = correct_detections(detections, ["gorilla"])
        self.assertEqual(verification_labels(corrected)[0]["label"], "gorilla")
        self.assertTrue(corrected[0]["verified"])
        self.assertEqual(correct_detections(detections, ["False detection"]), [])
        self.assertEqual(annotate(image, detections).size, image.size)

    def test_storage_roundtrip_separates_modes_and_saves_review(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "test.sqlite3"
            record = dict(mode="YOLO detection", camera="Test", lat=26.6, lon=93.3, source="frame.jpg",
                          restricted=1, authorized=0, threshold=.35, status="Review required", reason="test",
                          detections=[{"label": "person", "confidence": .8, "box": [1, 1, 90, 90]}])
            key = save_observation(record, db)
            self.assertEqual(len(history(db_path=db)), 1)
            update_review(key, "Investigating", db)
            self.assertEqual(history("YOLO detection", db)[0]["review"], "Investigating")
            update_detections(key, [], "No detections", "False detection removed", db)
            self.assertEqual(history(db_path=db)[0]["detections"], [])
            self.assertEqual(history(db_path=db)[0]["status"], "No detections")
            with self.assertRaises(ValueError):
                update_review(key, "bad status", db)


if __name__ == "__main__":
    unittest.main()
