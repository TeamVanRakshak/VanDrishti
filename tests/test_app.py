import tempfile
import unittest
import io
from PIL import Image
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import core


class DashboardWorkflow(unittest.TestCase):
    def test_analysis_corrections_and_alert_context(self):
        with tempfile.TemporaryDirectory() as directory:
            original_connect = core.connect
            predictions = [{"label": "bear", "confidence": .9, "box": [10, 10, 100, 100]},
                           {"label": "person", "confidence": .8, "box": [120, 10, 200, 100]}]
            upload = io.BytesIO()
            Image.new("RGB", (300, 300), "green").save(upload, format="PNG")
            upload.name = "forest.png"
            with patch.object(core, "connect", side_effect=lambda db_path=None: original_connect(Path(directory) / "app.sqlite3")), patch("detector.load_model", return_value=object()), patch("detector.predict", return_value=predictions), patch("streamlit.file_uploader", return_value=upload):
                app = AppTest.from_file("app.py", default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.tabs), 2)
                self.assertFalse(next(t for t in app.toggle if t.label == "Restricted location").value)
                next(t for t in app.toggle if t.label == "Restricted location").set_value(True).run()
                next(b for b in app.button if b.label == "Analyze image →").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Potential intrusion" in e.value for e in app.error))
                self.assertEqual(len(core.history()), 1)
                self.assertEqual(core.history()[0]["detections"][0]["label"], "animal (unverified)")
                next(s for s in app.selectbox if s.label.startswith("Object 1")).select("gorilla")
                next(s for s in app.selectbox if s.label.startswith("Object 2")).select("False detection")
                next(b for b in app.button if b.label == "Save corrected labels").click().run()
                self.assertEqual(len(app.exception), 0)
                saved = core.history()[0]
                self.assertEqual(saved["detections"][0]["label"], "gorilla")
                self.assertEqual(len(saved["detections"]), 1)
                self.assertEqual(saved["status"], "No rule triggered")
                next(t for t in app.toggle if t.label == "Known authorized visit").set_value(True).run()
                next(b for b in app.button if b.label == "Analyze image →").click().run()
                self.assertTrue(any("Authorized activity" in e.value for e in app.success))
                self.assertEqual(len(core.history()), 2)


if __name__ == "__main__":
    unittest.main()
