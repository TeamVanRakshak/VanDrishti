"""Real models through the API: expected labels, and identical results to the dashboard's pipeline."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import core
import detector
from backend.api import create_app
from test_backend import wait_done

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
EXPECTED = {"bus.jpg": {"person", "bus"}, "giraffes.jpg": {"giraffe"}, "zebra.jpg": {"zebra"}, "gorilla.jpg": {"gorilla"}}


class RealModelsThroughApi(unittest.TestCase):
    def test_sample_images(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory)) as client:
            for engine in ("Wildlife", "Standard"):
                for name, expected in EXPECTED.items():
                    with self.subTest(engine=engine, image=name):
                        with open(SAMPLES / name, "rb") as handle:
                            response = client.post("/ingest/image", files={"file": (name, handle, "image/jpeg")},
                                                   data={"engine": engine, "restricted": "true"})
                        self.assertEqual(response.status_code, 202, response.text)
                        record = wait_done(client, response.json()["id"], timeout=600)
                        self.assertEqual(record["state"], "done", record.get("error"))
                        if engine == "Wildlife":  # the same expectations as smoke_test.py
                            self.assertLessEqual(expected, {d["label"] for d in record["detections"]})
                        # The dashboard runs exactly this: load_image -> predict -> analyze.
                        model = client.app.state.worker.models[engine]
                        detections, status, reason = core.analyze(detector.predict(model, core.load_image(SAMPLES / name), .35), True, False)
                        self.assertEqual(record["detections"], detections)
                        self.assertEqual((record["status"], record["reason"]), (status, reason))
            self.assertEqual(len(client.get("/detections", params={"label": "bus"}).json()), 2)


if __name__ == "__main__":
    unittest.main()
