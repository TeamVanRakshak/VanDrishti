import io
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import core
from backend.api import create_app


def png(width=64, height=48, color="green"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeModels:
    """Stands in for detector.load_model/predict; predictions are chosen by image width."""

    def __init__(self, by_width=None, default=None):
        self.by_width, self.default = by_width or {}, default or []
        self.threads, self.gate, self.error = [], None, None

    def load_model(self, engine):
        return f"model-{engine}"

    def predict(self, model, image, threshold):
        self.threads.append(threading.current_thread().name)
        if self.gate:
            self.gate.wait(10)
        if self.error:
            raise self.error
        return [dict(d) for d in self.by_width.get(image.width, self.default)]


def wait_done(client, observation_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/observations/{observation_id}").json()
        if record["state"] in ("done", "failed"):
            return record
        time.sleep(.05)
    raise AssertionError(f"{observation_id} did not finish: {record}")


def ingest(client, data=None, name="frame.png", **fields):
    return client.post("/ingest/image", files={"file": (name, data if data is not None else png(), "image/png")},
                       data={k: str(v) for k, v in fields.items()})


BEAR = {"label": "bear", "confidence": .9, "box": [10, 10, 100, 100]}
PERSON = {"label": "person", "confidence": .8, "box": [120, 10, 200, 100]}
CAR = {"label": "car", "confidence": .6, "box": [5, 5, 50, 50]}


class Api(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def client(self, models, **options):
        return TestClient(create_app(self.data, models.load_model, models.predict, **options))

    def test_inference_runs_in_the_background_worker(self):
        models = FakeModels(default=[BEAR])
        models.gate = threading.Event()
        with self.client(models) as client:
            started = time.perf_counter()
            response = ingest(client)
            self.assertEqual(response.status_code, 202)
            self.assertLess(time.perf_counter() - started, 5)  # returned while the model was still blocked
            self.assertIn(client.get(response.json()["observation"]).json()["state"], ("queued", "processing"))
            models.gate.set()
            self.assertEqual(wait_done(client, response.json()["id"])["state"], "done")
        self.assertEqual(models.threads, ["inference-worker"])

    def test_results_match_the_shared_dashboard_pipeline(self):
        models = FakeModels(default=[BEAR, PERSON])
        with self.client(models) as client:
            record = wait_done(client, ingest(client, restricted=True, camera_id="CAM-01", lat=26.6, lon=93.3).json()["id"])
        detections, status, reason = core.analyze([BEAR, PERSON], True, False)
        self.assertEqual(record["detections"], detections)
        self.assertEqual((record["status"], record["reason"]), (status, reason))
        self.assertEqual(record["status"], "Review required")
        self.assertEqual(record["detections"][0]["label"], "animal (unverified)")
        self.assertEqual((record["camera_id"], record["lat"], record["lon"], record["restricted"]), ("CAM-01", 26.6, 93.3, True))
        self.assertEqual((record["width"], record["height"], record["source_name"]), (64, 48, "frame.png"))

    def test_detection_filters(self):
        models = FakeModels(by_width={64: [BEAR, PERSON], 65: [CAR], 66: [PERSON]})
        with self.client(models) as client:
            ids = [wait_done(client, ingest(client, png(w), camera_id=c).json()["id"])["id"]
                   for w, c in ((64, "CAM-01"), (65, "CAM-02"), (66, "CAM-02"))]
            get = lambda **q: client.get("/detections", params=q).json()
            self.assertEqual(len(get()), 4)
            self.assertEqual({d["observation_id"] for d in get(category="person")}, {ids[0], ids[2]})
            self.assertEqual([d["label"] for d in get(category="vehicle")], ["car"])
            self.assertEqual(len(get(category="animal")), 1)
            self.assertEqual(len(get(camera_id="CAM-02")), 2)
            self.assertEqual(len(get(observation_id=ids[0])), 2)
            self.assertEqual(len(get(label="person", camera_id="CAM-02")), 1)
            self.assertEqual([d["label"] for d in get(min_confidence=.85)], ["animal (unverified)"])
            now = datetime.now(timezone.utc)
            self.assertEqual(len(get(since=(now - timedelta(minutes=5)).isoformat())), 4)
            self.assertEqual(get(since=(now + timedelta(minutes=5)).isoformat()), [])
            self.assertEqual(get(until=(now - timedelta(minutes=5)).isoformat()), [])
            self.assertEqual(len(get(limit=2)) + len(get(limit=2, offset=2)), 4)
            self.assertEqual(get(observation_id=ids[2])[0]["box"], PERSON["box"])
            self.assertEqual(client.get("/detections", params={"category": "tiger"}).status_code, 422)
            self.assertEqual(client.get("/detections", params={"limit": 0}).status_code, 422)

    def test_records_persist_and_unfinished_work_resumes_after_restart(self):
        models = FakeModels(default=[PERSON])
        with self.client(models) as client:
            finished = wait_done(client, ingest(client).json()["id"])
        with self.client(models, start_worker=False) as client:  # service stops before this job runs
            waiting = ingest(client).json()["id"]
            self.assertEqual(client.get(f"/observations/{waiting}").json()["state"], "queued")
        interrupted = self.ingest_and_mark_processing(models)
        with self.client(models) as client:
            self.assertEqual(client.get(f"/observations/{finished['id']}").json(), finished)
            self.assertEqual(wait_done(client, waiting)["state"], "done")
            self.assertEqual(wait_done(client, interrupted)["state"], "done")
            self.assertEqual(len(client.get("/detections").json()), 3)

    def ingest_and_mark_processing(self, models):
        """Simulate a crash mid-inference: the job is left in the 'processing' state."""
        with self.client(models, start_worker=False) as client:
            observation_id = ingest(client).json()["id"]
        conn = sqlite3.connect(self.data / "vandrishti.sqlite3")
        with conn:
            conn.execute("UPDATE observations SET state = 'processing' WHERE id = ?", (observation_id,))
        conn.close()
        return observation_id

    def test_bad_uploads_are_rejected_with_clear_errors(self):
        models = FakeModels()
        big = io.BytesIO()
        Image.new("RGB", (5000, 5000)).save(big, format="PNG")  # 25 megapixels, small file
        cases = [
            ("not an image", dict(data=b"hello", name="frame.jpg"), 422),
            ("empty file", dict(data=b"", name="frame.jpg"), 422),
            ("wrong type", dict(data=png(), name="frame.gif"), 415),
            ("no extension", dict(data=png(), name="frame"), 415),
            ("over 15 MB", dict(data=b"\0" * (15 * 1024 * 1024 + 1), name="frame.png"), 413),
            ("over 24 megapixels", dict(data=big.getvalue(), name="frame.png"), 413),
            ("threshold out of range", dict(threshold=2), 422),
            ("unknown engine", dict(engine="Turbo"), 422),
            ("latitude without longitude", dict(lat=10), 422),
            ("latitude out of range", dict(lat=100, lon=10), 422),
        ]
        with self.client(models) as client:
            for description, fields, expected in cases:
                with self.subTest(description):
                    response = ingest(client, **fields)
                    self.assertEqual(response.status_code, expected, response.text)
                    self.assertIn("detail", response.json())
            self.assertEqual(client.post("/ingest/image").status_code, 422)  # no file at all
            self.assertEqual(client.get("/observations/missing").status_code, 404)
            self.assertEqual(client.get("/observations").json(), [])

    def test_inference_failure_is_recorded_and_service_keeps_working(self):
        models = FakeModels(default=[PERSON])
        models.error = RuntimeError("weights missing")
        with self.client(models) as client, self.assertLogs("vandrishti.worker", "ERROR"):
            failed = wait_done(client, ingest(client).json()["id"])
            self.assertEqual(failed["state"], "failed")
            self.assertIn("weights missing", failed["error"])
            models.error = None
            self.assertEqual(wait_done(client, ingest(client).json()["id"])["state"], "done")
            self.assertEqual(client.get("/health").json()["status"], "ok")

    def test_backlog_limit(self):
        with self.client(FakeModels(), start_worker=False, max_backlog=2) as client:
            self.assertEqual([ingest(client).status_code for _ in range(3)], [202, 202, 503])

    def test_health(self):
        with self.client(FakeModels(default=[CAR])) as client:
            wait_done(client, ingest(client, engine="Standard").json()["id"])
            health = client.get("/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["worker_running"])
        self.assertEqual((health["backlog"], health["models_loaded"], health["observations"]), (0, ["Standard"], {"done": 1}))


if __name__ == "__main__":
    unittest.main()
