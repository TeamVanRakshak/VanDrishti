"""Start the real API server, send a sample image over HTTP and check the result.  python -m backend.smoke_test"""
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx2 as httpx

ROOT = Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as data:
        server = subprocess.Popen([sys.executable, "-m", "backend", "--port", str(port), "--data-dir", data], cwd=ROOT)
        try:
            deadline = time.time() + 120
            while True:
                try:
                    if httpx.get(f"{base}/health").json()["status"] == "ok":
                        break
                except httpx.TransportError:
                    pass
                if time.time() > deadline or server.poll() is not None:
                    sys.exit("API server did not become healthy")
                time.sleep(.5)
            with open(ROOT / "samples" / "bus.jpg", "rb") as handle:
                response = httpx.post(f"{base}/ingest/image", files={"file": ("bus.jpg", handle, "image/jpeg")},
                                      data={"camera_id": "CAM-03", "restricted": "true"})
            assert response.status_code == 202, response.text
            observation = response.json()["observation"]
            deadline = time.time() + 300
            while (record := httpx.get(base + observation).json())["state"] not in ("done", "failed"):
                assert time.time() < deadline, "analysis did not finish"
                time.sleep(.5)
            labels = {d["label"] for d in record["detections"]}
            assert record["state"] == "done" and {"person", "bus"} <= labels, record
            assert record["status"] == "Review required", record["status"]
            people = httpx.get(f"{base}/detections", params={"category": "person", "camera_id": "CAM-03"}).json()
            assert people and all(d["category"] == "person" for d in people), people
            assert httpx.get(f"{base}/docs").status_code == 200
            print(f"API SMOKE TEST PASSED: {len(record['detections'])} objects ({sorted(labels)}), "
                  f"{len(people)} people via /detections, {record['seconds']} s inference")
        finally:
            server.terminate()
            server.wait(30)


if __name__ == "__main__":
    main()
