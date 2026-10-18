"""Download and prepare the three demo clips in demo/clips/ (see demo/SOURCES.md for licences).

    python demo/fetch_clips.py

Clips are re-encoded to at most 1280x720 MP4 so every team laptop can replay them as camera feeds.
"""
import csv
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / "clips"
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/transcoded/"
VIDEOS = {
    "day_elephant_herd.mp4": COMMONS + "f/f1/A_herd_of_elephant_of_Hoollongapar_Gibbon_Sanctuary%2C_Jorhat_-_Assam.webm/"
                                      "A_herd_of_elephant_of_Hoollongapar_Gibbon_Sanctuary%2C_Jorhat_-_Assam.webm.1080p.vp9.webm",
    "drone_vehicle_forest.mp4": COMMONS + "e/ea/Aerial_views_of_a_car_driving_in_snow_through_a_forest.webm/"
                                          "Aerial_views_of_a_car_driving_in_snow_through_a_forest.webm.1080p.vp9.webm",
}
BIRDSAI = "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/TrainReal/"
THERMAL_SEQUENCE = "0000000067_0000000059"  # two people and six animals, from the training split (not the evaluation set)
THERMAL_FPS = 25  # BIRDSAI does not document the original frame rate


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VanRakshak-demo"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def writer(path, fps, width, height):
    scale = min(1., 1280 / width, 720 / height)
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size), size


def convert(source, target):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        sys.exit(f"OpenCV cannot read {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    out, size = writer(target, fps, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        frames += 1
    capture.release()
    out.release()
    return frames, fps


def thermal(target):
    listing = json.loads(download("https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix=conservationdrones/v01/"
                                  f"TrainReal/images/{THERMAL_SEQUENCE}/&fields=items(name)&maxResults=5000"))
    names = sorted(item["name"] for item in listing["items"] if item["name"].endswith(".jpg"))
    with ThreadPoolExecutor(16) as pool:
        images = list(pool.map(lambda n: download("https://storage.googleapis.com/public-datasets-lila/" + n), names))
    first = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_COLOR)
    out, size = writer(target, THERMAL_FPS, first.shape[1], first.shape[0])
    for data in images:
        out.write(cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), size))
    out.release()
    # Ground-truth tracks (MOT format) for later tracking tests.
    rows = [r for r in csv.reader(io.StringIO(download(BIRDSAI + f"annotations/{THERMAL_SEQUENCE}.csv").decode())) if len(r) >= 8]
    with open(target.with_suffix(".csv"), "w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return len(images), THERMAL_FPS


def main():
    CLIPS.mkdir(exist_ok=True)
    for name, url in VIDEOS.items():
        target = CLIPS / name
        if not target.exists():
            source = CLIPS / (name + ".download.webm")
            source.write_bytes(download(url))
            frames, fps = convert(source, target)
            source.unlink()
            print(f"{name}: {frames} frames at {fps:.1f} fps")
    target = CLIPS / "night_thermal_people.mp4"
    if not target.exists():
        frames, fps = thermal(target)
        print(f"{target.name}: {frames} frames at {fps} fps, ground truth in {target.with_suffix('.csv').name}")
    for clip in sorted(CLIPS.glob("*.mp4")):
        capture = cv2.VideoCapture(str(clip))
        frames, ok = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.read()[0]
        capture.release()
        if not ok or frames < 100:
            sys.exit(f"{clip.name} is not a playable clip")
        print(f"OK {clip.name}: {frames} frames")


if __name__ == "__main__":
    main()
