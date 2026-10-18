# VanRakshak

For the complete teammate handoff, running commands, architecture, current limitations and hackathon completion plan, read [CURRENT_PROTOTYPE.md](CURRENT_PROTOTYPE.md).

Local wildlife imagery analysis, camera mapping and operator-reviewed access alerts.

## Start

Double-click **start.cmd** and open http://127.0.0.1:8501. Keep the terminal open. Stop with Ctrl+C.

On another Windows laptop, install Python 3.12 or 3.13, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The ZIP includes the prepared wildlife detector. Setup installs dependencies and downloads the CLIP naming weights (about 338 MB). If prepared detector weights are missing, setup prepares them using the official YOLOE weights and a larger text encoder. Internet is needed for initial setup and map tiles. Inference thereafter runs locally without internet. The virtual environment is not portable; recreate it on another machine.

## API (backend)

The FastAPI service that will replace the dashboard: double-click **start-backend.cmd** and open http://127.0.0.1:8000/docs. It accepts image uploads, analyses them in the background with the same pipeline as the dashboard, and serves filtered detections. See [backend/README.md](backend/README.md).

## Recognition engines

**Wildlife (default):** YOLOE-11m finds objects at 960-pixel inference size. Prepared text embeddings allow it to search for 47 named wildlife/object targets. CLIP ViT-B/32 then compares animal crops against candidate names. A sufficiently separated match becomes an AI species estimate; weak distinctions remain `animal (unverified)`. No reserve-specific training has been performed.

**Standard:** the original YOLO11n model at 640 pixels. It is faster, has limited COCO categories and cannot identify gorillas directly. Animal species remain unverified in this engine.

The Wildlife engine improves species coverage, but is not uniformly more accurate on every image. In particular, a gorilla can still be confused with a bear or person. The small included comparison is evidence of behavior on those examples, not a field-accuracy benchmark. Consult `evaluation/RESULTS.md` before interpreting results.

## Analyze and correct

Upload a JPG, PNG or WebP image. Optionally enter actual camera coordinates. Mark a camera as restricted only when that setting is known. Click **Analyze image**.

Names followed by `?` are AI estimates. **Detection score** concerns the object detector; **Name similarity** is CLIP cosine similarity, not a probability or calibrated species confidence. Original detector labels, candidate names and scores remain available in the JSON report.

Expand **Review and correct detected objects** to confirm a species or remove a false detection. Saving updates the image, database, reports and alert. Operator corrections are distinguished from AI estimates and do not retrain either model. When a possible human conflicts with the naming model, it remains in access review instead of silently being removed.

## Activity and mapping

**Activity & alerts** displays local SQLite history, ranger review states and CSV export. Pins use only coordinates supplied by the operator, not GPS inferred from an image. Log timestamps are analysis times in UTC. Counts are per-frame detections, not unique tracked animals. PNG and JSON downloads are available with each analysis.

An access alert requires a detected person/vehicle, an operator-marked restricted location and no confirmed authorized visit. It cannot establish intent or criminal activity. No detections does not prove an area is safe. Lighting, fog, occlusion, small objects and aerial viewpoints can reduce recognition quality.

SQLite stores metadata and upload filenames in `data/observations.sqlite3`; image pixels stay in the session unless downloaded. This local app has no authentication: keep it on loopback.

## Checks

One command runs every check; a build phase is complete only when it exits with code 0:

```powershell
.\.venv\Scripts\python.exe run_checks.py                   # unit tests, real-model smoke test, evaluation set
.\.venv\Scripts\python.exe run_checks.py --eval --phase 0  # also measure both engines (several minutes)
```

## Evaluation set

`evaluation/eval_set/manifest.json` lists 300 held-out images with ground-truth boxes: 100 day (ENA24 camera traps, COCO), 100 night (ENA24 infrared camera traps, BIRDSAI thermal drone) and 100 aerial (WAID, VisDrone). `setup.ps1` downloads the images and checks their SHA-256; to do it by hand, run `python evaluation/eval_set.py fetch`. Images are not committed, and WAID and VisDrone images must not be redistributed.

`python evaluation/evaluate.py` scores each engine per condition and source: precision and recall for person, vehicle and animal at IoU 0.5, recall by object size, species accuracy, and category confusions. Results are saved in `evaluation/results/`. Targets for later phases are in `evaluation/targets.json`.

## Demo clips

`python demo/fetch_clips.py` prepares three clips in `demo/clips/`: a daytime elephant herd in Assam, a night thermal drone video with people and animals (with ground-truth tracks), and drone footage of a car in a forest. Attribution and licences: `demo/SOURCES.md`.

Sources: https://docs.ultralytics.com/models/yoloe/ and https://github.com/ultralytics/CLIP. Ultralytics licensing: https://www.ultralytics.com/license (AGPL-3.0 / enterprise options). Image attribution: `samples/SOURCES.md`.
