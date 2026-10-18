# Backend

FastAPI service for VanDrishti. Phase 1 covers image ingestion and detection queries; streams, tracking, zones and alerts come in later phases.

## Run

Double-click **start-backend.cmd**, or run:

```powershell
.\.venv\Scripts\python.exe -m backend            # http://127.0.0.1:8000, interactive docs at /docs
.\.venv\Scripts\python.exe -m backend --port 8001 --data-dir D:\vandrishti-data
```

The API has no authentication yet: keep it on 127.0.0.1.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Service status, worker state, queue backlog, loaded models, observation counts |
| `POST /ingest/image` | Upload one image (multipart). Returns `202` with an observation id; analysis runs in the background |
| `GET /observations/{id}` | One observation: `state` (`queued`, `processing`, `done`, `failed`), settings, alert status and reason, detections |
| `GET /observations` | Recent observations; filter by `camera_id`, `state` |
| `GET /detections` | Detections from finished observations, newest first; filter by `since`, `until`, `category`, `label`, `camera_id`, `observation_id`, `min_confidence`; page with `limit` (max 1000) and `offset` |

`POST /ingest/image` form fields: `file` (JPG, PNG or WebP, up to 15 MB and 24 megapixels), `engine` (`Wildlife` default, or `Standard`), `threshold` (0.10–0.95, default 0.35), `camera_id`, `lat` and `lon` (both or neither), `restricted`, `authorized`.

Errors: `413` file or image too large, `415` unsupported file type, `422` unreadable image or invalid field, `503` queue full (64 waiting), `404` unknown observation.

Detection `category` is `person`, `vehicle`, `animal` or `other`. Times are UTC, recorded when the image was received.

```powershell
curl.exe -F "file=@samples/bus.jpg" -F "camera_id=CAM-03" -F "restricted=true" http://127.0.0.1:8000/ingest/image
curl.exe http://127.0.0.1:8000/observations/<id>
curl.exe "http://127.0.0.1:8000/detections?category=person&camera_id=CAM-03"
```

## Design

- **No inference in request handlers.** The handler validates the upload, stores the prepared image and queues the job. One background thread runs the models.
- **The database is the queue.** Jobs are rows in `observations`, so queued work survives a restart, and a job interrupted mid-inference is re-run.
- **Same pipeline as the dashboard.** Both use `core.load_image` (size limit, EXIF rotation, 1920 px), `detector.predict` and `core.analyze` (labelling policy and access rule). The prepared image is stored losslessly, so the API analyses exactly the pixels the dashboard would.
- **Storage:** `data/vandrishti.sqlite3` (WAL mode) with one row per detection for filtering, and prepared images in `data/images/`. Both are git-ignored. Stored images are kept until deleted.

## Files

`api.py` routes and validation · `worker.py` background inference · `store.py` SQLite · `smoke_test.py` starts the real server and sends a sample image (`python -m backend.smoke_test`).
