"""VanDrishti HTTP API (Phase 1: image ingestion and detection queries)."""
import io
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError

import detector
from backend.store import Store
from backend.worker import Worker
from core import ROOT, ImageTooLarge, load_image

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # same limit as the dashboard (.streamlit/config.toml)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class Engine(str, Enum):
    wildlife = "Wildlife"
    standard = "Standard"


class Category(str, Enum):
    person = "person"
    vehicle = "vehicle"
    animal = "animal"
    other = "other"


def create_app(data_dir=ROOT / "data", load_model=None, predict=None, max_backlog=64, start_worker=True):
    """Build the app. Tests pass their own data directory and fake model functions."""
    data_dir = Path(data_dir)
    images = data_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "vandrishti.sqlite3")
    worker = Worker(store, load_model or detector.load_model, predict or detector.predict)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="VanDrishti API", version="0.1.0", lifespan=lifespan,
                  description="Wildlife monitoring and threat detection. Detections are AI estimates for ranger review.")
    app.state.store, app.state.worker = store, worker

    @app.get("/health")
    def health():
        return {"status": "ok" if worker.alive or not start_worker else "degraded", "worker_running": worker.alive,
                "backlog": store.queued(), "models_loaded": sorted(worker.models), "observations": store.counts()}

    @app.post("/ingest/image", status_code=status.HTTP_202_ACCEPTED)
    def ingest_image(
            file: Annotated[UploadFile, File(description="JPG, PNG or WebP, up to 15 MB and 24 megapixels")],
            engine: Annotated[Engine, Form()] = Engine.wildlife,
            threshold: Annotated[float, Form(ge=.1, le=.95)] = .35,
            camera_id: Annotated[str | None, Form(max_length=100)] = None,
            lat: Annotated[float | None, Form(ge=-90, le=90)] = None,
            lon: Annotated[float | None, Form(ge=-180, le=180)] = None,
            restricted: Annotated[bool, Form()] = False,
            authorized: Annotated[bool, Form(description="Operator-confirmed patrol or service visit")] = False):
        if (lat is None) != (lon is None):
            raise HTTPException(422, "Provide both lat and lon, or neither.")
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in EXTENSIONS:
            raise HTTPException(415, "Upload a JPG, PNG or WebP image.")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 15 MB.")
        if not data:
            raise HTTPException(422, "The uploaded file is empty.")
        try:
            image = load_image(io.BytesIO(data))
        except ImageTooLarge as error:
            raise HTTPException(413, str(error))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, "This file could not be read as a supported image.")
        if store.queued() >= max_backlog:
            raise HTTPException(503, "The analysis queue is full. Try again shortly.")
        observation_id = uuid.uuid4().hex[:12]
        path = images / f"{observation_id}.png"
        image.save(path)  # lossless, so the worker analyses exactly the pixels the dashboard would
        store.create(path, engine=engine.value, threshold=threshold, camera_id=camera_id, lat=lat, lon=lon,
                     restricted=restricted, authorized=authorized, source_name=name, width=image.width,
                     height=image.height, observation_id=observation_id)
        worker.notify()
        return {"id": observation_id, "state": "queued", "observation": f"/observations/{observation_id}"}

    @app.get("/observations/{observation_id}")
    def get_observation(observation_id: str):
        record = store.observation(observation_id)
        if record is None:
            raise HTTPException(404, "Observation not found.")
        return record

    @app.get("/observations")
    def list_observations(camera_id: str | None = None, state: str | None = None,
                          limit: Annotated[int, Query(ge=1, le=500)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
        return store.observations(camera_id, state, limit, offset)

    @app.get("/detections")
    def list_detections(since: datetime | None = None, until: datetime | None = None, category: Category | None = None,
                        label: str | None = None, camera_id: str | None = None, observation_id: str | None = None,
                        min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
                        limit: Annotated[int, Query(ge=1, le=1000)] = 100, offset: Annotated[int, Query(ge=0)] = 0):
        """Detections from finished observations, newest first. Times are when the image was received (UTC)."""
        return store.detections(since, until, category.value if category else None, label, camera_id, observation_id,
                                min_confidence, limit, offset)

    return app
