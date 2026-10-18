"""SQLite storage for observations (one per ingested image) and their detections (one row per object)."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core import category

CATEGORY = {"Human": "person", "Vehicle": "vehicle", "Animal": "animal", "Other": "other"}
OBSERVATION_FIELDS = ("id", "received_at", "completed_at", "state", "error", "engine", "threshold", "camera_id", "lat", "lon",
                      "restricted", "authorized", "source_name", "width", "height", "seconds", "status", "reason")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def timestamp(value):
    """Normalise a datetime to the stored UTC string format; naive values are treated as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY, received_at TEXT NOT NULL, completed_at TEXT,
                    state TEXT NOT NULL, error TEXT, engine TEXT NOT NULL, threshold REAL NOT NULL,
                    camera_id TEXT, lat REAL, lon REAL, restricted INTEGER NOT NULL, authorized INTEGER NOT NULL,
                    source_name TEXT NOT NULL, image_path TEXT NOT NULL, width INTEGER, height INTEGER,
                    seconds REAL, status TEXT, reason TEXT);
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id TEXT NOT NULL REFERENCES observations(id),
                    received_at TEXT NOT NULL, camera_id TEXT, label TEXT NOT NULL, category TEXT NOT NULL,
                    confidence REAL NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observations_state ON observations(state);
                CREATE INDEX IF NOT EXISTS detections_time ON detections(received_at);
                CREATE INDEX IF NOT EXISTS detections_filter ON detections(category, label, camera_id);
                CREATE INDEX IF NOT EXISTS detections_observation ON detections(observation_id);""")

    @contextmanager
    def connect(self):
        """One short-lived connection per operation: commit on success, always close."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, image_path, *, engine, threshold, camera_id, lat, lon, restricted, authorized, source_name, width, height, observation_id=None):
        observation_id = observation_id or uuid.uuid4().hex[:12]
        with self.connect() as conn:
            conn.execute("""INSERT INTO observations (id, received_at, state, engine, threshold, camera_id, lat, lon, restricted,
                            authorized, source_name, image_path, width, height) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (observation_id, now(), "queued", engine, threshold, camera_id, lat, lon, int(restricted),
                          int(authorized), source_name, str(image_path), width, height))
        return observation_id

    def requeue_interrupted(self):
        """Jobs that were mid-inference when the service stopped go back to the queue."""
        with self.connect() as conn:
            return conn.execute("UPDATE observations SET state = 'queued' WHERE state = 'processing'").rowcount

    def claim_next(self):
        """Atomically take the oldest queued job; the database is the queue, so nothing is lost on restart."""
        with self.connect() as conn:
            row = conn.execute("""UPDATE observations SET state = 'processing' WHERE id = (
                                      SELECT id FROM observations WHERE state = 'queued' ORDER BY received_at, rowid LIMIT 1)
                                  RETURNING *""").fetchone()
        return row

    def queued(self):
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM observations WHERE state IN ('queued', 'processing')").fetchone()[0]

    def complete(self, observation_id, detections, status, reason, seconds):
        with self.connect() as conn:
            row = conn.execute("SELECT received_at, camera_id FROM observations WHERE id = ?", (observation_id,)).fetchone()
            conn.execute("DELETE FROM detections WHERE observation_id = ?", (observation_id,))  # a resumed job may have partial rows
            conn.executemany("""INSERT INTO detections (observation_id, received_at, camera_id, label, category, confidence,
                                x1, y1, x2, y2, data) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                             [(observation_id, row["received_at"], row["camera_id"], d["label"], CATEGORY[category(d["label"])],
                               d["confidence"], *d["box"], json.dumps(d)) for d in detections])
            conn.execute("UPDATE observations SET state = 'done', completed_at = ?, status = ?, reason = ?, seconds = ?, error = NULL WHERE id = ?",
                         (now(), status, reason, round(seconds, 3), observation_id))

    def fail(self, observation_id, error):
        with self.connect() as conn:
            conn.execute("UPDATE observations SET state = 'failed', completed_at = ?, error = ? WHERE id = ?", (now(), error, observation_id))

    def observation(self, observation_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM observations WHERE id = ?", (observation_id,)).fetchone()
            if row is None:
                return None
            detections = conn.execute("SELECT data FROM detections WHERE observation_id = ? ORDER BY id", (observation_id,)).fetchall()
        return self._observation(row) | {"detections": [json.loads(d["data"]) for d in detections]}

    def observations(self, camera_id=None, state=None, limit=50, offset=0):
        sql, args = "SELECT * FROM observations WHERE 1=1", []
        if camera_id:
            sql, args = sql + " AND camera_id = ?", args + [camera_id]
        if state:
            sql, args = sql + " AND state = ?", args + [state]
        with self.connect() as conn:
            rows = conn.execute(sql + " ORDER BY received_at DESC, rowid DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return [self._observation(r) for r in rows]

    def detections(self, since=None, until=None, category=None, label=None, camera_id=None, observation_id=None,
                   min_confidence=None, limit=100, offset=0):
        sql, args = "SELECT * FROM detections WHERE 1=1", []
        for clause, value in (("received_at >= ?", since and timestamp(since)), ("received_at < ?", until and timestamp(until)),
                              ("category = ?", category), ("label = ?", label), ("camera_id = ?", camera_id),
                              ("observation_id = ?", observation_id), ("confidence >= ?", min_confidence)):
            if value is not None:
                sql, args = f"{sql} AND {clause}", args + [value]
        with self.connect() as conn:
            rows = conn.execute(sql + " ORDER BY received_at DESC, id LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return [{"id": r["id"], "observation_id": r["observation_id"], "received_at": r["received_at"], "camera_id": r["camera_id"],
                 "label": r["label"], "category": r["category"], "confidence": r["confidence"],
                 "box": [r["x1"], r["y1"], r["x2"], r["y2"]], "details": json.loads(r["data"])} for r in rows]

    def counts(self):
        with self.connect() as conn:
            return {r["state"]: r["n"] for r in conn.execute("SELECT state, COUNT(*) AS n FROM observations GROUP BY state")}

    @staticmethod
    def _observation(row):
        record = {key: row[key] for key in OBSERVATION_FIELDS}
        record["restricted"], record["authorized"] = bool(record["restricted"]), bool(record["authorized"])
        return record
