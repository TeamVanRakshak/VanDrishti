"""Background inference: requests only queue work; this thread runs the models.

The observations table is the queue, so queued work survives a restart.
"""
import logging
import threading
import time

from PIL import Image

from core import analyze

log = logging.getLogger("vandrishti.worker")


class Worker:
    def __init__(self, store, load_model, predict, idle_poll=2.0):
        self.store, self.load_model, self.predict = store, load_model, predict
        self.idle_poll = idle_poll
        self.wake, self.stopping = threading.Event(), threading.Event()
        self.models = {}
        self.thread = None
        self.processed = 0

    def start(self):
        self.store.requeue_interrupted()
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, name="inference-worker", daemon=True)
        self.thread.start()

    def stop(self, timeout=60):
        """Finish the current job, then stop. Remaining queued jobs stay in the database."""
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout)

    def notify(self):
        self.wake.set()

    @property
    def alive(self):
        return bool(self.thread and self.thread.is_alive())

    def model(self, engine):
        if engine not in self.models:
            self.models[engine] = self.load_model(engine)
        return self.models[engine]

    def _run(self):
        while not self.stopping.is_set():
            try:
                job = self.store.claim_next()
                if job is None:
                    self.wake.wait(self.idle_poll)
                    self.wake.clear()
                    continue
                self.process(job)
            except Exception:  # never let one job (or a locked database) kill the worker
                log.exception("Worker loop error")
                time.sleep(self.idle_poll)

    def process(self, job):
        try:
            with Image.open(job["image_path"]) as opened:
                image = opened.convert("RGB")
            model = self.model(job["engine"])  # first use loads the weights; not counted as inference time
            start = time.perf_counter()
            raw = self.predict(model, image, job["threshold"])
            detections, status, reason = analyze(raw, bool(job["restricted"]), bool(job["authorized"]))
            self.store.complete(job["id"], detections, status, reason, time.perf_counter() - start)
        except Exception as error:
            log.exception("Inference failed for %s", job["id"])
            self.store.fail(job["id"], f"{type(error).__name__}: {error}")
        self.processed += 1
