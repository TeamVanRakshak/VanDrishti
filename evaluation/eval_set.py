"""Held-out evaluation set: 100 day, 100 night and 100 aerial images with ground-truth boxes.

    python evaluation/eval_set.py fetch    # everyone: download the images in manifest.json and verify them
    python evaluation/eval_set.py verify   # check every image is present and unchanged
    python evaluation/eval_set.py select   # maintainers only: rebuild manifest.json from the public sources

Ground truth uses three categories (person, vehicle, animal). Animal boxes also carry a species
name when the source species is in our vocabulary; otherwise species is null.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SET = ROOT / "eval_set"
IMAGES = SET / "images"
MANIFEST = SET / "manifest.json"
CACHE = ROOT / ".cache"
SEED = 2026
QUOTAS = {"day": {"ena24": 60, "coco": 40}, "night": {"ena24": 50, "birdsai": 50}, "aerial": {"waid": 50, "visdrone": 50}}

LILA = "https://storage.googleapis.com/public-datasets-lila/"
BIRDSAI = LILA + "conservationdrones/v01/conservationdrones-testset/TestReal/"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VISDRONE_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
WAID_RAW = "https://raw.githubusercontent.com/xiaohuicui/WAID/main/"
SOURCES = {
    "ena24": {"name": "ENA24-detection (LILA BC)", "licence": "CDLA-Permissive-1.0", "url": "https://lila.science/datasets/ena24detection"},
    "coco": {"name": "COCO val2017", "licence": "Annotations CC BY 4.0; images keep their Flickr licences (recorded per image)", "url": "https://cocodataset.org"},
    "birdsai": {"name": "BIRDSAI / Conservation Drones TestReal (LILA BC)", "licence": "CDLA-Permissive-1.0", "url": "https://lila.science/datasets/conservationdrones"},
    "waid": {"name": "WAID test split", "licence": "No licence file in the repository; Roboflow copy listed as CC BY 4.0. Do not redistribute images", "url": "https://github.com/xiaohuicui/WAID"},
    "visdrone": {"name": "VisDrone2019-DET val (Ultralytics mirror)", "licence": "Academic research use; do not redistribute images", "url": "https://github.com/VisDrone/VisDrone-Dataset"},
}

ENA24_SPECIES = {"American Black Bear": "bear", "White_Tailed_Deer": "deer", "Red Fox": "fox", "Grey Fox": "fox",
                 "Dog": "dog", "Domestic Cat": "cat", "Horse": "horse", "Chicken": "bird", "Wild Turkey": "bird",
                 "American Crow": "bird", "Bird": "bird"}
COCO_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck", "boat", "airplane", "train"}
COCO_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
BIRDSAI_SPECIES = {1: "elephant", 2: "lion", 3: "giraffe", 4: "dog", 5: "crocodile", 6: "hippopotamus", 7: "zebra", 8: "rhinoceros"}
WAID_SPECIES = {0: "sheep", 1: "cow", 2: None, 3: None, 4: None, 5: "zebra"}  # sheep, cattle, seal, camelus, kiang, zebra


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VanRakshak-eval"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"Download failed: {url} ({error})") from error


def download_all(urls):
    """Download in parallel, caching files so an interrupted selection can resume."""
    folder = CACHE / "files"
    folder.mkdir(parents=True, exist_ok=True)

    def one(url):
        path = folder / urllib.parse.quote(url.split("://", 1)[1], safe="")[-150:]
        if not path.exists():
            path.write_bytes(download(url))
        return url, path.read_bytes()
    with ThreadPoolExecutor(16) as pool:
        return dict(pool.map(one, urls))


def cached(url):
    path = CACHE / url.rsplit("/", 1)[1]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...", flush=True)
        path.with_suffix(".part").write_bytes(download(url))
        path.with_suffix(".part").rename(path)
    return path


def image_bytes(entry):
    if "zip" in entry:
        with zipfile.ZipFile(cached(entry["zip"])) as archive:
            return archive.read(entry["member"])
    return download(entry["url"])


def box(category, x1, y1, x2, y2, width, height, species=None):
    x1, y1, x2, y2 = max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return {"category": category, "species": species, "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}


def keep(entry, data, selected):
    from PIL import Image
    entry["width"], entry["height"] = Image.open(io.BytesIO(data)).size
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    IMAGES.mkdir(parents=True, exist_ok=True)
    (IMAGES / f'{entry["id"]}.jpg').write_bytes(data)
    selected.append(entry)


def is_infrared(data):
    """Camera traps shoot near-infrared at night, giving almost identical colour channels."""
    import numpy as np
    from PIL import Image
    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((160, 90)), dtype=float)
    return float(np.abs(pixels[..., 0] - pixels[..., 1]).mean() + np.abs(pixels[..., 1] - pixels[..., 2]).mean()) < 3


def select_ena24(rng, selected):
    data = json.loads(cached(LILA + "ena24/ena24_public.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    by_image = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    images = [i for i in data["images"] if i["id"] in by_image]
    rng.shuffle(images)
    vehicle = [i for i in images if any(names[a["category_id"]] == "Vehicle" for a in by_image[i["id"]])]
    others = [i for i in images if i not in vehicle]
    need = {("day", True): 15, ("day", False): 45, ("night", True): 5, ("night", False): 45}
    for has_vehicle, pool in ((True, vehicle), (False, others)):
        for start in range(0, len(pool), 48):
            if not need[("day", has_vehicle)] and not need[("night", has_vehicle)]:
                break
            batch = pool[start:start + 48]
            fetched = download_all([LILA + "ena24/images/" + i["file_name"] for i in batch])
            for info in batch:
                url = LILA + "ena24/images/" + info["file_name"]
                condition = "night" if is_infrared(fetched[url]) else "day"
                if not need[(condition, has_vehicle)]:
                    continue
                need[(condition, has_vehicle)] -= 1
                boxes = []
                for a in by_image[info["id"]]:
                    name = names[a["category_id"]]
                    x, y, bw, bh = a["bbox"]
                    boxes.append(box("vehicle" if name == "Vehicle" else "animal", x, y, x + bw, y + bh,
                                     info["width"], info["height"], ENA24_SPECIES.get(name)))
                entry = {"id": f'ena24-{info["id"]}', "condition": condition, "source": "ena24", "url": url,
                         "boxes": [b for b in boxes if b], "ignore": []}
                keep(entry, fetched[url], selected)
        if has_vehicle:
            # Vehicles are rare, especially at night; give any shortfall to animal images.
            for condition in ("day", "night"):
                need[(condition, False)] += need[(condition, True)]
                need[(condition, True)] = 0
    if any(need.values()):
        raise RuntimeError(f"ENA24 ran out of images; still needed {need}")


def select_coco(rng, selected):
    with zipfile.ZipFile(cached(COCO_ANNOTATIONS)) as archive:
        data = json.loads(archive.read("annotations/instances_val2017.json"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    licences = {l["id"]: l["name"] for l in data["licenses"]}
    by_image = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    images = sorted(data["images"], key=lambda i: i["id"])
    rng.shuffle(images)

    def has(info, wanted):
        return any(not a["iscrowd"] and names[a["category_id"]] in wanted for a in by_image.get(info["id"], []))
    chosen = [i for i in images if has(i, COCO_VEHICLES)][:20]
    chosen += [i for i in images if has(i, {"person"}) and i not in chosen][:20]
    fetched = download_all([i["coco_url"] for i in chosen])
    for info in chosen:
        boxes, ignore = [], []
        for a in by_image[info["id"]]:
            name = names[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            category = "person" if name == "person" else "vehicle" if name in COCO_VEHICLES else "animal" if name in COCO_ANIMALS else None
            if category and a["iscrowd"]:
                ignore.append([x, y, x + bw, y + bh])
            elif category:
                boxes.append(box(category, x, y, x + bw, y + bh, info["width"], info["height"], name if category == "animal" else None))
        entry = {"id": f'coco-{info["id"]}', "condition": "day", "source": "coco", "url": info["coco_url"],
                 "image_licence": licences[info["license"]], "boxes": [b for b in boxes if b], "ignore": ignore}
        keep(entry, fetched[info["coco_url"]], selected)


def select_birdsai(rng, selected):
    listing = json.loads(download("https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix="
                                  "conservationdrones/v01/conservationdrones-testset/TestReal/annotations/&delimiter=/&fields=items(name)"))
    folders = json.loads(download("https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix="
                                  "conservationdrones/v01/conservationdrones-testset/TestReal/images/&delimiter=/&fields=prefixes"))
    with_images = {p.rstrip("/").rsplit("/", 1)[1] for p in folders["prefixes"]}
    # Some annotation files have no image folder; skip those sequences.
    sequences = [item["name"].rsplit("/", 1)[1][:-4] for item in listing["items"] if item["name"].endswith(".csv")]
    sequences = [s for s in sequences if s in with_images]
    annotations = download_all([BIRDSAI + f"annotations/{s}.csv" for s in sequences])
    candidates = {"person": [], "animal": []}
    for sequence in sequences:
        existing = set()  # image files are named by video frame number and have gaps
        listing_url = ("https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix=conservationdrones/v01/"
                       f"conservationdrones-testset/TestReal/images/{sequence}/&fields=items(name),nextPageToken")
        page = json.loads(download(listing_url))
        while True:
            existing |= {int(item["name"][-14:-4]) for item in page.get("items", []) if item["name"].endswith(".jpg")}
            if "nextPageToken" not in page:
                break
            page = json.loads(download(listing_url + "&pageToken=" + urllib.parse.quote(page["nextPageToken"])))
        frames = {}
        for row in csv.reader(io.StringIO(annotations[BIRDSAI + f"annotations/{sequence}.csv"].decode())):
            if len(row) < 8:  # a few files contain truncated lines
                continue
            frame, obj, x, y, w, h, cls, species = (int(v) for v in row[:8])
            frames.setdefault(frame, {})[obj] = (x, y, w, h, cls, species)
        for frame in sorted(f for f in frames if f in existing)[::10]:
            # BIRDSAI frame numbers may be 0- or 1-based depending on the sequence. Only keep frames whose
            # boxes barely change in the next frame, so the labels are right under either convention.
            now, after = frames[frame], frames.get(frame + 1, {})
            if set(now) != set(after) or any(iou_xywh(now[o], after[o]) < .7 for o in now):
                continue
            kind = "person" if any(v[4] == 1 for v in now.values()) else "animal"
            candidates[kind].append((sequence, frame, list(now.values())))
    print(f'BIRDSAI candidates: {len(candidates["person"])} with people, {len(candidates["animal"])} animal-only', flush=True)
    picks = []
    for kind in ("person", "animal"):
        pool = candidates[kind]
        rng.shuffle(pool)
        # Spread picks across sequences rather than taking many near-identical frames from one video.
        by_sequence = {}
        for c in pool:
            by_sequence.setdefault(c[0], []).append(c)
        wanted = 25 if kind == "person" else 50 - len(picks)
        while len(picks) < (25 if kind == "person" else 50) and any(by_sequence.values()):
            for sequence in list(by_sequence):
                if by_sequence[sequence] and wanted:
                    picks.append(by_sequence[sequence].pop())
                    wanted -= 1
    fetched = download_all([BIRDSAI + f"images/{s}/{s}_{f:010d}.jpg" for s, f, _ in picks])
    for sequence, frame, objects in picks:
        url = BIRDSAI + f"images/{sequence}/{sequence}_{frame:010d}.jpg"
        from PIL import Image
        w, h = Image.open(io.BytesIO(fetched[url])).size
        boxes = [box("person" if cls == 1 else "animal", x, y, x + bw, y + bh, w, h, None if cls == 1 else BIRDSAI_SPECIES.get(species))
                 for x, y, bw, bh, cls, species in objects]
        entry = {"id": f"birdsai-{sequence}-{frame}", "condition": "night", "source": "birdsai", "url": url,
                 "tags": ["aerial", "thermal"], "boxes": [b for b in boxes if b], "ignore": []}
        keep(entry, fetched[url], selected)


def select_waid(rng, selected):
    folders = json.loads(download("https://api.github.com/repos/xiaohuicui/WAID/contents/WAID/images"))
    tree = next(f for f in folders if f["name"] == "test")["sha"]
    names = sorted(t["path"] for t in json.loads(download(f"https://api.github.com/repos/xiaohuicui/WAID/git/trees/{tree}"))["tree"])
    rng.shuffle(names)
    names = names[:50]
    label = lambda n: WAID_RAW + "WAID/labels/test/" + n.rsplit(".", 1)[0] + ".txt"
    fetched = download_all([WAID_RAW + "WAID/images/test/" + n for n in names] + [label(n) for n in names])
    from PIL import Image
    for name in names:
        url = WAID_RAW + "WAID/images/test/" + name
        w, h = Image.open(io.BytesIO(fetched[url])).size
        boxes = []
        for line in fetched[label(name)].decode().split("\n"):
            if line.strip():
                cls, cx, cy, bw, bh = line.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append(box("animal", cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, w, h, WAID_SPECIES[int(cls)]))
        entry = {"id": "waid-" + hashlib.sha1(name.encode()).hexdigest()[:12], "condition": "aerial", "source": "waid", "url": url,
                 "boxes": [b for b in boxes if b], "ignore": []}
        keep(entry, fetched[url], selected)


def select_visdrone(rng, selected):
    from PIL import Image
    with zipfile.ZipFile(cached(VISDRONE_ZIP)) as archive:
        members = sorted(n for n in archive.namelist() if n.endswith(".jpg"))
        rng.shuffle(members)
        for member in members[:50]:
            data = archive.read(member)
            w, h = Image.open(io.BytesIO(data)).size
            label = member.replace("/images/", "/annotations/").rsplit(".", 1)[0] + ".txt"
            boxes, ignore = [], []
            for line in archive.read(label).decode().split("\n"):
                if not line.strip():
                    continue
                x, y, bw, bh, _, cls = (int(v) for v in line.strip().rstrip(",").split(",")[:6])
                if cls in (0, 11):  # ignored regions and "others"
                    ignore.append([x, y, x + bw, y + bh])
                else:
                    boxes.append(box("person" if cls in (1, 2) else "vehicle", x, y, x + bw, y + bh, w, h))
            entry = {"id": "visdrone-" + Path(member).stem, "condition": "aerial", "source": "visdrone",
                     "zip": VISDRONE_ZIP, "member": member, "boxes": [b for b in boxes if b], "ignore": ignore}
            keep(entry, data, selected)


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.


def select():
    selected = []
    for name, step in [("ena24", select_ena24), ("coco", select_coco), ("birdsai", select_birdsai),
                       ("waid", select_waid), ("visdrone", select_visdrone)]:
        before = len(selected)
        step(random.Random(f"{SEED}-{name}"), selected)
        print(f"{name}: {len(selected) - before} images", flush=True)
    counts = {c: {s: sum(1 for e in selected if e["condition"] == c and e["source"] == s) for s in q} for c, q in QUOTAS.items()}
    if counts != QUOTAS:
        raise RuntimeError(f"Selection does not match quotas: {counts}")
    MANIFEST.write_text(json.dumps({"version": 1, "seed": SEED, "quotas": QUOTAS, "sources": SOURCES, "images": selected}, indent=1), encoding="utf-8")
    print(f"Wrote {MANIFEST} with {len(selected)} images.")


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def problems():
    """Return a list of missing or changed images (empty when the set is complete)."""
    issues = []
    for entry in load_manifest()["images"]:
        path = IMAGES / f'{entry["id"]}.jpg'
        if not path.exists():
            issues.append(f'missing {path.name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            issues.append(f'changed {path.name}')
    return issues


def fetch():
    entries = load_manifest()["images"]
    IMAGES.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if "zip" in entry:
            cached(entry["zip"])  # download each archive once, before the parallel workers start

    def one(entry):
        path = IMAGES / f'{entry["id"]}.jpg'
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            return
        data = image_bytes(entry)
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f'{entry["id"]}: downloaded image does not match the manifest checksum')
        path.write_bytes(data)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, entries))
    print(f"Evaluation set ready: {len(entries)} images in {IMAGES}")


def verify():
    issues = problems()
    for issue in issues[:20]:
        print(issue)
    print("Evaluation set OK" if not issues else f"{len(issues)} problem(s). Run: python evaluation/eval_set.py fetch")
    return not issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "verify", "select"])
    command = parser.parse_args().command
    if command == "select":
        select()
    elif command == "fetch":
        fetch()
    else:
        sys.exit(0 if verify() else 1)
