"""Run recognition engines over the evaluation set and save per-condition precision and recall.

    python evaluation/evaluate.py                      # both engines, results in evaluation/results/
    python evaluation/evaluate.py --engine Standard --limit 20
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from PIL import Image
from core import category
from detector import load_model, predict
from evaluation import eval_set
from evaluation.metrics import CATEGORIES, score_image, summarize

THRESHOLD = .35  # the app's default confidence threshold
GROUP = {"Human": "person", "Vehicle": "vehicle", "Animal": "animal"}


def run(engine, entries, threshold=THRESHOLD):
    model = load_model(engine)
    predict(model, Image.new("RGB", (64, 64)), threshold)  # warm-up, excluded from timing
    per_image, seconds = [], []
    for index, entry in enumerate(entries, 1):
        image = Image.open(eval_set.IMAGES / f'{entry["id"]}.jpg').convert("RGB")
        start = time.perf_counter()
        detections = predict(model, image, threshold)
        seconds.append(time.perf_counter() - start)
        predictions = [{"category": GROUP.get(category(d["label"]), "other"), "label": d["label"],
                        "confidence": d["confidence"], "box": d["box"]} for d in detections]
        scored = score_image(entry["boxes"], entry["ignore"], predictions)
        per_image.append({"id": entry["id"], "condition": entry["condition"], "source": entry["source"],
                          "seconds": round(seconds[-1], 3), "predictions": predictions, **scored})
        if index % 25 == 0:
            print(f"  {engine}: {index}/{len(entries)}", flush=True)
    groups = {"overall": per_image}
    for key in ("condition", "source"):
        for value in sorted({r[key] for r in per_image}):
            groups[value] = [r for r in per_image if r[key] == value]
    return {"engine": engine, "threshold": threshold, "iou_threshold": .5,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "images": len(per_image),
            "seconds_per_image": round(sum(seconds) / len(seconds), 3),
            "metrics": {name: summarize(rows) for name, rows in groups.items()}, "per_image": per_image}


def table(result):
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    lines = [f'{result["engine"]}: {result["images"]} images, {result["seconds_per_image"]} s/image, threshold {result["threshold"]}',
             f'{"group":<10}' + "".join(f"{c + ' P':>11}{c + ' R':>11}" for c in CATEGORIES) + f'{"species acc":>13}']
    for name, m in result["metrics"].items():
        lines.append(f"{name:<10}" + "".join(f'{fmt(m[c]["precision"]):>11}{fmt(m[c]["recall"]):>11}' for c in CATEGORIES)
                     + f'{fmt(m["species"]["accuracy"]):>13}')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", nargs="+", default=["Standard", "Wildlife"], choices=["Standard", "Wildlife"])
    parser.add_argument("--limit", type=int, help="evaluate only the first N images (quick check, not saved)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    issues = eval_set.problems()
    if issues:
        sys.exit(f"Evaluation set incomplete ({len(issues)} problems). Run: python evaluation/eval_set.py fetch")
    entries = eval_set.load_manifest()["images"][:args.limit]
    for engine in args.engine:
        result = run(engine, entries)
        print(table(result))
        if not args.limit:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{engine.lower()}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f'Saved {args.out / (engine.lower() + ".json")}')


if __name__ == "__main__":
    main()
