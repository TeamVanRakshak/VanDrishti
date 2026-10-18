"""Detection scoring: greedy IoU matching per category, then precision, recall and species accuracy."""
CATEGORIES = ("person", "vehicle", "animal")
SIZES = (("small", 32 ** 2), ("medium", 96 ** 2), ("large", float("inf")))
UNVERIFIED = "animal (unverified)"


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def size(b):
    area = (b[2] - b[0]) * (b[3] - b[1])
    return next(name for name, limit in SIZES if area < limit)


def inside(point, region):
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def score_image(truth, ignore, predictions, threshold=.5):
    """Match predictions to ground truth of the same category.

    truth: [{"category", "species", "box"}]; predictions: [{"category", "label", "confidence", "box"}].
    Predictions outside our three categories, or centred in an ignore region and unmatched, are not counted.
    """
    gt = [dict(t, size=size(t["box"]), matched=False, predicted_label=None) for t in truth]
    false_positives = []
    for p in sorted(predictions, key=lambda p: -p["confidence"]):
        if p["category"] not in CATEGORIES:
            continue
        best, best_iou = None, threshold
        for t in gt:
            if not t["matched"] and t["category"] == p["category"]:
                overlap = iou(p["box"], t["box"])
                if overlap >= best_iou:
                    best, best_iou = t, overlap
        if best is not None:
            best.update(matched=True, predicted_label=p["label"])
            continue
        centre = ((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2)
        if any(inside(centre, region) for region in ignore):
            continue
        confused = max(gt, key=lambda t: iou(p["box"], t["box"]), default=None)
        confused = confused["category"] if confused and iou(p["box"], confused["box"]) >= threshold else None
        false_positives.append({"category": p["category"], "label": p["label"], "confused_with": confused})
    return {"truth": gt, "false_positives": false_positives}


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def summarize(results):
    """Aggregate score_image outputs into per-category metrics."""
    summary = {"images": len(results)}
    for category in CATEGORIES:
        gt = [t for r in results for t in r["truth"] if t["category"] == category]
        tp = sum(t["matched"] for t in gt)
        fp = sum(1 for r in results for f in r["false_positives"] if f["category"] == category)
        summary[category] = {
            "objects": len(gt), "tp": tp, "fn": len(gt) - tp, "fp": fp,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, len(gt)),
            "recall_by_size": {name: ratio(sum(t["matched"] for t in gt if t["size"] == name), sum(1 for t in gt if t["size"] == name))
                               for name, _ in SIZES},
        }
    named = [t for r in results for t in r["truth"] if t["category"] == "animal" and t["matched"] and t["species"]]
    correct = sum(t["predicted_label"] == t["species"] for t in named)
    abstained = sum(t["predicted_label"] == UNVERIFIED for t in named)
    summary["species"] = {"matched_with_known_species": len(named), "correct": correct, "abstained": abstained,
                          "wrong": len(named) - correct - abstained, "accuracy": ratio(correct, len(named)),
                          "accuracy_when_named": ratio(correct, len(named) - abstained)}
    confusions = {}
    for r in results:
        for f in r["false_positives"]:
            if f["confused_with"] and f["confused_with"] != f["category"]:
                key = f'{f["confused_with"]} predicted as {f["category"]}'
                confusions[key] = confusions.get(key, 0) + 1
    summary["confusions"] = confusions
    return summary
