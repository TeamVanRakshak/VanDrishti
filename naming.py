"""Independent crop-level naming. Similarity is not a probability."""
from pathlib import Path
from vocabulary import WILDLIFE

NAMES = WILDLIFE + ["person", "car", "truck", "bus", "motorcycle", "bicycle", "boat", "airplane", "train"]


def load_namer():
    import clip
    import torch
    root = Path(__file__).resolve().parent
    weights = root / ".runtime" / "clip" / "ViT-B-32.pt"
    if not weights.exists():
        raise FileNotFoundError("Naming weights missing. Run setup.ps1 or download_namer.py while online.")
    model, preprocess = clip.load(str(weights), device="cpu")
    with torch.inference_mode():
        text = model.encode_text(clip.tokenize([f"a photo of a {name}" for name in NAMES]))
        text /= text.norm(dim=-1, keepdim=True)
    return model, preprocess, text


def apply_name_scores(item, candidates):
    """Keep disagreements and weak distinctions visible instead of inventing certainty."""
    item = dict(item)
    top, second = candidates[:2]
    margin = top["similarity"] - second["similarity"]
    item.update(name_candidates=candidates, name_similarity=top["similarity"], name_margin=round(margin, 4), naming_model="CLIP ViT-B/32")
    detected = item.get("model_label", item["label"])
    if detected in WILDLIFE:
        matching_estimates = top["label"] == detected and item.get("confidence", 0) >= .6
        if top["label"] in WILDLIFE and top["similarity"] >= .24 and (margin >= .015 or matching_estimates):
            item.update(label=top["label"], species_candidate=True, naming_status="AI estimate", verified=False)
            if matching_estimates and margin < .015:
                item["naming_status"] = "AI estimate — matching predictions, close alternatives"
        else:
            item.update(label="animal (unverified)", species_candidate=False, naming_status="Uncertain species", verified=False)
    elif detected == "person" and top["label"] in WILDLIFE:
        # Do not remove a possible human from intrusion review based on a crop classifier.
        item.update(naming_status="Conflicting predictions — review", verified=False)
    else:
        item["naming_status"] = "AI estimate"
    return item


def refine_names(image, detections, namer):
    import torch
    model, preprocess, text = namer
    output = []
    with torch.inference_mode():
        for item in detections:
            if item["label"] not in WILDLIFE and item["label"] != "person":
                output.append(item)
                continue
            x1, y1, x2, y2 = item["box"]
            box = (max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2)))
            if box[2] <= box[0] or box[3] <= box[1]:
                output.append(item)
                continue
            feature = model.encode_image(preprocess(image.crop(box)).unsqueeze(0))
            feature /= feature.norm(dim=-1, keepdim=True)
            top = (feature @ text.T)[0].topk(3)
            candidates = [{"label": NAMES[int(i)], "similarity": round(float(v), 4)} for v, i in zip(top.values, top.indices)]
            output.append(apply_name_scores(item, candidates))
    return output
