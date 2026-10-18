# Recognition upgrade — 18 October 2026

Wildlife combines YOLOE-11m localization at 960 pixels with CLIP ViT-B/32 crop naming. Standard retains YOLO11n at 640 pixels. These are pretrained models with a configured vocabulary, not custom-trained reserve models.

## Available-image comparison

At detection threshold 0.35, using the application's resizing and inference path:

| Input | Standard raw predictions | Wildlife predictions |
|---|---|---|
| Giraffes | giraffe | giraffe; second animal unverified |
| Zebra | zebra | zebra |
| People and bus | person, bus | person, bus |
| Clean gorilla photograph | bear | gorilla (AI estimate) |
| User screenshot | person, bear | person, gorilla; false traffic-sign detections |

`upgrade.json` contains full boxes, original labels, crop candidates, cosine similarities and timings. Standard raw animal labels are displayed as unverified animals in the app.

The screenshot is **not fully corrected**: the left gorilla is still detected as a possible person; overlay labels are mistaken for signs. Disagreements are flagged and possible humans remain in review. The rear gorilla gets a matching gorilla estimate, with close alternatives visible. The clean-photo result supports a concrete naming improvement, not universal gorilla recognition.

Standard took about 0.2–0.32 seconds/image; Wildlife 1.17–3.52 seconds after loading in this run. Initial loading is slower. Hardware and workload affect timings. Different configured resolutions mean this is not a model-only benchmark.

## Naming policy and limits

The detector searches 47 targets; CLIP compares crops against 40 animal/human/vehicle names. It cannot name arbitrary objects outside those vocabularies. A new name in the vocabulary does not establish its accuracy.

Cosine similarity is not a probability. A species change requires similarity at least 0.24 and a top-two margin at least 0.015. Matching detector/crop predictions can retain a name when detection score is at least 0.6; close alternatives stay recorded. These are heuristic settings, not calibrated guarantees. Original detector scores are preserved separately and are not reused as species confidence. Operator corrections remain distinct.

Five inputs were inspected during development, not held out. COCO images may overlap pretraining. No mAP, accuracy percentage, recall or field-reliability claim follows from these checks. The user's screenshot also contains prior annotations and UI text. Nine automated regression checks passed for rules, storage/migration, correction, uncertainty and score handling. Broader species, day/night, aerial and camera-location validation remains necessary.

See `../samples/SOURCES.md` for photo attribution and the CC BY-SA 3.0 license covering annotated gorilla derivatives.
