from __future__ import annotations

import csv
import io
import json
from collections import Counter

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from core import CAMERAS, annotate, assess, category, history, save_observation, update_review, verification_labels, correct_detections, update_detections, ANIMALS, VEHICLES
from detector import load_model, predict

st.set_page_config(page_title="VanRakshak | Wildlife command center", page_icon="🌿", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1480px}
h1{font-size:3.1rem!important;letter-spacing:-1.5px;margin-bottom:0!important}
h2,h3{letter-spacing:-.5px}
[data-testid="stMetric"]{background:#152521;border:1px solid #2a4037;padding:18px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#aabfb3}
[data-testid="stSidebar"]{border-right:1px solid #294136}
.eyebrow{font-size:12px;letter-spacing:3px;color:#9be7b1;font-weight:700}
.hero{padding:26px 30px;background:linear-gradient(115deg,#1e3c2e,#14271f);border:1px solid #34553f;border-radius:18px;margin-bottom:24px}
.hero p{color:#b9cfc0;max-width:780px;font-size:17px}
.footer{color:#90a99c;font-size:12px;padding-top:22px;border-top:1px solid #294136;margin-top:30px}
.stButton>button[kind="primary"]{color:#102219;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def model(profile):
    return load_model(profile)

with st.sidebar:
    st.markdown("## 🌿 VanRakshak")
    st.caption("FOREST INTELLIGENCE")
    mode = "YOLO detection"
    profile = st.selectbox("Recognition engine", ["Wildlife", "Standard"], help="Wildlife uses YOLOE plus crop-level CLIP naming. Standard uses the faster, limited-category YOLO11n detector.")
    st.caption("Wildlife: broader species estimates, slower processing." if profile == "Wildlife" else "Standard: fast detection with limited species coverage.")
    st.divider()
    st.markdown("#### Camera location")
    camera = st.selectbox("Monitoring point", ["Location not assigned", "Custom camera"])
    if camera == "Custom camera":
        lat = st.number_input("Latitude", -90., 90., value=None, format="%.5f")
        lon = st.number_input("Longitude", -180., 180., value=None, format="%.5f")
    else:
        lat, lon = None, None
        st.caption("Add camera coordinates to map this observation.")
    restricted = st.toggle("Restricted location", value=False)
    authorized = st.toggle("Known authorized visit", value=False, help="Operator confirmation of a patrol or service visit; not inferred by AI.")
    threshold = st.slider("Minimum detection confidence", .10, .95, .35, .05)
    st.caption("Location and access settings apply when you run an analysis.")
    st.divider()
    st.caption("SDG 15 · Life on Land\n\nSDG 9 · Innovation\n\nSDG 13 · Ecosystem protection")

st.markdown('''<div class="hero"><div class="eyebrow">WILDLIFE MONITORING & THREAT DETECTION</div>
<h1>Eyes on the forest.</h1><p>Turn camera imagery into wildlife observations and location-based alerts for ranger review.</p>
</div>''', unsafe_allow_html=True)

monitor, activity = st.tabs(["◎  Analyze imagery", "⌖  Activity & alerts"])

if st.session_state.get("result_version") != 4:
    st.session_state.pop("result", None)
    st.session_state.result_version = 4

with monitor:
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        upload = st.file_uploader("Upload camera imagery", type=["jpg", "jpeg", "png", "webp"])
        image = None
        source = ""
        image_file = upload
        if image_file is not None:
            try:
                with Image.open(image_file) as opened:
                    if opened.width * opened.height > 24_000_000:
                        st.error("Please use an image below 24 megapixels.")
                    else:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        image.thumbnail((1920, 1920))
                        source = upload.name
                        st.image(image, caption=source, width="stretch")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                st.error("This file could not be read as a supported image. Try another JPG or PNG.")
        else:
            st.markdown("### Upload a frame to begin")
            st.caption("JPG, PNG or WebP · up to 15 MB · below 24 megapixels")
        location_incomplete = camera == "Custom camera" and (lat is None or lon is None)
        if location_incomplete:
            st.info("Enter both camera coordinates, or select Location not assigned.")
        run = st.button("Analyze image →", type="primary", disabled=image is None or location_incomplete, width="stretch")
    with right:
        st.markdown("### Observation settings")
        st.write("**Location:**", camera)
        st.write("**Zone:**", "Restricted" if restricted else "Open / monitored")
        st.write("**Access:**", "Operator-confirmed authorized" if authorized else "Not confirmed")
        st.write("**Confidence threshold:**", f"{threshold:.0%}")
        st.caption("Alerts are based on people or vehicles at a camera marked restricted. They do not establish identity, intent, or an exact position within the image.")
        with st.expander("Recognition coverage & visibility"):
            if profile == "Wildlife":
                from vocabulary import WILDLIFE
                st.write("Named wildlife targets: " + ", ".join(WILDLIFE) + ".")
                st.write("YOLOE locates objects; CLIP compares animal crops against candidate names. Names are AI estimates, not verified species. Weak distinctions remain unverified. Detection scores and naming similarities measure different things.")
            else:
                st.write("Standard supports the original COCO animal categories. Gorillas, deer, tigers, rhinos and leopards are not supported; animal species remain unverified.")
            st.write("Lighting, small aerial objects, camouflage, fog and occlusion can reduce accuracy. There is no reserve-specific training. Image brightness alone cannot establish detection reliability.")
    if run and image is not None:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Analyzing imagery…"):
                detections = verification_labels(predict(model(profile), image, threshold))
                status, reason = assess(detections, restricted, authorized)
                brightness = float(np.asarray(image.convert("L")).mean())
                record = dict(mode=mode, camera=camera, lat=lat, lon=lon, source=source,
                              restricted=int(restricted), authorized=int(authorized), threshold=threshold,
                              status=status, reason=reason, detections=detections)
                record_id = save_observation(record)
                st.session_state.result = {"record": record, "image": annotate(image, detections), "brightness": brightness, "id": record_id, "original": image.copy(), "profile": profile}
        except Exception as exc:
            st.error(f"Analysis could not complete: {exc}")
            st.caption("Run setup.ps1 to install dependencies and download the model weights.")
    result = st.session_state.get("result")
    if result and result["record"]["mode"] == mode:
        rec = result["record"]
        if rec["detections"] and "original" in result:
            with st.expander("Review and correct detected objects"):
                st.caption("Corrections are operator-confirmed labels. Model scores remain the original scores, not confidence in your correction. Removing or changing a person/vehicle recalculates the alert.")
                options = sorted(ANIMALS | VEHICLES | {"person"}) + ["False detection"]
                with st.form(f'corrections-{result["id"]}'):
                    labels = []
                    for index, item in enumerate(rec["detections"]):
                        choices = options if item["label"] in options else options + [item["label"]]
                        labels.append(st.selectbox(f'Object {index + 1} · model suggested {item.get("model_label", item["label"])}', choices, index=choices.index(item["label"]), key=f'{result["id"]}-label-{index}'))
                    if st.form_submit_button("Save corrected labels"):
                        rec["detections"] = correct_detections(rec["detections"], labels)
                        rec["status"], rec["reason"] = assess(rec["detections"], rec["restricted"], rec["authorized"])
                        update_detections(result["id"], rec["detections"], rec["status"], rec["reason"])
                        result["image"] = annotate(result["original"], rec["detections"])
                        st.success("Corrected labels saved. Results and alert updated.")

        st.divider()
        st.markdown("### Analysis result")
        st.caption(f'Saved observation {result["id"]} · {rec["mode"]} · {rec["camera"]} · {rec["source"]} · threshold {rec["threshold"]:.0%}')
        st.caption(f'Recognition engine used: {result.get("profile", "Standard")}. Species followed by ? are AI estimates awaiting review.')
        if any(d["label"] == "animal (unverified)" for d in rec["detections"]):
            st.warning("Species could not be distinguished reliably for one or more animals. Review the image and correct labels above.")
        if any(d.get("naming_status", "").startswith("Conflicting") for d in rec["detections"]):
            st.warning("The detector and naming model disagree on an object. A possible person remains in access review until an operator verifies it.")
        counts = Counter(category(d["label"]) for d in rec["detections"])
        cols = st.columns(4)
        for col, title in zip(cols, ["Animal", "Human", "Vehicle", "Other"]):
            col.metric(title + " detections", counts[title])
        col_a, col_b = st.columns([1.6, 1], gap="large")
        with col_a:
            st.image(result["image"], caption="Annotated analysis result", width="stretch")
            buf = io.BytesIO()
            result["image"].save(buf, format="PNG")
            st.download_button("Download annotated image", buf.getvalue(), f'vanrakshak-{result["id"]}.png', "image/png")
        with col_b:
            if rec["status"] == "Review required":
                st.error("⚑ Potential intrusion · Review required")
            elif rec["detections"]:
                st.success(rec["status"])
            else:
                st.info(rec["status"])
            st.write(rec["reason"])
            if result["brightness"] < 65:
                st.warning("Low average brightness. Detection reliability may be reduced; inspect the image manually.")
            st.caption("Confidence is a model score, not a guarantee that the label is correct. Counts are per image and do not represent unique tracked individuals.")
            if rec["detections"]:
                st.dataframe([{"Suggested name": d["label"], "Type": category(d["label"]), "Detector label": d.get("model_label", d["label"]), "Verification": "Operator confirmed" if d.get("verified") else d.get("naming_status", "Unverified"), "Detection score": f'{d["confidence"]:.0%}', "Name similarity": d.get("name_similarity"), "Name candidates": ", ".join(c["label"] for c in d.get("name_candidates", []))} for d in rec["detections"]], hide_index=True, width="stretch")
                st.caption("Name similarity is cosine similarity, not a probability. Detector scores do not measure confidence in a corrected or separately estimated species.")
            st.download_button("Download JSON report", json.dumps({"id": result["id"], **rec}, indent=2), f'vanrakshak-{result["id"]}.json', "application/json")



with activity:
    st.markdown("### Activity across the reserve")
    rows = history("YOLO detection")
    for row in rows:
        row["detections"] = verification_labels(row["detections"])
        if row["camera"] in CAMERAS:
            row.update(camera="Location not assigned", lat=None, lon=None)
    st.caption("Latest 500 observations. Timestamps are UTC; map pins represent configured cameras, not object GPS.")
    if rows:
        a, b, c = st.columns(3)
        a.metric("Frames analyzed", len(rows))
        b.metric("Frames requiring review", sum(r["status"] == "Review required" and r["review"] not in {"Resolved", "False alarm"} for r in rows))
        c.metric("Camera locations", len({(r["lat"], r["lon"]) for r in rows if r["lat"] is not None and r["lon"] is not None}))
        points = {(r["lat"], r["lon"]) for r in rows if r["detections"] and r["lat"] is not None and r["lon"] is not None}
        if points:
            st.map([{"lat": p[0], "lon": p[1]} for p in points], latitude="lat", longitude="lon", zoom=11)
        else:
            st.info("No detected activity to map yet.")
        st.caption("Map tiles require internet. Camera coordinates remain available in the table and CSV offline.")
        table = [{"ID": r["id"], "Time (UTC)": r["captured_at"], "Mode": r["mode"], "Camera": r["camera"], "Latitude": r["lat"], "Longitude": r["lon"], "Objects": ", ".join(d["label"] for d in r["detections"]) or "None", "Alert": r["status"], "Review": r["review"]} for r in rows]
        st.dataframe(table, hide_index=True, width="stretch")
        export = io.StringIO(newline="")
        writer = csv.DictWriter(export, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        st.download_button("Export activity CSV", export.getvalue(), "vanrakshak-activity.csv", "text/csv")
        with st.expander("Update ranger review"):
            choices = {r["id"]: r for r in rows}
            chosen = st.selectbox("Observation", list(choices), format_func=lambda key: f'{key} · {choices[key]["source"]} · {choices[key]["mode"]}')
            states = ["Unreviewed", "Investigating", "Resolved", "False alarm"]
            review = st.selectbox("Review status", states, index=states.index(choices[chosen]["review"]))
            if st.button("Save review status"):
                update_review(chosen, review)
                st.rerun()
    else:
        st.info("No saved observations yet. Analyze an image to create your first record.")

st.markdown('<div class="footer">VANRAKSHAK · Human-reviewed alerts · Local processing · Camera coordinates supplied by operator</div>', unsafe_allow_html=True)
