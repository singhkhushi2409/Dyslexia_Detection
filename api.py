import os
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "handwriting_model.pkl")
FEATURE_COLUMNS = [
    "ink_density", "connected_components", "mean_height", "height_variation",
    "mean_width", "width_variation", "baseline_drift", "mean_spacing",
    "spacing_variation", "estimated_stroke_width", "slant_variation",
    "margin_alignment", "overlap_ratio",
]


def extract_features(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read handwriting image.")
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    height, width = binary.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [
        {"x": row[cv2.CC_STAT_LEFT], "y": row[cv2.CC_STAT_TOP],
         "w": row[cv2.CC_STAT_WIDTH], "h": row[cv2.CC_STAT_HEIGHT]}
        for row in stats[1:]
        if row[cv2.CC_STAT_AREA] >= 20
        and row[cv2.CC_STAT_HEIGHT] >= 5
        and row[cv2.CC_STAT_WIDTH] >= 2
    ]
    if len(components) < 2:
        raise ValueError("Not enough handwriting detected. Upload a clearer sample.")
    components.sort(key=lambda item: item["x"])
    heights = np.array([item["h"] for item in components], dtype=float)
    widths = np.array([item["w"] for item in components], dtype=float)
    baselines = np.array([item["y"] + item["h"] for item in components], dtype=float)
    gaps = np.array([
        right["x"] - left["x"] - left["w"]
        for left, right in zip(components, components[1:])
        if right["x"] - left["x"] - left["w"] >= 0
    ], dtype=float)
    mean_gap = float(np.mean(gaps)) if len(gaps) else 0.0
    gap_variation = float(np.std(gaps) / (mean_gap + 1e-6) * 100) if len(gaps) else 0.0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_region = distance[binary > 0]
    slants = []
    for item in components:
        roi = binary[item["y"]:item["y"] + item["h"], item["x"]:item["x"] + item["w"]]
        moments = cv2.moments(roi)
        if abs(moments["mu02"]) > 1e-6:
            slants.append(np.degrees(0.5 * np.arctan2(
                2 * moments["mu11"], moments["mu20"] - moments["mu02"])))
    overlaps = sum(right["x"] < left["x"] + left["w"] for left, right in zip(components, components[1:]))
    return [
        np.count_nonzero(binary) / (height * width) * 100,
        float(len(components)), float(np.mean(heights)),
        float(np.std(heights) / (np.mean(heights) + 1e-6) * 100),
        float(np.mean(widths)), float(np.std(widths) / (np.mean(widths) + 1e-6) * 100),
        float(np.std(baselines)), mean_gap, gap_variation,
        float(np.mean(ink_region) * 2) if len(ink_region) else 0.0,
        float(np.std(slants)) if slants else 0.0,
        float(np.std([item["x"] for item in components])),
        overlaps / (len(components) - 1) * 100,
    ]


model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a handwriting image as 'image'."}), 400
    temporary_path = None
    try:
        suffix = os.path.splitext(upload.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            upload.save(temporary.name)
            temporary_path = temporary.name
        features = extract_features(temporary_path)
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        if hasattr(model, "feature_names_in_"):
            frame = frame[model.feature_names_in_]
        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        probability = next(float(value) * 100 for label, value in zip(model.classes_, probabilities) if int(label) == 1)
        return jsonify({
            "prediction": prediction,
            "dyslexiaProbability": probability,
            "verdict": "Higher likelihood of dyslexia markers detected" if prediction == 1 else "Lower likelihood of dyslexia markers detected",
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 422
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)