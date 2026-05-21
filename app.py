from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Color Tables ────────────────────────────────────────────────

digit_colors = {
    "black": 0, "brown": 1, "red": 2,    "orange": 3,
    "yellow": 4,"green": 5, "blue": 6,   "violet": 7,
    "grey": 8,  "white": 9
}

multiplier = {
    "black": 1,         "brown": 10,       "red": 100,
    "orange": 1000,     "yellow": 10000,   "green": 100000,
    "blue": 1000000,    "violet": 10000000,"grey": 100000000,
    "white": 1000000000,"gold": 0.1,       "silver": 0.01
}

tolerance = {
    "brown": "±1%",  "red": "±2%",    "green": "±0.5%",
    "blue": "±0.25%","violet": "±0.1%","grey": "±0.05%",
    "gold": "±5%",   "silver": "±10%"
}

color_hex = {
    "black":  "#222222",
    "brown":  "#7B3F00",
    "red":    "#e53935",
    "orange": "#FB8C00",
    "yellow": "#FDD835",
    "green":  "#43A047",
    "blue":   "#1E88E5",
    "violet": "#8E24AA",
    "grey":   "#9E9E9E",
    "white":  "#F5F5F5",
    "gold":   "#FFD700",
    "silver": "#C0C0C0"
}

all_colors = [
    "black","brown","red","orange",
    "yellow","green","blue","violet",
    "grey","white","gold","silver"
]

# ── Detection ───────────────────────────────────────────────────

def detect_bands(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return []

    image = cv2.resize(image, (700, 250))

    # Crop centre ROI to remove background noise
    h, w = image.shape[:2]
    roi = image[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    color_ranges = {
        "black":  ([0,   0,   0],   [180, 255,  55]),
        "brown":  ([5,   70,  30],  [20,  255, 170]),
        "red":    ([0,  130,  70],  [10,  255, 255]),
        "red2":   ([165,130,  70],  [180, 255, 255]),
        "orange": ([11, 100, 100],  [24,  255, 255]),
        "yellow": ([22, 100, 100],  [34,  255, 255]),
        "green":  ([35,  50,  50],  [85,  255, 255]),
        "blue":   ([90,  50,  50],  [130, 255, 255]),
        "violet": ([131, 50,  50],  [160, 255, 255]),
        "grey":   ([0,   0,   80],  [180,  35, 195]),
        "white":  ([0,   0,  195],  [180,  25, 255]),
        "gold":   ([16,  80, 130],  [34,  255, 255]),
        "silver": ([0,   0,  120],  [180,  38, 215])
    }

    positions = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for color_key, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        actual_color = "red" if color_key == "red2" else color_key

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 900:
                x, y, wc, hc = cv2.boundingRect(cnt)
                if hc > 50 and wc < 130 and (hc / max(wc, 1)) > 0.8:
                    positions.append((x + wc // 2, actual_color, area))

    if not positions:
        return []

    positions.sort(key=lambda p: p[0])

    # Merge detections that are very close (same band)
    merged = []
    last_x = -999
    for x, color, area in positions:
        if x - last_x > 35:
            merged.append((x, color))
            last_x = x

    # Deduplicate consecutive same colors
    final = []
    for _, color in merged:
        if not final or final[-1] != color:
            final.append(color)

    return final[:5]


# ── Calculation ─────────────────────────────────────────────────

def calculate_resistance(bands):
    try:
        n = len(bands)
        if n < 4:
            return {
                "success": False,
                "error": f"Need at least 4 bands — only {n} detected.",
                "value_text": "—", "tolerance": "—",
                "ohms": 0, "band_count": n
            }

        if n >= 5:
            value = (
                digit_colors[bands[0]] * 100 +
                digit_colors[bands[1]] * 10  +
                digit_colors[bands[2]]
            ) * multiplier[bands[3]]
            tol = tolerance.get(bands[4], "±5%")
        else:
            value = (
                digit_colors[bands[0]] * 10 +
                digit_colors[bands[1]]
            ) * multiplier[bands[2]]
            tol = tolerance.get(bands[3], "±5%")

        if value >= 1_000_000_000:
            vt = f"{value/1_000_000_000:.2f} GΩ"
        elif value >= 1_000_000:
            vt = f"{value/1_000_000:.2f} MΩ"
        elif value >= 1_000:
            vt = f"{value/1_000:.2f} kΩ"
        else:
            vt = f"{value:.2f} Ω"

        return {
            "success": True,
            "value_text": vt,
            "tolerance": tol,
            "ohms": value,
            "band_count": n,
            "error": None
        }

    except KeyError as e:
        return {"success": False, "error": f"Invalid color: {e}", "value_text": "Error", "tolerance": "—", "ohms": 0, "band_count": 0}
    except Exception as e:
        return {"success": False, "error": str(e), "value_text": "Error", "tolerance": "—", "ohms": 0, "band_count": 0}


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/app", methods=["GET", "POST"])
def index():
    image_path = None
    detected   = []
    result     = {}

    if request.method == "POST":
        if "bands" in request.form:
            bands      = request.form.getlist("bands")
            image_path = request.form.get("image_path", "")
            result     = calculate_resistance(bands)
            detected   = bands

        elif "image" in request.files:
            file = request.files["image"]
            if file and file.filename:
                filename  = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                image_path = save_path
                detected   = detect_bands(save_path)
                if len(detected) >= 4:
                    result = calculate_resistance(detected)

    return render_template(
        "index.html",
        image_path = image_path,
        detected   = detected,
        result     = result,
        all_colors = all_colors,
        color_hex  = color_hex
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
