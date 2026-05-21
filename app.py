from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Color Tables ──────────────────────────────────────────────────────────────

digit_colors = {
    "black": 0, "brown": 1, "red": 2,    "orange": 3,
    "yellow": 4, "green": 5, "blue": 6,   "violet": 7,
    "grey": 8,   "white": 9
}

multiplier = {
    "black": 1,          "brown": 10,        "red": 100,
    "orange": 1000,      "yellow": 10000,    "green": 100000,
    "blue": 1000000,     "violet": 10000000, "grey": 100000000,
    "white": 1000000000, "gold": 0.1,        "silver": 0.01
}

tolerance = {
    "brown":  "±1%",   "red":    "±2%",   "green":  "±0.5%",
    "blue":   "±0.25%","violet": "±0.1%", "grey":   "±0.05%",
    "gold":   "±5%",   "silver": "±10%",  "none":   "±20%"
}

# Hex colors for visual band display in the UI
color_hex = {
    "black":  "#1a1a1a", "brown":  "#7B3F00", "red":    "#e53935",
    "orange": "#FB8C00", "yellow": "#FDD835", "green":  "#43A047",
    "blue":   "#1E88E5", "violet": "#8E24AA", "grey":   "#757575",
    "white":  "#F5F5F5", "gold":   "#FFD700", "silver": "#C0C0C0"
}

all_colors = [
    "black", "brown", "red",    "orange",
    "yellow","green", "blue",   "violet",
    "grey",  "white", "gold",   "silver"
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_resistance(value):
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f} GΩ"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f} MΩ"
    elif value >= 1_000:
        return f"{value/1_000:.2f} kΩ"
    else:
        return f"{value:.2f} Ω"


# ─── Band Detection ────────────────────────────────────────────────────────────

def detect_bands(image_path):
    """
    Detect up to 5 color bands from a resistor image using HSV analysis.
    Returns list of color name strings.
    """
    image = cv2.imread(image_path)
    if image is None:
        return []

    # Resize to standard working size
    image = cv2.resize(image, (700, 250))

    # Crop center strip where bands live (remove background edges)
    h, w = image.shape[:2]
    roi = image[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply slight Gaussian blur to reduce noise
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    color_ranges = {
        "black":  ([0, 0, 0],    [180, 255, 55]),
        "brown":  ([5, 70, 30],  [20,  255, 170]),
        "red":    ([0, 130, 70], [10,  255, 255]),
        "red2":   ([165,130,70], [180, 255, 255]),   # red wraps hue
        "orange": ([11, 100, 100],[24, 255, 255]),
        "yellow": ([22, 100, 100],[34, 255, 255]),
        "green":  ([35, 50, 50], [85,  255, 255]),
        "blue":   ([90, 50, 50], [130, 255, 255]),
        "violet": ([131, 50, 50],[160, 255, 255]),
        "grey":   ([0, 0, 80],   [180, 35, 195]),
        "white":  ([0, 0, 195],  [180, 25, 255]),
        "gold":   ([16, 80, 130],[34, 255, 255]),
        "silver": ([0, 0, 120],  [180, 38, 215])
    }

    positions = []

    for color_key, (lower, upper) in color_ranges.items():
        lower_arr = np.array(lower)
        upper_arr = np.array(upper)
        mask = cv2.inRange(hsv, lower_arr, upper_arr)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Map red2 back to red
        actual_color = "red" if color_key == "red2" else color_key

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 900:
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                aspect = h_cnt / max(w_cnt, 1)
                # Bands are tall and narrow
                if h_cnt > 50 and w_cnt < 130 and aspect > 0.8:
                    cx = x + w_cnt // 2
                    positions.append((cx, actual_color, area))

    if not positions:
        return []

    # Sort by x position
    positions.sort(key=lambda p: p[0])

    # Merge detections that are very close (same band detected twice)
    merged = []
    last_x = -999
    for x, color, area in positions:
        if x - last_x > 35:
            merged.append((x, color))
            last_x = x
        else:
            # Keep the larger detection (more reliable)
            if area > merged[-1][1] if merged else 0:
                merged[-1] = (x, color)

    # Deduplicate consecutive same colors
    final = []
    for x, color in merged:
        if not final or final[-1] != color:
            final.append(color)

    return final[:5]


# ─── Resistance Calculation ────────────────────────────────────────────────────

def calculate_resistance(bands):
    """
    Calculate resistance from 4 or 5 band list.
    Returns dict with value_text, tolerance, ohms, band_count, success.
    """
    try:
        n = len(bands)

        if n < 4:
            return {
                "success": False,
                "error": f"Need at least 4 bands — only {n} detected.",
                "value_text": "—",
                "tolerance": "—",
                "ohms": 0
            }

        if n >= 5:
            # 5-band: d1 d2 d3 × multiplier | tolerance
            b1, b2, b3, b4, b5 = bands[0], bands[1], bands[2], bands[3], bands[4]
            value = (
                digit_colors[b1] * 100 +
                digit_colors[b2] * 10 +
                digit_colors[b3]
            ) * multiplier[b4]
            tol = tolerance.get(b5, "±5%")
        else:
            # 4-band: d1 d2 × multiplier | tolerance
            b1, b2, b3, b4 = bands[0], bands[1], bands[2], bands[3]
            value = (
                digit_colors[b1] * 10 +
                digit_colors[b2]
            ) * multiplier[b3]
            tol = tolerance.get(b4, "±5%")

        value_text = format_resistance(value)

        return {
            "success": True,
            "value_text": value_text,
            "tolerance": tol,
            "ohms": value,
            "band_count": n,
            "error": None
        }

    except KeyError as e:
        return {
            "success": False,
            "error": f"Invalid band color: {e}",
            "value_text": "Error",
            "tolerance": "—",
            "ohms": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "value_text": "Error",
            "tolerance": "—",
            "ohms": 0
        }


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/app", methods=["GET", "POST"])
def index():
    image_path = None
    detected = []
    result = {}

    if request.method == "POST":

        # ── Manual correction form submitted
        if "bands" in request.form:
            bands = request.form.getlist("bands")
            image_path = request.form.get("image_path", "")
            result = calculate_resistance(bands)
            detected = bands

        # ── New image uploaded
        elif "image" in request.files:
            file = request.files["image"]
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                image_path = save_path
                detected = detect_bands(save_path)

                # Auto-calculate if we got enough bands
                if len(detected) >= 4:
                    result = calculate_resistance(detected)
            else:
                result = {"success": False, "error": "Invalid file type. Use PNG, JPG, or JPEG."}

    return render_template(
        "index.html",
        image_path=image_path,
        detected=detected,
        result=result,
        all_colors=all_colors,
        color_hex=color_hex
    )


# ─── API Endpoint (bonus: for AJAX use) ───────────────────────────────────────

@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    data = request.get_json()
    bands = data.get("bands", [])
    result = calculate_resistance(bands)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
