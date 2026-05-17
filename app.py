from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

digit_colors = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "grey": 8,
    "white": 9
}

multiplier = {
    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1000,
    "yellow": 10000,
    "green": 100000,
    "blue": 1000000,
    "violet": 10000000,
    "grey": 100000000,
    "white": 1000000000,
    "gold": 0.1,
    "silver": 0.01
}

tolerance = {
    "brown": "±1%",
    "red": "±2%",
    "green": "±0.5%",
    "blue": "±0.25%",
    "violet": "±0.1%",
    "grey": "±0.05%",
    "gold": "±5%",
    "silver": "±10%"
}

all_colors = [
    "black","brown","red","orange",
    "yellow","green","blue","violet",
    "grey","white","gold","silver"
]

def detect_bands(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (700, 250))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {

        "black": ([0,0,0],[180,255,60]),

        "brown": ([5,80,40],[20,255,180]),

        "red": ([0,120,70],[10,255,255]),

        "orange": ([10,100,100],[25,255,255]),

        "yellow": ([20,100,100],[35,255,255]),

        "green": ([35,50,50],[85,255,255]),

        "blue": ([90,50,50],[130,255,255]),

        "violet": ([130,50,50],[160,255,255]),

        "grey": ([0,0,80],[180,40,200]),

        "white": ([0,0,200],[180,30,255]),

        "gold": ([15,80,120],[35,255,255]),

        "silver": ([0,0,120],[180,40,220])
    }

    detected = []

    positions = []

    for color, (lower, upper) in color_ranges.items():

        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1200:

                x,y,w,h = cv2.boundingRect(cnt)

                if h > 60 and w < 120:

                    positions.append((x,color))

    positions = sorted(positions, key=lambda x:x[0])

    for p in positions:
        detected.append(p[1])

    final = []

    for color in detected:

        if color not in final:
            final.append(color)

    return final[:5]

def calculate_resistance(bands):

    try:

        if len(bands) < 4:
            return "Could not detect 4 valid bands."

        band1 = bands[0]
        band2 = bands[1]
        band3 = bands[2]
        band4 = bands[3]

        value = (
            (digit_colors[band1] * 10)
            + digit_colors[band2]
        ) * multiplier[band3]

        tol = tolerance.get(band4, "±5%")

        if value >= 1000000:
            value_text = f"{value/1000000:.1f} MΩ"

        elif value >= 1000:
            value_text = f"{value/1000:.1f} kΩ"

        else:
            value_text = f"{value} Ω"

        return f"Resistance: {value_text} | Tolerance: {tol}"

    except:
        return "Calculation Failed"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/app", methods=["GET","POST"])
def index():

    image_path = None
    detected = []
    answer = ""

    if request.method == "POST":

        if "bands" in request.form:

            bands = request.form.getlist("bands")

            image_path = request.form["image_path"]

            answer = calculate_resistance(bands)

            detected = bands

        else:

            file = request.files["image"]

            filename = secure_filename(file.filename)

            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(save_path)

            image_path = save_path

            detected = detect_bands(save_path)

    return render_template(
        "index.html",
        image_path=image_path,
        detected=detected,
        answer=answer,
        all_colors=all_colors
    )

if __name__ == "__main__":
    app.run(debug=True)