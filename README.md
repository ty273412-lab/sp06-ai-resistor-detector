# ResistorScan — SP06 AI Resistor Detector

A Flask web application that detects resistor color bands using OpenCV HSV analysis and calculates resistance values.

## Project Structure

```
resistor_app/
├── app.py                   # Flask backend (improved)
├── templates/
│   ├── home.html            # Landing page
│   └── index.html           # Main detector app
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/             # Auto-created on first run
└── README.md
```

## Setup

```bash
pip install flask opencv-python numpy werkzeug
python app.py
```

App runs at: http://localhost:10000

## Key Improvements Over Original

### Backend (app.py)
- **5-band resistor support** (3 digit bands + multiplier + tolerance)
- **Better HSV detection**: morphological cleanup (MORPH_CLOSE + MORPH_OPEN) removes noise
- **Red hue wrapping**: Red occupies both ends of the HSV hue circle; added `red2` range (165–180)
- **Smarter band merging**: nearby detections within 35px are merged instead of duplicated
- **Aspect ratio filtering**: ensures detected contours are band-shaped (tall + narrow)
- **ROI cropping**: strips background noise by cropping center 70% of image
- **Structured result dict**: `calculate_resistance()` returns a dict (success, value_text, tolerance, ohms, band_count) instead of a plain string
- **API endpoint**: `/api/calculate` accepts JSON for AJAX use
- **File validation**: checks allowed extensions before saving
- **GΩ range**: formats extremely high resistance values correctly

### Frontend
- **Dark theme** with grid background and glowing orbs
- **Drag-and-drop** image upload with instant thumbnail preview
- **Resistor body visualisation** showing detected bands with correct colors
- **Live color indicators** under each dropdown that update on change
- **Responsive layout**: adapts to mobile
- **Landing page** with color band reference chart and methodology overview
- **Structured result display** with separate resistance and tolerance metrics
- **Sticky navigation** with breadcrumb

## How Detection Works

1. Image is resized to 700×250 px for a consistent working size
2. Centre ROI (15–85% height, 5–95% width) is cropped to remove background
3. Image converted to HSV color space
4. Per-color HSV masks applied with Gaussian blur pre-processing
5. Morphological operations clean up noise (close small gaps, open isolated pixels)
6. Contours filtered by area (>900 px²), height (>50 px), and aspect ratio (>0.8)
7. Valid contours sorted by x-position, merged if closer than 35 px
8. Up to 5 unique colors returned in left-to-right order

## Color Band Formula

**4-band**: `(D1×10 + D2) × Multiplier` | Tolerance = Band 4  
**5-band**: `(D1×100 + D2×10 + D3) × Multiplier` | Tolerance = Band 5

