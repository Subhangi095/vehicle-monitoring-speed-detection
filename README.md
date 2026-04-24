# Vehicle Monitoring & Speed Detection System
## Setup & Usage Guide

---

## 1. Install Python Dependencies

```bash
pip install opencv-python opencv-contrib-python numpy pytesseract imutils
```

---

## 2. Install Tesseract OCR (for License Plate Reading)

| OS      | Command / Link |
|---------|---------------|
| Windows | Download installer: https://github.com/UB-Mannheim/tesseract/wiki |
| Ubuntu  | `sudo apt install tesseract-ocr` |
| macOS   | `brew install tesseract` |

**Windows users:** After installing, uncomment and set this line in `vehicle_monitor.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 3. Download YOLO Files (Recommended for Best Detection)

```bash
# YOLOv4-tiny (fast, good accuracy)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

Place all three files in the **same folder** as `vehicle_monitor.py`.

> **No YOLO?** The script automatically falls back to background subtraction — no extra files needed, but accuracy is lower.

---

## 4. Configure the Script

Edit the `CONFIG` dictionary at the top of `vehicle_monitor.py`:

```python
CONFIG = {
    "source": "traffic.mp4",     # ← your video file, or 0 for webcam
    "roi_real_metres": 15.0,     # ← real-world distance between the two detection lines
    "line1_frac": 0.40,          # ← upper line position (40% down the frame)
    "line2_frac": 0.70,          # ← lower line position (70% down the frame)
    ...
}
```

### Speed Calibration
Measure the real-world distance (in metres) between the two yellow lines in your scene.
Set `roi_real_metres` to that value.  The more accurate this is, the more accurate your speed readings.

---

## 5. Run

```bash
python vehicle_monitor.py
```

Press **Q** to stop.

---

## 6. Outputs

| File | Description |
|------|-------------|
| `output_annotated.avi` | Video with bounding boxes, labels, speeds, plate text |
| `detections.csv` | Log of every detected vehicle: timestamp, ID, class, speed, plate |
| `plates/` | Cropped license plate images (if `save_plates: True`) |

---

## 7. How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Video Frame │────▶│  YOLO / BGSub │────▶│ Centroid Tracker │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                          ┌─────────────────────────┼──────────────────┐
                          │                         │                  │
                   ┌──────▼──────┐          ┌───────▼──────┐  ┌───────▼──────┐
                   │   Speed     │          │   Vehicle    │  │   License    │
                   │  Estimator  │          │   Counter    │  │   Plate OCR  │
                   └──────┬──────┘          └───────┬──────┘  └───────┬──────┘
                          │                         │                  │
                          └─────────────────────────▼──────────────────┘
                                              Annotated Frame + CSV Log
```

### Speed Detection
- Two horizontal lines drawn across the frame.
- When a tracked vehicle crosses line 1, a timer starts.
- When it crosses line 2, elapsed time is measured.
- Speed = real-world distance ÷ elapsed time (converted to km/h).

### Vehicle Classification
- YOLO identifies: **car, truck, bus, motorbike, bicycle**.
- Fallback BGSub uses bounding box size heuristics.

### License Plate Recognition
- Haar cascade detects plate region within the vehicle bounding box.
- Tesseract OCR reads alphanumeric characters.
- Plates are saved as images and logged in the CSV.

---

## 8. Tips

- **Better accuracy**: Use a camera mounted at an angle (not top-down) so vehicles are visible long enough to cross both lines.
- **GPU acceleration**: Install OpenCV with CUDA support and set `DNN_TARGET_CUDA` in the script.
- **Webcam**: Set `"source": 0` for live camera feed.
- **Night footage**: Pre-process with CLAHE contrast enhancement before detection.
