"""
Vehicle Monitoring & Speed Detection System
============================================
Features:
  - Speed detection (pixels/frame → km/h via calibration)
  - Vehicle counting & classification (car, truck, bus, motorbike)
  - License plate recognition (OCR via pytesseract)
  - Live annotated video output + CSV log

Requirements:
    pip install opencv-python opencv-contrib-python numpy pytesseract imutils
    Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
"""

import cv2
import numpy as np
import csv
import time
import os
from collections import defaultdict
from datetime import datetime

# ── Try optional imports ──────────────────────────────────────────────────────
try:
    import pytesseract
    OCR_AVAILABLE = True
    # Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract not installed – license plate OCR disabled.")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Input: 0 = webcam, or path to video file
    "source": "traffic.mp4",

    # Output
    "output_video": "output_annotated.avi",
    "output_csv":   "detections.csv",
    "save_plates":  True,          # Save cropped plate images
    "plates_dir":   "plates",

    # Speed estimation
    # Real-world distance (metres) covered by the ROI height in pixels
    "roi_real_metres": 15.0,       # calibrate for your scene
    "fps_override": None,          # set to e.g. 30 if auto-detect fails

    # Detection lines (fraction of frame height)
    "line1_frac": 0.40,            # upper counting/speed line
    "line2_frac": 0.70,            # lower counting/speed line

    # YOLO / Haar cascade paths  (edit to your local paths)
    "yolo_weights": "yolov4-tiny.weights",
    "yolo_cfg":     "yolov4-tiny.cfg",
    "coco_names":   "coco.names",

    # Fallback: use MOG2 background subtraction if YOLO files missing
    "fallback_bgsub": True,

    # YOLO inference
    "conf_threshold": 0.45,
    "nms_threshold":  0.40,
    "input_size":     (416, 416),

    # Tracking
    "max_disappeared": 30,         # frames before track removed
    "max_distance":    80,         # px – max centroid jump between frames
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# Colours per class
CLASS_COLORS = {
    "car":       (255, 200,  50),
    "truck":     ( 50, 200, 255),
    "bus":       ( 50, 255, 150),
    "motorbike": (255,  50, 200),
    "bicycle":   (200, 255,  50),
    "unknown":   (180, 180, 180),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CENTROID TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.objects   = {}   # id → centroid
        self.bboxes    = {}   # id → (x,y,w,h)
        self.labels    = {}   # id → class label
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid, bbox, label):
        self.objects[self.next_id]   = centroid
        self.bboxes[self.next_id]    = bbox
        self.labels[self.next_id]    = label
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        for d in (self.objects, self.bboxes, self.labels, self.disappeared):
            d.pop(oid, None)

    def update(self, detections):
        """detections: list of (centroid, bbox, label)"""
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.bboxes, self.labels

        if not self.objects:
            for c, b, l in detections:
                self.register(c, b, l)
            return self.objects, self.bboxes, self.labels

        obj_ids   = list(self.objects.keys())
        obj_cents = np.array(list(self.objects.values()))
        det_cents = np.array([d[0] for d in detections])

        # Distance matrix
        D = np.linalg.norm(obj_cents[:, None] - det_cents[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            self.objects[oid]     = detections[c][0]
            self.bboxes[oid]      = detections[c][1]
            self.labels[oid]      = detections[c][2]
            self.disappeared[oid] = 0
            used_rows.add(r); used_cols.add(c)

        unused_rows = set(range(len(obj_ids))) - used_rows
        unused_cols = set(range(len(detections))) - used_cols

        for r in unused_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in unused_cols:
            self.register(*detections[c])

        return self.objects, self.bboxes, self.labels

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEED ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SpeedEstimator:
    def __init__(self, line1_y, line2_y, real_metres, fps):
        self.line1_y     = line1_y
        self.line2_y     = line2_y
        self.real_metres = real_metres
        self.fps         = fps
        self.pix_dist    = abs(line2_y - line1_y)
        self.mpp         = real_metres / self.pix_dist   # metres per pixel

        self._cross1 = {}   # id → frame number when crossing line1
        self._cross2 = {}
        self.speeds  = {}   # id → km/h

    def update(self, frame_no, objects):
        for oid, (cx, cy) in objects.items():
            # Line 1 crossing
            if oid not in self._cross1 and abs(cy - self.line1_y) < 8:
                self._cross1[oid] = frame_no
            # Line 2 crossing
            if oid in self._cross1 and oid not in self._cross2 and abs(cy - self.line2_y) < 8:
                self._cross2[oid] = frame_no
                elapsed_frames = self._cross2[oid] - self._cross1[oid]
                if elapsed_frames > 0:
                    elapsed_sec = elapsed_frames / self.fps
                    speed_mps   = self.real_metres / elapsed_sec
                    speed_kmh   = speed_mps * 3.6
                    self.speeds[oid] = round(speed_kmh, 1)

        return self.speeds

# ═══════════════════════════════════════════════════════════════════════════════
#  LICENSE PLATE READER
# ═══════════════════════════════════════════════════════════════════════════════

def load_plate_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    return None

def read_plate(frame, bbox, plate_cascade, save_dir=None, vehicle_id=None):
    """Detect & OCR license plate within a vehicle bounding box."""
    x, y, w, h = bbox
    roi = frame[max(0,y):y+h, max(0,x):x+w]
    if roi.size == 0:
        return None, None

    plate_text = None
    plate_img  = None

    if plate_cascade is not None:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 15))
        for (px, py, pw, ph) in plates:
            plate_img = roi[py:py+ph, px:px+pw]
            if OCR_AVAILABLE and plate_img.size > 0:
                plate_text = ocr_plate(plate_img)
            break  # take first detected plate

    if save_dir and plate_img is not None and vehicle_id is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"plate_{vehicle_id}_{int(time.time())}.jpg")
        cv2.imwrite(fname, plate_img)

    return plate_text, plate_img

def ocr_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    text = ''.join(c for c in text if c.isalnum())
    return text if len(text) >= 4 else None

# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolo(weights, cfg, names):
    net    = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names) as f:
        classes = [l.strip() for l in f.readlines()]
    out_layers = [net.getLayerNames()[i - 1]
                  for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, out_layers

def detect_yolo(frame, net, classes, out_layers, conf_thr, nms_thr, inp_size):
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, inp_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid    = int(np.argmax(scores))
            conf   = float(scores[cid])
            if conf < conf_thr:
                continue
            label = classes[cid] if cid < len(classes) else "unknown"
            if label not in VEHICLE_CLASSES:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw/2) * W)
            y = int((cy - bh/2) * H)
            boxes.append([x, y, int(bw*W), int(bh*H)])
            confidences.append(conf)
            class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]] if class_ids[i] < len(classes) else "unknown"
            cx, cy = x + w//2, y + h//2
            results.append(((cx, cy), (x, y, w, h), label))
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND SUBTRACTION FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bgsub(frame, fgmask):
    results = []
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask    = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2
        # Rough size-based classification
        if w > 120 and h > 60:
            label = "truck"
        elif w > 80:
            label = "car"
        else:
            label = "motorbike"
        results.append(((cx, cy), (x, y, w, h), label))
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  HUD DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, line1_y, line2_y, counts, fps_display):
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # Detection lines
    cv2.line(frame, (0, line1_y), (W, line1_y), (0, 255, 255), 2)
    cv2.line(frame, (0, line2_y), (W, line2_y), (0, 200, 255), 2)
    cv2.putText(frame, "SPEED LINE 1", (10, line1_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "SPEED LINE 2", (10, line2_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Stats panel (top-left)
    panel_h = 30 + len(counts) * 22 + 30
    cv2.rectangle(overlay, (0, 0), (220, panel_h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "VEHICLE MONITOR", (8, 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 220, 50), 1)
    y_off = 42
    for label, cnt in sorted(counts.items()):
        color = CLASS_COLORS.get(label, (180,180,180))
        cv2.putText(frame, f"{label.upper():12s}: {cnt}", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        y_off += 22

    total = sum(counts.values())
    cv2.putText(frame, f"TOTAL : {total}", (10, y_off + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # FPS top-right
    cv2.putText(frame, f"FPS {fps_display:.1f}", (W - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

    # Timestamp bottom-left
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def draw_vehicle(frame, oid, bbox, label, speed, plate_text):
    x, y, w, h = bbox
    color = CLASS_COLORS.get(label, (180, 180, 180))
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    info = f"ID:{oid} {label}"
    if speed:
        info += f"  {speed} km/h"
    if plate_text:
        info += f"  [{plate_text}]"

    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, info, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    cap = cv2.VideoCapture(cfg["source"])
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {cfg['source']}")
        return

    fps = cfg["fps_override"] or cap.get(cv2.CAP_PROP_FPS) or 25
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {cfg['source']}  |  {W}×{H} @ {fps:.1f} fps")

    line1_y = int(H * cfg["line1_frac"])
    line2_y = int(H * cfg["line2_frac"])

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(cfg["output_video"], fourcc, fps, (W, H))

    # CSV
    csv_file = open(cfg["output_csv"], "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "vehicle_id", "label", "speed_kmh", "plate"])

    # Detector
    use_yolo = False
    if os.path.exists(cfg["yolo_weights"]) and os.path.exists(cfg["yolo_cfg"]) \
            and os.path.exists(cfg["coco_names"]):
        print("[INFO] Loading YOLO …")
        net, classes, out_layers = load_yolo(
            cfg["yolo_weights"], cfg["yolo_cfg"], cfg["coco_names"])
        use_yolo = True
        print("[INFO] YOLO loaded.")
    elif cfg["fallback_bgsub"]:
        print("[WARN] YOLO files not found – using background subtraction fallback.")
        bgsub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
    else:
        print("[ERROR] No detector available."); return

    # Plate cascade
    plate_cascade = load_plate_cascade()

    tracker  = CentroidTracker(cfg["max_disappeared"], cfg["max_distance"])
    estimator = SpeedEstimator(line1_y, line2_y, cfg["roi_real_metres"], fps)

    counts      = defaultdict(int)
    counted_ids = set()
    plate_cache = {}   # id → plate text
    frame_no    = 0
    t_prev      = time.time()
    fps_display = fps

    print("[INFO] Processing … press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # ── Detect ──────────────────────────────────────────────
        if use_yolo:
            detections = detect_yolo(frame, net, classes, out_layers,
                                     cfg["conf_threshold"], cfg["nms_threshold"],
                                     cfg["input_size"])
        else:
            fgmask     = bgsub.apply(frame)
            detections = detect_bgsub(frame, fgmask)

        # ── Track ───────────────────────────────────────────────
        objects, bboxes, labels = tracker.update(detections)

        # ── Speed ───────────────────────────────────────────────
        speeds = estimator.update(frame_no, objects)

        # ── Draw & log ──────────────────────────────────────────
        for oid, centroid in objects.items():
            bbox  = bboxes.get(oid, (0,0,0,0))
            label = labels.get(oid, "unknown")
            speed = speeds.get(oid)

            # Count each vehicle once
            if oid not in counted_ids:
                counted_ids.add(oid)
                counts[label] += 1

            # Plate (run every 15 frames to save CPU)
            if frame_no % 15 == 0 and oid not in plate_cache:
                ptext, _ = read_plate(frame, bbox, plate_cascade,
                                       cfg["plates_dir"] if cfg["save_plates"] else None,
                                       oid)
                if ptext:
                    plate_cache[oid] = ptext

            plate_text = plate_cache.get(oid)

            draw_vehicle(frame, oid, bbox, label, speed, plate_text)

            # CSV row when speed is first calculated
            if speed and oid not in plate_cache.get("_logged", set()):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([ts, oid, label, speed, plate_text or ""])
                csv_file.flush()

        # ── HUD ─────────────────────────────────────────────────
        t_now = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        draw_hud(frame, line1_y, line2_y, counts, fps_display)

        writer.write(frame)
        cv2.imshow("Vehicle Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User quit.")
            break

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"[DONE] Output saved: {cfg['output_video']} | {cfg['output_csv']}")
    print(f"[DONE] Vehicle counts: {dict(counts)}")


if __name__ == "__main__":
    main()
