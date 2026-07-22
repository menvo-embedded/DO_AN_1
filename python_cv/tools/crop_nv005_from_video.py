import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATASET_CROPS_ROOT

VIDEO_PATH = Path(r"D:\VID20260503225026.mp4")
OUT_DIR = DATASET_CROPS_ROOT / "NV005_city_20260503"

OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = Path("yolo11n.pt")
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = ROOT_DIR.parent / "yolo11n.pt"

model = YOLO(str(YOLO_WEIGHTS))

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print("Cannot open video:", VIDEO_PATH)
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

print("=" * 80)
print("OFFLINE CROP NV005")
print("=" * 80)
print("Video      :", VIDEO_PATH)
print("Resolution :", f"{width}x{height}")
print("FPS        :", fps)
print("Frames     :", total_frames)
print("Duration   :", round(duration, 2), "seconds")
print("Output     :", OUT_DIR)
print("=" * 80)

# Take one frame every 0.5 seconds to avoid near-duplicate crops.
FRAME_STEP = max(1, int(fps * 0.5))

saved = 0
processed = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_STEP != 0:
        frame_idx += 1
        continue

    processed += 1

    results = model(frame, classes=[0], conf=0.45, verbose=False)[0]

    best_box = None
    best_area = 0

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area = box_w * box_h

            if box_h < 250 or box_w < 80:
                continue

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            out_file = OUT_DIR / f"NV005_city_f{frame_idx:06d}_{saved:05d}.jpg"
            cv2.imwrite(str(out_file), crop)
            saved += 1

    if processed % 100 == 0:
        print(f"Processed sampled frames: {processed} | saved crops: {saved}")

    frame_idx += 1

cap.release()

print("=" * 80)
print("DONE")
print("Processed sampled frames:", processed)
print("Saved crops:", saved)
print("Output:", OUT_DIR)
print("=" * 80)
