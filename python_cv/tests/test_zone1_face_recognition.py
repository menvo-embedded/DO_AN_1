import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import face_recognition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import CAM_ZONE1_INDEX, DEBUG_DIR


# ============================================================
# CONFIG
# ============================================================

CAM_INDEX = CAM_ZONE1_INDEX
OUTPUT_DIR = DEBUG_DIR / "face_recognition_zone1_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "Zone 1 - face_recognition HOG Test"

# HOG: nhẹ, chạy CPU, phù hợp test nhanh
DETECT_MODEL = "hog"

# Resize nhỏ để tăng tốc. Nếu mặt khó detect, đổi SCALE = 1.0
SCALE = 0.75

SAVE_COOLDOWN_SEC = 2.0
AUTO_SAVE = True


# ============================================================
# HELPERS
# ============================================================

def save_face(frame_bgr, face_location, encoding, prefix="auto"):
    """
    face_location format của face_recognition:
    top, right, bottom, left
    """
    top, right, bottom, left = face_location

    h, w = frame_bgr.shape[:2]

    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - 1))
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w - 1))

    if right <= left or bottom <= top:
        print("Invalid bbox, skip save.")
        return

    face_crop = frame_bgr[top:bottom, left:right].copy()

    annotated = frame_bgr.copy()
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(
        annotated,
        "face_recognition HOG",
        (left, max(25, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{prefix}_face_{ts}"

    full_path = OUTPUT_DIR / f"{base}_full.jpg"
    crop_path = OUTPUT_DIR / f"{base}_crop.jpg"
    emb_path = OUTPUT_DIR / f"{base}_encoding.npy"

    cv2.imwrite(str(full_path), annotated)
    cv2.imwrite(str(crop_path), face_crop)

    encoding = np.asarray(encoding, dtype=np.float32)
    np.save(str(emb_path), encoding)

    print(f"SAVED: {base}")
    print(f"  bbox: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"  encoding_dim: {len(encoding)}")
    print(f"  full: {full_path}")
    print(f"  crop: {crop_path}")
    print(f"  npy : {emb_path}")


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("ZONE 1 FACE_RECOGNITION HOG TEST")
print("=" * 70)
print("CAM_INDEX :", CAM_INDEX)
print("MODEL     :", DETECT_MODEL)
print("SCALE     :", SCALE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Press Q/ESC to quit.")
print("Press S to save manually.")
print("=" * 70)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera index={CAM_INDEX}")
    raise SystemExit

last_save_time = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("WARN: Cannot read frame.")
        time.sleep(0.1)
        continue

    display = frame.copy()

    # Resize để detect nhanh hơn
    if SCALE != 1.0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    else:
        small = frame

    # face_recognition dùng RGB
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations_small = face_recognition.face_locations(
        rgb_small,
        model=DETECT_MODEL
    )

    # Scale bbox về ảnh gốc
    locations = []
    for top, right, bottom, left in locations_small:
        if SCALE != 1.0:
            top = int(top / SCALE)
            right = int(right / SCALE)
            bottom = int(bottom / SCALE)
            left = int(left / SCALE)

        locations.append((top, right, bottom, left))

    encodings = []
    if locations:
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb_full,
            known_face_locations=locations,
            num_jitters=1
        )

    # Vẽ bbox
    for i, loc in enumerate(locations):
        top, right, bottom, left = loc

        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Face {i+1}",
            (left, max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if locations:
        cv2.putText(
            display,
            f"Faces: {len(locations)} | Encodings: {len(encodings)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        now = time.time()

        if AUTO_SAVE and encodings and (now - last_save_time >= SAVE_COOLDOWN_SEC):
            save_face(frame, locations[0], encodings[0], prefix="auto")
            saved_count += 1
            last_save_time = now

    else:
        cv2.putText(
            display,
            "No face detected - move closer / face camera / improve light",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        display,
        "Q/ESC: quit | S: save",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key == ord("s"):
        if locations and encodings:
            save_face(frame, locations[0], encodings[0], prefix="manual")
            saved_count += 1
        else:
            print("No face / encoding to save.")

cap.release()
cv2.destroyAllWindows()

print("=" * 70)
print("DONE")
print("Saved count:", saved_count)
print("Output dir :", OUTPUT_DIR)
print("=" * 70)
