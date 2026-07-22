import cv2
import sys
import time
import os
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO
from config.settings import DATASET_CROPS_ROOT, YOLO_WEIGHTS, CAM_ZONE1_INDEX, CAM_ZONE2_RTSP, CAM_ZONE3_RTSP

def crop_from_source(zone, duration=21600):
    output_dir = DATASET_CROPS_ROOT / "unknown"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_WEIGHTS))

    if zone == 1:
        cap = cv2.VideoCapture(CAM_ZONE1_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[INFO] Zone 1 live crop started")
    elif zone == 2:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[INFO] Zone 2 live crop started")
    else:  # zone == 3
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(CAM_ZONE3_RTSP, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[INFO] Zone 3 live crop started")

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    start_time  = time.time()
    frame_count = 0
    saved_count = 0
    SKIP        = 25

    print(f"[INFO] Saving to: {output_dir}")
    print("[INFO] q=stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if zone == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1
        elapsed = time.time() - start_time

        # Display
        display = cv2.resize(frame.copy(), (640, 480))
        cv2.putText(display, f"Zone{zone} | Crops: {saved_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Time: {elapsed:.0f}s / {duration}s",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(f"Live Crop Zone{zone}", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if elapsed >= duration:
            break

        # Crop moi 1 giay
        if frame_count % SKIP != 0:
            continue

        results = model(frame, classes=[0], conf=0.25, verbose=False)[0]
        for j, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h_box, w_box = y2 - y1, x2 - x1
            if h_box < 64 or w_box < 32:
                continue
            pad_h = int(0.1 * h_box)
            pad_w = int(0.1 * w_box)
            y1p = max(0, y1 - pad_h)
            y2p = min(frame.shape[0], y2 + pad_h)
            x1p = max(0, x1 - pad_w)
            x2p = min(frame.shape[1], x2 + pad_w)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
                continue
            crop_resized = cv2.resize(crop, (128, 256))
            ts = datetime.now().strftime("%H%M%S_%f")
            save_path = output_dir / f"z{zone}_{ts}_{j}.jpg"
            cv2.imwrite(str(save_path), crop_resized)
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved_count} crops to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone",     type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--duration", type=int, default=21600)
    args = parser.parse_args()
    crop_from_source(args.zone, args.duration)
