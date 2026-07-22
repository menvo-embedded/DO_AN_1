import argparse
import cv2
import sys
import time
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CAM_ZONE1_INDEX, CAM_ZONE2_RTSP, CAM_ZONE3_RTSP, WAREHOUSE_DATASET_ROOT

# RTSP chat luong cao (subtype=0)
CAM_ZONE2_HQ = CAM_ZONE2_RTSP.replace("subtype=1", "subtype=0")
CAM_ZONE3_HQ = CAM_ZONE3_RTSP.replace("subtype=1", "subtype=0")

def record(zone, person_id, duration=600, hq=True):
    output_dir = WAREHOUSE_DATASET_ROOT / "dataset_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{person_id}_zone{zone}_{ts}.mp4"

    if zone == 1:
        cap = cv2.VideoCapture(CAM_ZONE1_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[INFO] Recording Zone 1 (index={CAM_ZONE1_INDEX})")
    elif zone == 2:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        rtsp = CAM_ZONE2_HQ if hq else CAM_ZONE2_RTSP
        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        print(f"[INFO] Recording Zone 2 RTSP ({'HQ subtype=0' if hq else 'LQ subtype=1'})")
    else:  # zone == 3
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        rtsp = CAM_ZONE3_HQ if hq else CAM_ZONE3_RTSP
        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        print(f"[INFO] Recording Zone 3 RTSP ({'HQ subtype=0' if hq else 'LQ subtype=1'})")

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if zone == 1:
        w, h = h, w

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] {duration}s | {w}x{h} @ {fps}fps")
    print("[INFO] q=stop, s=snapshot")

    start_time  = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if zone == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        writer.write(frame)
        frame_count += 1

        elapsed = time.time() - start_time
        remain  = max(0, duration - elapsed)
        display = cv2.resize(frame, (640, 480))

        cv2.putText(display, f"REC Zone{zone} | {person_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, f"Time: {elapsed:.0f}s / {duration}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Remain: {remain:.0f}s",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frames: {frame_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        bar_w = int(640 * min(1.0, elapsed / duration))
        cv2.rectangle(display, (0, 470), (bar_w, 480), (0, 255, 0), -1)
        cv2.imshow(f"Recording Zone{zone} - {person_id}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Stopped")
            break
        if key == ord("s"):
            snap = output_dir / f"{person_id}_snap_{frame_count}.jpg"
            cv2.imwrite(str(snap), frame)
            print(f"[INFO] Snapshot: {snap}")
        if elapsed >= duration:
            print("[INFO] Done")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[DONE] {frame_count} frames saved")
    print(f"[FILE] {output_path} ({size_mb:.1f}MB)")
    print(f"[NEXT] python scripts/auto_crop_dataset.py --video {output_path} --id {person_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone",     type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--id",       type=str, required=True)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--lq",       action="store_true", help="Dung chat luong thap (subtype=1)")
    args = parser.parse_args()
    record(args.zone, args.id, args.duration, hq=not args.lq)
