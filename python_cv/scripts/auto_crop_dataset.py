"""
Auto-crop dataset từ video.
Usage:
  python scripts/auto_crop_dataset.py --video path/to/video.mp4 --id NV001
  python scripts/auto_crop_dataset.py --video path/to/video.mp4 --id NV001 --skip 3
"""
import argparse
import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from config.settings import DATASET_CROPS_ROOT, YOLO_WEIGHTS

def blur_score(img):
    import cv2 as _cv2
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    return _cv2.Laplacian(gray, _cv2.CV_64F).var()

def auto_crop(video_path: str, person_id: str, skip: int = 3,
              min_blur: float = 80.0, conf: float = 0.65):
    output_dir = DATASET_CROPS_ROOT / person_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(YOLO_WEIGHTS))
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"[INFO] Output: {output_dir}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Bỏ qua frame theo skip interval
        if frame_count % skip != 0:
            continue

        results = model(frame, classes=[0], conf=conf, verbose=False)[0]

        for j, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h_box = y2 - y1
            w_box = x2 - x1

            # Bỏ box quá nhỏ
            if h_box < 64 or w_box < 32:
                continue

            # Thêm padding 10%
            pad_h = int(0.1 * h_box)
            pad_w = int(0.1 * w_box)
            y1p = max(0, y1 - pad_h)
            y2p = min(frame.shape[0], y2 + pad_h)
            x1p = max(0, x1 - pad_w)
            x2p = min(frame.shape[1], x2 + pad_w)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            # Filter blur
            if blur_score(crop) < min_blur:
                continue

            # Resize về chuẩn Re-ID
            crop_resized = cv2.resize(crop, (128, 256))

            # Lưu
            filename = f"{person_id}_f{frame_count:06d}_{j}.jpg"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), crop_resized)
            saved_count += 1

        # Progress
        if frame_count % 300 == 0:
            pct = frame_count / total_frames * 100
            print(f"  Progress: {pct:.1f}% | Saved: {saved_count} crops")

    cap.release()
    print(f"\n[DONE] Saved {saved_count} crops to {output_dir}")
    print(f"[INFO] From {frame_count} frames (every {skip} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--id",    required=True, help="Person ID (e.g. NV001)")
    parser.add_argument("--skip",  type=int, default=3,   help="Process every N frames (default=3)")
    parser.add_argument("--blur",  type=float, default=80.0, help="Min blur score (default=80)")
    parser.add_argument("--conf",  type=float, default=0.65, help="YOLO confidence (default=0.65)")
    args = parser.parse_args()

    auto_crop(
        video_path=args.video,
        person_id=args.id,
        skip=args.skip,
        min_blur=args.blur,
        conf=args.conf,
    )
