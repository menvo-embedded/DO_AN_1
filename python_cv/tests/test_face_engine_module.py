import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.face_insightface_engine import InsightFaceEngine
from config.settings import CAM_ZONE2_RTSP


engine = InsightFaceEngine(
    gallery_path="data/face_gallery/insightface_gallery.pkl",
    model_name="buffalo_sc",
    use_gpu=True,
    det_size=(640, 640),
    det_thresh=0.50,
    face_threshold=0.38,
    face_margin=0.09,
    topk_mean=5,
)

cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Cannot open Zone 2 camera")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    result = engine.identify_image(frame)

    print(
        result["status"],
        result["employee_id"],
        result["name"],
        "score=", round(result["score"], 3),
        "second=", round(result["second_score"], 3),
        "margin=", round(result["margin"], 3),
    )

    bbox = result.get("bbox")

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        color = (0, 255, 0) if result["status"] == "MATCH" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{result['employee_id']} {result['score']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imshow("Face Engine Module Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord("q"), ord("Q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
