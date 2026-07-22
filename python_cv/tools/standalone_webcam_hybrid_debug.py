import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    CAM_ZONE2_INDEX,
    FACE_MATCH_MARGIN,
    FACE_MATCH_THRESHOLD,
    REID_MATCH_MARGIN,
    REID_MATCH_THRESHOLD,
    YOLO_CLASSES,
    YOLO_CONF,
    YOLO_IMG_SIZE,
    YOLO_IOU,
    YOLO_WEIGHTS,
)
from reid.face_insightface_engine import InsightFaceEngine
from reid.gallery import Gallery
from reid.reid_engine import ReIDEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone webcam debug for Body Re-ID + InsightFace. "
            "This does not import or run main.py, CameraZone2, or FusionLayer."
        )
    )
    parser.add_argument("--camera-index", type=int, default=CAM_ZONE2_INDEX)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--save-every", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMG_SIZE)
    parser.add_argument("--iou", type=float, default=YOLO_IOU)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "standalone_webcam_hybrid_debug",
    )
    return parser.parse_args()


def open_webcam(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index={index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def rank_body(reid: ReIDEngine, gallery_data: dict, crop):
    embedding = reid.get_embedding(crop)
    if embedding is None:
        return []

    rows = []
    for emp_id, embeds in gallery_data.items():
        score = reid.match_score(embedding, embeds)
        rows.append((emp_id, score))

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def body_decision(ranking):
    if not ranking:
        return "UNKNOWN", "", 0.0, "", 0.0, 0.0

    best_id, best_score = ranking[0]
    second_id, second_score = ranking[1] if len(ranking) > 1 else ("", 0.0)
    margin = best_score - second_score

    if best_score >= REID_MATCH_THRESHOLD and margin >= REID_MATCH_MARGIN:
        decision = f"MATCH:{best_id}"
    else:
        decision = "UNKNOWN"

    return decision, best_id, best_score, second_id, second_score, margin


def face_decision(face_result: dict):
    status = face_result.get("status", "UNKNOWN")
    ranking = face_result.get("ranking", [])

    if status == "MATCH":
        best_id = face_result.get("employee_id", "Unknown")
        decision = f"MATCH:{best_id}"
    else:
        best_id = ranking[0].get("employee_id", "Unknown") if ranking else "Unknown"
        decision = status

    return (
        decision,
        best_id,
        float(face_result.get("score", 0.0)),
        float(face_result.get("second_score", -1.0)),
        float(face_result.get("margin", 0.0)),
        face_result.get("bbox"),
        face_result.get("det_score", ""),
        ranking,
    )


def final_decision(body_best_id: str, body_score: float, body_margin: float, face_result: dict | None):
    if face_result is None:
        if body_score >= REID_MATCH_THRESHOLD and body_margin >= REID_MATCH_MARGIN:
            return body_best_id, "body_match_no_face_engine"
        return "Unknown", "body_unknown_no_face_engine"

    face_status = face_result.get("status", "UNKNOWN")
    face_id = face_result.get("employee_id", "Unknown")

    if face_status == "MATCH":
        if body_best_id and body_best_id != face_id:
            return face_id, f"face_overrides_body_conflict:{body_best_id}"
        return face_id, "face_match"

    if face_status == "NO_FACE":
        if body_score >= REID_MATCH_THRESHOLD and body_margin >= REID_MATCH_MARGIN:
            return body_best_id, "no_face_body_match"
        return "Unknown", "no_face_body_unknown"

    # UNKNOWN/NO_GALLERY: do not trust normal body threshold; this is only a debug policy.
    if body_score >= 0.95 and body_margin >= 0.035:
        return body_best_id, "face_unknown_strict_body_match"

    return "Unknown", f"face_{face_status.lower()}_blocks_weak_body"


def draw_label(image, text, x, y, color):
    cv2.rectangle(image, (x, max(0, y - 22)), (x + min(520, 9 * len(text)), y), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_id
    crops_dir = out_dir / "crops"
    frames_dir = out_dir / "frames"
    annotated_dir = out_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "webcam_hybrid_debug.csv"

    print("=" * 100)
    print("STANDALONE WEBCAM HYBRID DEBUG")
    print("=" * 100)
    print("This tool does not run main.py, CameraZone2, or FusionLayer.")
    print("Camera index:", args.camera_index)
    print("Output:", out_dir)
    print("Body threshold/margin:", REID_MATCH_THRESHOLD, REID_MATCH_MARGIN)
    print("Face threshold/margin:", FACE_MATCH_THRESHOLD, FACE_MATCH_MARGIN)

    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    yolo = YOLO(str(YOLO_WEIGHTS))
    reid = ReIDEngine()
    gallery_data = Gallery().all()
    face_engine = None
    if not args.no_face:
        face_engine = InsightFaceEngine(
            gallery_path=str(ROOT_DIR / "data" / "face_gallery" / "insightface_gallery.pkl"),
            model_name="buffalo_sc",
            use_gpu=True,
            face_threshold=FACE_MATCH_THRESHOLD,
            face_margin=FACE_MATCH_MARGIN,
        )

    cap = open_webcam(args.camera_index, args.width, args.height)

    fieldnames = [
        "timestamp",
        "frame_index",
        "detection_index",
        "crop_path",
        "frame_path",
        "annotated_path",
        "bbox",
        "bbox_width",
        "bbox_height",
        "bbox_aspect",
        "body_decision",
        "body_best_id",
        "body_score",
        "body_second_id",
        "body_second_score",
        "body_margin",
        "body_ranking",
        "face_decision",
        "face_best_id",
        "face_score",
        "face_second_score",
        "face_margin",
        "face_det_score",
        "face_bbox",
        "face_ranking",
        "final_id",
        "final_reason",
    ]

    start = time.time()
    last_saved = 0.0
    frame_index = 0
    saved_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame_index += 1
            now = time.time()
            if now - last_saved < args.save_every:
                if args.display:
                    cv2.imshow("standalone webcam hybrid debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            results = yolo(
                frame,
                classes=YOLO_CLASSES,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                if args.display:
                    cv2.imshow("standalone webcam hybrid debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = frames_dir / f"frame_{ts}.jpg"
            annotated = frame.copy()
            cv2.imwrite(str(frame_path), frame)

            frame_h, frame_w = frame.shape[:2]
            for det_index, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w))
                y2 = max(0, min(y2, frame_h))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                body_rank = rank_body(reid, gallery_data, crop)
                (
                    b_decision,
                    b_best_id,
                    b_score,
                    b_second_id,
                    b_second_score,
                    b_margin,
                ) = body_decision(body_rank)

                face_result = face_engine.identify_image(crop) if face_engine is not None else None
                if face_result is not None:
                    (
                        f_decision,
                        f_best_id,
                        f_score,
                        f_second_score,
                        f_margin,
                        f_bbox,
                        f_det_score,
                        face_rank,
                    ) = face_decision(face_result)
                else:
                    f_decision = "DISABLED"
                    f_best_id = "Unknown"
                    f_score = 0.0
                    f_second_score = -1.0
                    f_margin = 0.0
                    f_bbox = None
                    f_det_score = ""
                    face_rank = []

                final_id, reason = final_decision(b_best_id, b_score, b_margin, face_result)

                crop_name = (
                    f"crop_{ts}_det{det_index}_final-{final_id}_"
                    f"body-{b_best_id}-{b_score:.3f}_face-{f_best_id}-{f_score:.3f}.jpg"
                )
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                color = (0, 180, 0) if final_id != "Unknown" else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"final={final_id} body={b_best_id}:{b_score:.2f}/{b_margin:.2f} "
                    f"face={f_best_id}:{f_score:.2f}/{f_margin:.2f}"
                )
                draw_label(annotated, label, x1, max(22, y1), color)

                row = {
                    "timestamp": ts,
                    "frame_index": frame_index,
                    "detection_index": det_index,
                    "crop_path": str(crop_path),
                    "frame_path": str(frame_path),
                    "annotated_path": "",
                    "bbox": f"{x1},{y1},{x2},{y2}",
                    "bbox_width": x2 - x1,
                    "bbox_height": y2 - y1,
                    "bbox_aspect": f"{((y2 - y1) / max(1, x2 - x1)):.6f}",
                    "body_decision": b_decision,
                    "body_best_id": b_best_id,
                    "body_score": f"{b_score:.6f}",
                    "body_second_id": b_second_id,
                    "body_second_score": f"{b_second_score:.6f}",
                    "body_margin": f"{b_margin:.6f}",
                    "body_ranking": ";".join(f"{emp}:{score:.6f}" for emp, score in body_rank),
                    "face_decision": f_decision,
                    "face_best_id": f_best_id,
                    "face_score": f"{f_score:.6f}",
                    "face_second_score": f"{f_second_score:.6f}",
                    "face_margin": f"{f_margin:.6f}",
                    "face_det_score": f"{float(f_det_score):.6f}" if f_det_score != "" else "",
                    "face_bbox": f_bbox,
                    "face_ranking": ";".join(
                        f"{item.get('employee_id')}:{float(item.get('score', 0.0)):.6f}"
                        for item in face_rank
                    ),
                    "final_id": final_id,
                    "final_reason": reason,
                }
                writer.writerow(row)
                saved_count += 1
                print(
                    f"saved={saved_count:03d} det={det_index} final={final_id} "
                    f"reason={reason} body={b_best_id}:{b_score:.3f}/{b_margin:.3f} "
                    f"face={f_best_id}:{f_score:.3f}/{f_margin:.3f} decision={f_decision}"
                )

            annotated_path = annotated_dir / f"annotated_{ts}.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            f.flush()
            last_saved = now

            if args.display:
                cv2.imshow("standalone webcam hybrid debug", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if args.display:
        cv2.destroyAllWindows()

    print()
    print("Saved run:", out_dir)
    print("Saved crops:", crops_dir)
    print("Saved frames:", frames_dir)
    print("Saved annotated:", annotated_dir)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
