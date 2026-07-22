"""
Zone 2 hybrid Re-ID video test.

Pipeline:
- local video input
- YOLO person detection
- project ByteTrack wrapper for track_id
- body Re-ID against python_cv/data/gallery/gallery.pkl
- optional InsightFace verification against python_cv/data/face_gallery/insightface_gallery.pkl
- frame-local fusion decision and duplicate-ID review

No RFID, no entry line, no crossing, no database writes, no gallery updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]  # python_cv/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_VIDEO = Path("D:/VID_20260507_014519.mp4")
OUTPUT_DIR = ROOT_DIR / "outputs" / "zone2_hybrid_reid_video_test"
FACE_GALLERY_PATH = ROOT_DIR / "data" / "face_gallery" / "insightface_gallery.pkl"

EMPLOYEE_NAMES = {
    "NV001": "Bo Man",
    "NV002": "Me Mai",
    "NV003": "Anh Minh",
    "NV004": "Chi Dung",
    "NV005": "Toi",
}


cv2 = None
np = None
Detections = None
YOLO = None
Gallery = None
ReIDEngine = None
InsightFaceEngine = None
Tracker = None
YOLO_CLASSES = [0]
YOLO_CONF = 0.4
YOLO_IMG_SIZE = 640
YOLO_WEIGHTS = ROOT_DIR.parent / "yolo11n-seg.pt"
REID_MATCH_THRESHOLD = 0.93
REID_MATCH_MARGIN = 0.015
FACE_MATCH_THRESHOLD = 0.38
FACE_MATCH_MARGIN = 0.09


def load_runtime_dependencies():
    global cv2, np, Detections, YOLO
    global Gallery, ReIDEngine, InsightFaceEngine, Tracker
    global YOLO_CLASSES, YOLO_CONF, YOLO_IMG_SIZE, YOLO_WEIGHTS
    global REID_MATCH_THRESHOLD, REID_MATCH_MARGIN, FACE_MATCH_THRESHOLD, FACE_MATCH_MARGIN

    try:
        import cv2 as _cv2
        import numpy as _np
        from supervision import Detections as _Detections
        from ultralytics import YOLO as _YOLO

        from config.settings import (
            FACE_MATCH_MARGIN as _FACE_MATCH_MARGIN,
            FACE_MATCH_THRESHOLD as _FACE_MATCH_THRESHOLD,
            REID_MATCH_MARGIN as _REID_MATCH_MARGIN,
            REID_MATCH_THRESHOLD as _REID_MATCH_THRESHOLD,
            YOLO_CLASSES as _YOLO_CLASSES,
            YOLO_CONF as _YOLO_CONF,
            YOLO_IMG_SIZE as _YOLO_IMG_SIZE,
            YOLO_WEIGHTS as _YOLO_WEIGHTS,
        )
        from reid.face_insightface_engine import InsightFaceEngine as _InsightFaceEngine
        from reid.gallery import Gallery as _Gallery
        from reid.reid_engine import ReIDEngine as _ReIDEngine
        from tracking.tracker import Tracker as _Tracker

    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"[ERROR] Missing Python dependency: {missing}\n"
            "Install project dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    Detections = _Detections
    YOLO = _YOLO
    Gallery = _Gallery
    ReIDEngine = _ReIDEngine
    InsightFaceEngine = _InsightFaceEngine
    Tracker = _Tracker
    YOLO_CLASSES = _YOLO_CLASSES
    YOLO_CONF = _YOLO_CONF
    YOLO_IMG_SIZE = _YOLO_IMG_SIZE
    YOLO_WEIGHTS = _YOLO_WEIGHTS
    REID_MATCH_THRESHOLD = _REID_MATCH_THRESHOLD
    REID_MATCH_MARGIN = _REID_MATCH_MARGIN
    FACE_MATCH_THRESHOLD = _FACE_MATCH_THRESHOLD
    FACE_MATCH_MARGIN = _FACE_MATCH_MARGIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Zone 2 hybrid Body Re-ID + InsightFace on video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--show", action="store_true", help="Show realtime annotated window.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional runtime limit in video seconds.")
    parser.add_argument("--face", choices=("on", "off"), default="on", help="Enable/disable InsightFace verification.")
    parser.add_argument(
        "--target-id",
        default=None,
        help="Demo mode: only this employee ID can be shown as verified, e.g. NV005.",
    )
    parser.add_argument(
        "--hide-unknown-labels",
        action="store_true",
        help="In target demo mode, draw unknown boxes without text.",
    )
    parser.add_argument(
        "--only-show-confirmed",
        action="store_true",
        help="In target demo mode, draw only target verified tracks.",
    )
    return parser.parse_args()


def clip_bbox(xyxy, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def rank_body(reid, gallery_data: dict[str, list], crop_bgr) -> dict[str, Any]:
    emb = reid.get_embedding(crop_bgr)
    if emb is None:
        return {
            "body_id": None,
            "body_score": 0.0,
            "body_second_id": None,
            "body_second_score": 0.0,
            "body_margin": 0.0,
        }

    rows = []
    for employee_id, embeds in gallery_data.items():
        if not embeds:
            continue
        score = reid.match_score(emb, embeds)
        rows.append((employee_id, float(score)))

    if not rows:
        return {
            "body_id": None,
            "body_score": 0.0,
            "body_second_id": None,
            "body_second_score": 0.0,
            "body_margin": 0.0,
        }

    rows.sort(key=lambda x: x[1], reverse=True)
    body_id, body_score = rows[0]
    second_id, second_score = rows[1] if len(rows) > 1 else (None, 0.0)

    return {
        "body_id": body_id,
        "body_score": body_score,
        "body_second_id": second_id,
        "body_second_score": float(second_score),
        "body_margin": float(body_score - second_score),
    }


def run_face(face_engine, crop_bgr, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "face_status": "FACE_OFF",
            "face_id": None,
            "face_score": 0.0,
            "face_margin": 0.0,
            "face_best_id": None,
            "face_display": "OFF",
            "face_bbox": None,
        }

    result = face_engine.identify_image(crop_bgr)
    status = str(result.get("status", "UNKNOWN"))
    ranking = result.get("ranking") or []
    best = ranking[0] if ranking else None
    matched_id = result.get("employee_id") if status == "MATCH" else None
    if matched_id == "Unknown":
        matched_id = None

    best_id = best.get("employee_id") if best else matched_id
    score = float(result.get("score", 0.0))
    margin = float(result.get("margin", 0.0))

    if score < FACE_MATCH_THRESHOLD:
        matched_id = None
        if status == "MATCH":
            status = "UNKNOWN"

    if status == "MATCH" and matched_id:
        face_display = f"{matched_id} {score:.3f}"
    elif status == "NO_FACE":
        face_display = "NO_FACE"
    elif status == "NO_GALLERY":
        face_display = "NO_GALLERY"
    elif best_id:
        face_display = f"{status}:{best_id} {score:.3f}"
    else:
        face_display = status

    return {
        "face_status": status,
        "face_id": matched_id,
        "face_score": score,
        "face_margin": margin,
        "face_best_id": best_id,
        "face_display": face_display,
        "face_bbox": result.get("bbox"),
    }


def is_body_strict(body: dict[str, Any]) -> bool:
    return (
        body.get("body_id") is not None
        and float(body.get("body_score") or 0.0) >= REID_MATCH_THRESHOLD
        and float(body.get("body_margin") or 0.0) >= REID_MATCH_MARGIN
    )


def decide_zone2(body: dict[str, Any], face: dict[str, Any]) -> dict[str, Any]:
    body_id = body.get("body_id")
    body_score = float(body.get("body_score") or 0.0)
    body_margin = float(body.get("body_margin") or 0.0)
    face_score = float(face.get("face_score") or 0.0)
    face_id = face.get("face_id") if face_score >= FACE_MATCH_THRESHOLD else None
    face_status = face.get("face_status")

    final_id = None
    status = "UNKNOWN"
    decision_score = 0.0

    if face_id is not None:
        final_id = face_id
        decision_score = float(face.get("face_score") or 0.0)

        if body_id == face_id:
            status = "FACE_CONFIRMED"
        else:
            status = "REVIEW_FACE_BODY_CONFLICT"

    elif face_status in {"NO_FACE", "NO_GALLERY", "FACE_OFF"}:
        if is_body_strict(body):
            final_id = body_id
            status = "BODY_ONLY_STRICT"
            decision_score = body_score
        elif body_id is not None and (body_score > 0.0 or body_margin > 0.0):
            status = "REVIEW_LOW_CONFIDENCE"
            decision_score = body_score
        else:
            status = "UNKNOWN"

    else:
        if body_id is not None and (body_score > 0.0 or body_margin > 0.0):
            status = "REVIEW_LOW_CONFIDENCE"
            decision_score = max(body_score, float(face.get("face_score") or 0.0))
        else:
            status = "UNKNOWN"

    return {
        "final_id": final_id,
        "status": status,
        "decision_score": float(decision_score),
    }


def apply_duplicate_review(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        final_id = row.get("final_id")
        if final_id:
            groups.setdefault(final_id, []).append(row)

    for same_id_rows in groups.values():
        if len(same_id_rows) < 2:
            continue

        same_id_rows.sort(key=lambda x: float(x.get("decision_score") or 0.0), reverse=True)
        for loser in same_id_rows[1:]:
            loser["duplicate_flag"] = True
            loser["status"] = "REVIEW_DUPLICATE_ID"


def apply_target_demo_policy(rows: list[dict[str, Any]], target_id: str | None) -> None:
    if not target_id:
        return

    target_id = target_id.upper()
    verified_statuses = {"FACE_CONFIRMED", "BODY_ONLY_STRICT"}

    for row in rows:
        row["raw_final_id"] = row.get("final_id")
        row["raw_status"] = row.get("status")

        is_target_verified = (
            row.get("final_id") == target_id
            and row.get("status") in verified_statuses
        )

        if is_target_verified:
            row["demo_verified"] = True
            continue

        row["demo_verified"] = False
        row["final_id"] = None
        row["status"] = "UNKNOWN_PERSON"


def draw_text(frame, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.52):
    y = max(16, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def status_color(status: str) -> tuple[int, int, int]:
    if status == "FACE_CONFIRMED":
        return (0, 220, 80)
    if status == "BODY_ONLY_STRICT":
        return (0, 190, 255)
    if status.startswith("REVIEW"):
        return (0, 100, 255)
    return (180, 180, 180)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "frame_id",
        "timestamp",
        "track_id",
        "bbox",
        "body_id",
        "body_score",
        "body_margin",
        "body_second_id",
        "body_second_score",
        "face_id",
        "face_score",
        "face_status",
        "face_best_id",
        "face_margin",
        "final_id",
        "status",
        "raw_final_id",
        "raw_status",
        "demo_verified",
        "duplicate_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    load_runtime_dependencies()
    target_id = args.target_id.upper() if args.target_id else None

    video_path = args.video.expanduser()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = OUTPUT_DIR / f"zone2_hybrid_reid_{run_id}.mp4"
    output_csv = OUTPUT_DIR / f"zone2_hybrid_reid_{run_id}.csv"
    output_json = OUTPUT_DIR / f"zone2_hybrid_reid_{run_id}.json"

    print(f"[INFO] Video      : {video_path}")
    print(f"[INFO] Face mode  : {args.face}")
    if target_id:
        print(f"[INFO] Target demo: {target_id}")
    print(f"[INFO] Output dir : {OUTPUT_DIR}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        print("[ERROR] Cannot read video dimensions.")
        cap.release()
        return 2

    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[ERROR] Cannot create output video: {output_video}")
        return 2

    print(f"[INFO] Video info : {width}x{height} @ {fps:.2f} FPS, frames={frame_count}")

    model = YOLO(str(YOLO_WEIGHTS))
    tracker = Tracker()
    reid = ReIDEngine()
    gallery = Gallery()
    gallery_data = gallery.all()

    face_enabled = args.face == "on"
    face_engine = None
    if face_enabled:
        face_engine = InsightFaceEngine(
            gallery_path=str(FACE_GALLERY_PATH),
            model_name="buffalo_sc",
            use_gpu=True,
            face_threshold=FACE_MATCH_THRESHOLD,
            face_margin=FACE_MATCH_MARGIN,
        )

    csv_rows: list[dict[str, Any]] = []
    raw_frame_id = 0
    processed_frames = 0
    start_wall = time.time()
    stopped_by_user = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            raw_frame_id += 1
            timestamp = (raw_frame_id - 1) / fps
            if args.max_seconds is not None and timestamp > args.max_seconds:
                break

            processed_frames += 1

            result = model(
                frame,
                classes=YOLO_CLASSES,
                conf=YOLO_CONF,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
            )[0]
            detections = Detections.from_ultralytics(result)
            tracked = tracker.update(detections)

            annotated = frame.copy()
            frame_rows: list[dict[str, Any]] = []
            track_ids = tracked.tracker_id if tracked.tracker_id is not None else []

            for i, raw_track_id in enumerate(track_ids):
                track_id = int(raw_track_id)
                bbox = clip_bbox(tracked.xyxy[i], width, height)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                body = rank_body(reid, gallery_data, crop)
                face = run_face(face_engine, crop, face_enabled)
                decision = decide_zone2(body, face)

                row = {
                    "frame_id": raw_frame_id,
                    "timestamp": round(float(timestamp), 3),
                    "track_id": track_id,
                    "bbox": f"{x1},{y1},{x2},{y2}",
                    "body_id": body.get("body_id"),
                    "body_score": round(float(body.get("body_score") or 0.0), 6),
                    "body_margin": round(float(body.get("body_margin") or 0.0), 6),
                    "body_second_id": body.get("body_second_id"),
                    "body_second_score": round(float(body.get("body_second_score") or 0.0), 6),
                    "face_id": face.get("face_id"),
                    "face_score": round(float(face.get("face_score") or 0.0), 6),
                    "face_margin": round(float(face.get("face_margin") or 0.0), 6),
                    "face_status": face.get("face_status"),
                    "face_best_id": face.get("face_best_id"),
                    "final_id": decision.get("final_id"),
                    "status": decision.get("status"),
                    "raw_final_id": decision.get("final_id"),
                    "raw_status": decision.get("status"),
                    "demo_verified": False,
                    "decision_score": round(float(decision.get("decision_score") or 0.0), 6),
                    "duplicate_flag": False,
                    "_bbox_tuple": bbox,
                    "_face_bbox": face.get("face_bbox"),
                    "_face_display": face.get("face_display"),
                }
                frame_rows.append(row)

            apply_duplicate_review(frame_rows)
            apply_target_demo_policy(frame_rows, target_id)

            for row in frame_rows:
                x1, y1, x2, y2 = row["_bbox_tuple"]
                demo_verified = bool(row.get("demo_verified"))

                if target_id and args.only_show_confirmed and not demo_verified:
                    clean_row = dict(row)
                    clean_row.pop("_bbox_tuple", None)
                    clean_row.pop("_face_bbox", None)
                    clean_row.pop("_face_display", None)
                    csv_rows.append(clean_row)
                    continue

                if target_id:
                    color = (0, 220, 80) if demo_verified else (0, 165, 255)
                    thickness = 3 if demo_verified else 1
                else:
                    color = status_color(str(row["status"]))
                    thickness = 2

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                face_bbox = row.get("_face_bbox")
                if face_bbox is not None and (not target_id or demo_verified):
                    fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
                    cv2.rectangle(annotated, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (255, 180, 0), 1)

                if target_id:
                    if demo_verified:
                        target_name = EMPLOYEE_NAMES.get(target_id, target_id)
                        draw_text(
                            annotated,
                            f"T{row['track_id']} {target_id} {target_name} VERIFIED",
                            x1,
                            y1 - 42,
                            color,
                        )
                        draw_text(annotated, f"BODY score {float(row['body_score']):.3f}", x1, y1 - 24, color)
                        draw_text(annotated, f"FACE score {float(row['face_score']):.3f}", x1, y1 - 6, color)
                    elif not args.hide_unknown_labels:
                        draw_text(annotated, f"T{row['track_id']} UNKNOWN", x1, y1 - 8, color)
                else:
                    final_text = row["final_id"] or "Unknown"
                    body_text = f"BODY: {row['body_id'] or 'Unknown'} {float(row['body_score']):.3f}"
                    face_text = f"FACE: {row.get('_face_display') or 'Unknown'}"
                    status_text = f"STATUS: {row['status']}"
                    duplicate_text = " DUP" if row["duplicate_flag"] else ""

                    draw_text(annotated, f"T{row['track_id']}{duplicate_text}", x1, y1 - 60, color)
                    draw_text(annotated, body_text, x1, y1 - 42, color)
                    draw_text(annotated, face_text, x1, y1 - 24, color)
                    draw_text(annotated, f"FINAL: {final_text}", x1, y1 - 6, color)
                    draw_text(annotated, status_text, x1, y2 + 18, color)

                clean_row = dict(row)
                clean_row.pop("_bbox_tuple", None)
                clean_row.pop("_face_bbox", None)
                clean_row.pop("_face_display", None)
                csv_rows.append(clean_row)

            cv2.rectangle(annotated, (0, 0), (width, 34), (18, 18, 18), -1)
            draw_text(
                annotated,
                (
                    f"ZONE 2 TARGET DEMO {target_id} | t={timestamp:.2f}s | frame={raw_frame_id} | face={args.face}"
                    if target_id
                    else f"ZONE 2 HYBRID RE-ID | t={timestamp:.2f}s | frame={raw_frame_id} | face={args.face}"
                ),
                10,
                24,
                (240, 240, 240),
                scale=0.62,
            )

            writer.write(annotated)

            if args.show:
                cv2.imshow("Zone 2 Hybrid Re-ID Video Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break

            if processed_frames % 50 == 0:
                print(f"[INFO] frame={raw_frame_id} t={timestamp:.1f}s rows={len(csv_rows)}")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    write_csv(output_csv, csv_rows)
    summary = {
        "run_id": run_id,
        "video": str(video_path),
        "output_video": str(output_video),
        "output_csv": str(output_csv),
        "face_mode": args.face,
        "target_id": target_id,
        "hide_unknown_labels": bool(args.hide_unknown_labels),
        "only_show_confirmed": bool(args.only_show_confirmed),
        "face_gallery": str(FACE_GALLERY_PATH),
        "processed_frames": processed_frames,
        "rows": len(csv_rows),
        "stopped_by_user": stopped_by_user,
        "elapsed_wall_sec": round(time.time() - start_wall, 3),
        "status_counts": {},
    }
    for row in csv_rows:
        status = row.get("status", "UNKNOWN")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Zone 2 hybrid video test outputs:")
    print(f"  video: {output_video}")
    print(f"  csv  : {output_csv}")
    print(f"  json : {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
