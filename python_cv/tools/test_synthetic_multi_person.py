import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (  # noqa: E402
    REID_MATCH_MARGIN,
    REID_MATCH_THRESHOLD,
    YOLO_CLASSES,
    YOLO_IMG_SIZE,
    YOLO_IOU,
    YOLO_WEIGHTS,
)
from reid.gallery import Gallery  # noqa: E402
from reid.reid_engine import ReIDEngine  # noqa: E402


DEFAULT_ROOT = Path("D:/warehouse_dataset/synthetic_tests")
EMPLOYEE_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]


@dataclass
class Detection:
    bbox: list[int]
    det_conf: float
    track_id: int | None = None
    pred_emp_id: str | None = None
    reid_score: float = 0.0
    reid_second_id: str | None = None
    reid_second_score: float = 0.0
    reid_margin: float = 0.0
    best_emp_id: str | None = None
    best_score: float = 0.0
    second_emp_id: str | None = None
    second_score: float = 0.0
    margin: float = 0.0
    top5_scores: list[tuple[str, float]] | None = None
    reid_threshold_used: float = REID_MATCH_THRESHOLD
    reid_margin_used: float = REID_MATCH_MARGIN
    reject_reason: str = ""
    reid_call_method: str = "get_embedding+match_score"
    crop_shape: str = ""
    crop_debug_path: str = ""


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.35, max_age: int = 12):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_tracks = set(self.tracks.keys())

        for det in detections:
            best_id = None
            best_iou = 0.0

            for track_id in list(unmatched_tracks):
                score = bbox_iou(det.bbox, self.tracks[track_id]["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is not None and best_iou >= self.iou_threshold:
                det.track_id = best_id
                self.tracks[best_id] = {"bbox": det.bbox, "age": 0}
                unmatched_tracks.remove(best_id)
            else:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {"bbox": det.bbox, "age": 0}
                self.next_id += 1

        for track_id in list(unmatched_tracks):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                self.tracks.pop(track_id, None)

        return detections


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def detect_people(model: YOLO, frame: np.ndarray, conf_threshold: float) -> list[Detection]:
    result = model(
        frame,
        classes=YOLO_CLASSES,
        conf=conf_threshold,
        iou=YOLO_IOU,
        imgsz=YOLO_IMG_SIZE,
        verbose=False,
    )[0]

    h, w = frame.shape[:2]
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        bbox = clamp_bbox(xyxy, w, h)
        if bbox is None:
            continue
        conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        detections.append(Detection(bbox=bbox, det_conf=conf))

    return detections


def rank_reid(reid: ReIDEngine, gallery_data: dict, crop_bgr: np.ndarray) -> dict:
    debug = {
        "pred_emp_id": None,
        "best_emp_id": None,
        "best_score": 0.0,
        "second_emp_id": None,
        "second_score": 0.0,
        "margin": 0.0,
        "top5_scores": [],
        "reject_reason": "",
        "reid_call_method": "get_embedding+match_score",
    }

    if crop_bgr is None or crop_bgr.size == 0:
        debug["reject_reason"] = "invalid_crop"
        return debug

    if not gallery_data:
        debug["reject_reason"] = "no_gallery"
        return debug

    emb = reid.get_embedding(crop_bgr)
    if emb is None:
        debug["reject_reason"] = "no_embedding"
        return debug

    rows = []
    for emp_id, embeds in gallery_data.items():
        if not embeds:
            continue
        rows.append((emp_id, reid.match_score(emb, embeds)))

    if not rows:
        debug["reject_reason"] = "no_gallery"
        return debug

    rows.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score = rows[0]
    second_id, second_score = rows[1] if len(rows) > 1 else (None, 0.0)
    margin = float(best_score - second_score)

    debug.update({
        "best_emp_id": best_id,
        "best_score": float(best_score),
        "second_emp_id": second_id,
        "second_score": float(second_score),
        "margin": margin,
        "top5_scores": [(emp_id, float(score)) for emp_id, score in rows[:5]],
    })

    if best_score >= REID_MATCH_THRESHOLD and margin >= REID_MATCH_MARGIN:
        debug["pred_emp_id"] = best_id
        return debug

    if best_score < REID_MATCH_THRESHOLD:
        debug["reject_reason"] = "low_score"
    elif margin < REID_MATCH_MARGIN:
        debug["reject_reason"] = "low_margin"
    else:
        debug["reject_reason"] = "script_forced_unknown"
    return debug


def apply_reid(frame: np.ndarray, detections: list[Detection], reid: ReIDEngine, gallery_data: dict) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            det.reject_reason = "invalid_crop"
            continue
        det.crop_shape = f"{crop.shape[1]}x{crop.shape[0]}"
        result = rank_reid(reid, gallery_data, crop)
        det.pred_emp_id = result["pred_emp_id"]
        det.best_emp_id = result["best_emp_id"]
        det.best_score = result["best_score"]
        det.second_emp_id = result["second_emp_id"]
        det.second_score = result["second_score"]
        det.margin = result["margin"]
        det.top5_scores = result["top5_scores"]
        det.reject_reason = result["reject_reason"]
        det.reid_call_method = result["reid_call_method"]

        det.reid_score = det.best_score
        det.reid_second_id = det.second_emp_id
        det.reid_second_score = det.second_score
        det.reid_margin = det.margin


def load_ground_truth(gt_path: Path) -> dict:
    return json.loads(gt_path.read_text(encoding="utf-8"))


def gt_for_frame(gt: dict, frame_idx: int) -> list[dict]:
    frames = gt.get("frames", [])
    if frame_idx < 0 or frame_idx >= len(frames):
        return []
    return frames[frame_idx].get("objects", [])


def match_detections_to_gt(detections: list[Detection], gt_objects: list[dict], iou_threshold: float) -> list[dict]:
    matches = []
    used_det = set()
    used_gt = set()
    candidates = []

    for gi, gt_obj in enumerate(gt_objects):
        gt_bbox = [int(v) for v in gt_obj["bbox"]]
        for di, det in enumerate(detections):
            score = bbox_iou(det.bbox, gt_bbox)
            if score >= iou_threshold:
                candidates.append((score, gi, di))

    candidates.sort(reverse=True)
    for score, gi, di in candidates:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        matches.append({
            "gt_index": gi,
            "det_index": di,
            "iou": score,
            "gt": gt_objects[gi],
            "det": detections[di],
        })

    return matches


def color_for_label(label: str | None) -> tuple[int, int, int]:
    if label is None:
        return (0, 165, 255)
    palette = {
        "NV001": (80, 220, 80),
        "NV002": (255, 180, 80),
        "NV003": (80, 180, 255),
        "NV004": (220, 120, 255),
        "NV005": (255, 220, 80),
    }
    return palette.get(label, (180, 180, 180))


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = det.pred_emp_id or "Unknown"
        color = color_for_label(det.pred_emp_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        text = f"{prefix}{label} {det.reid_score:.3f}"
        y_text = max(18, y1 - 8)
        cv2.rectangle(out, (x1, y_text - 17), (min(x1 + 235, out.shape[1] - 1), y_text + 4), color, -1)
        cv2.putText(out, text, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def format_top5(scores: list[tuple[str, float]] | None) -> str:
    if not scores:
        return ""
    return ";".join(f"{emp_id}:{score:.6f}" for emp_id, score in scores)


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "NA"


def save_reid_crop_debug(
    frame: np.ndarray,
    det: Detection | None,
    case_name: str,
    frame_idx: int,
    gt_obj: dict | None,
    det_index: int | None,
    crop_dir: Path,
) -> str:
    if det is None:
        return ""

    x1, y1, x2, y2 = det.bbox
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return ""

    gt_id = gt_obj.get("emp_id") if gt_obj else "NO_GT"
    best_id = det.best_emp_id or "None"
    reason = det.reject_reason or "accepted"
    det_suffix = f"d{det_index}" if det_index is not None else "dNA"
    name = (
        f"{safe_name(case_name)}_f{frame_idx:05d}_gt-{safe_name(gt_id)}_"
        f"best-{safe_name(best_id)}_score-{det.best_score:.3f}_"
        f"reason-{safe_name(reason)}_{det_suffix}.jpg"
    )
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_path = crop_dir / name
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    det.crop_debug_path = str(out_path)
    return str(out_path)


def prediction_row(
    case_name: str,
    source_type: str,
    frame_idx: int,
    gt_obj: dict | None,
    det: Detection | None,
    match_iou: float,
) -> dict:
    return {
        "case_name": case_name,
        "source_type": source_type,
        "frame_idx": frame_idx,
        "gt_emp_id": gt_obj.get("emp_id") if gt_obj else "",
        "gt_bbox": json.dumps(gt_obj.get("bbox")) if gt_obj else "",
        "det_bbox": json.dumps(det.bbox) if det else "",
        "match_iou": f"{match_iou:.4f}" if match_iou else "0.0000",
        "track_id": det.track_id if det and det.track_id is not None else "",
        "pred_emp_id": det.pred_emp_id if det and det.pred_emp_id else ("Unknown" if det else "MISS"),
        "reid_score": f"{det.reid_score:.6f}" if det else "0.000000",
        "reid_second_id": det.reid_second_id if det and det.reid_second_id else "",
        "reid_second_score": f"{det.reid_second_score:.6f}" if det else "0.000000",
        "reid_margin": f"{det.reid_margin:.6f}" if det else "0.000000",
        "best_emp_id": det.best_emp_id if det and det.best_emp_id else "",
        "best_score": f"{det.best_score:.6f}" if det else "0.000000",
        "second_emp_id": det.second_emp_id if det and det.second_emp_id else "",
        "second_score": f"{det.second_score:.6f}" if det else "0.000000",
        "margin": f"{det.margin:.6f}" if det else "0.000000",
        "top5_scores": format_top5(det.top5_scores) if det else "",
        "reid_threshold_used": f"{det.reid_threshold_used:.6f}" if det else f"{REID_MATCH_THRESHOLD:.6f}",
        "reid_margin_used": f"{det.reid_margin_used:.6f}" if det else f"{REID_MATCH_MARGIN:.6f}",
        "reject_reason": det.reject_reason if det and det.reject_reason else "",
        "reid_call_method": det.reid_call_method if det else "",
        "crop_shape": det.crop_shape if det else "",
        "crop_debug_path": det.crop_debug_path if det else "",
        "det_conf": f"{det.det_conf:.6f}" if det else "0.000000",
        "correct": bool(gt_obj and det and det.pred_emp_id == gt_obj.get("emp_id")),
    }


def process_image_case(
    image_path: Path,
    gt_path: Path,
    review_dir: Path,
    crop_debug_dir: Path,
    model: YOLO,
    reid: ReIDEngine,
    gallery_data: dict,
    conf_threshold: float,
    iou_threshold: float,
    show: bool,
) -> tuple[list[dict], dict]:
    case_name = image_path.stem
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    gt = load_ground_truth(gt_path)
    representative_idx = len(gt.get("frames", [])) // 2
    gt_objects = gt_for_frame(gt, representative_idx)

    detections = detect_people(model, frame, conf_threshold)
    apply_reid(frame, detections, reid, gallery_data)

    matches = match_detections_to_gt(detections, gt_objects, iou_threshold)
    matched_gt = {m["gt_index"] for m in matches}
    matched_det = {m["det_index"] for m in matches}

    rows = []
    for match in matches:
        save_reid_crop_debug(
            frame,
            match["det"],
            case_name,
            representative_idx,
            match["gt"],
            match["det_index"],
            crop_debug_dir,
        )
        rows.append(prediction_row(
            case_name,
            "image",
            representative_idx,
            match["gt"],
            match["det"],
            match["iou"],
        ))
    for gi, gt_obj in enumerate(gt_objects):
        if gi not in matched_gt:
            rows.append(prediction_row(case_name, "image", representative_idx, gt_obj, None, 0.0))
    for di, det in enumerate(detections):
        if di not in matched_det:
            save_reid_crop_debug(frame, det, case_name, representative_idx, None, di, crop_debug_dir)
            rows.append(prediction_row(case_name, "image", representative_idx, None, det, 0.0))

    review = draw_detections(frame, detections)
    out_path = review_dir / f"{case_name}_review.jpg"
    cv2.imwrite(str(out_path), review, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    if show:
        cv2.imshow("synthetic image review", review)
        cv2.waitKey(1)

    summary = summarize_rows(case_name, "image", rows, len(gt_objects), len(detections))
    return rows, summary


def process_video_case(
    video_path: Path,
    gt_path: Path,
    review_dir: Path,
    crop_debug_dir: Path,
    model: YOLO,
    reid: ReIDEngine,
    gallery_data: dict,
    conf_threshold: float,
    iou_threshold: float,
    max_frames: int | None,
    save_video: bool,
    show: bool,
) -> tuple[list[dict], dict]:
    case_name = video_path.stem
    gt = load_ground_truth(gt_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or gt.get("fps", 15)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save_video:
        out_path = review_dir / f"{case_name}_review.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open review writer: {out_path}")

    tracker = IoUTracker()
    rows = []
    expected_total = 0
    detected_total = 0
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gt_objects = gt_for_frame(gt, frame_idx)
        expected_total += len(gt_objects)

        detections = detect_people(model, frame, conf_threshold)
        detections = tracker.update(detections)
        apply_reid(frame, detections, reid, gallery_data)
        detected_total += len(detections)

        matches = match_detections_to_gt(detections, gt_objects, iou_threshold)
        matched_gt = {m["gt_index"] for m in matches}
        matched_det = {m["det_index"] for m in matches}

        for match in matches:
            save_reid_crop_debug(
                frame,
                match["det"],
                case_name,
                frame_idx,
                match["gt"],
                match["det_index"],
                crop_debug_dir,
            )
            rows.append(prediction_row(case_name, "video", frame_idx, match["gt"], match["det"], match["iou"]))
        for gi, gt_obj in enumerate(gt_objects):
            if gi not in matched_gt:
                rows.append(prediction_row(case_name, "video", frame_idx, gt_obj, None, 0.0))
        for di, det in enumerate(detections):
            if di not in matched_det:
                save_reid_crop_debug(frame, det, case_name, frame_idx, None, di, crop_debug_dir)
                rows.append(prediction_row(case_name, "video", frame_idx, None, det, 0.0))

        review = draw_detections(frame, detections)
        if writer is not None:
            writer.write(review)
        if show:
            cv2.imshow("synthetic video review", review)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    summary = summarize_rows(case_name, "video", rows, expected_total, detected_total)
    summary["frames_processed"] = frame_idx
    return rows, summary


def summarize_rows(case_name: str, source_type: str, rows: list[dict], expected_total: int, detected_total: int) -> dict:
    matched = [r for r in rows if r["gt_emp_id"] and r["det_bbox"]]
    reid_correct = sum(1 for r in matched if r["pred_emp_id"] == r["gt_emp_id"])
    unknown = sum(1 for r in rows if r["det_bbox"] and r["pred_emp_id"] == "Unknown")
    recall = len(matched) / expected_total if expected_total else 0.0
    reid_acc = reid_correct / len(matched) if matched else 0.0

    confusions = Counter()
    for r in matched:
        gt_id = r["gt_emp_id"]
        pred_id = r["pred_emp_id"]
        if pred_id != gt_id:
            confusions[(gt_id, pred_id)] += 1

    common = "; ".join(f"{a}->{b}:{n}" for (a, b), n in confusions.most_common(5))
    return {
        "case_name": case_name,
        "source_type": source_type,
        "frames_processed": "",
        "expected_people_total": expected_total,
        "detected_people_total": detected_total,
        "matched_people_total": len(matched),
        "detection_recall": recall,
        "reid_correct": reid_correct,
        "reid_accuracy_on_matched": reid_acc,
        "unknown_or_low_conf_count": unknown,
        "common_confusions": common,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion_matrix(path: Path, rows: list[dict]) -> None:
    labels = EMPLOYEE_IDS + ["Unknown", "MISS"]
    matrix = {gt: {pred: 0 for pred in labels} for gt in EMPLOYEE_IDS}

    for row in rows:
        gt = row["gt_emp_id"]
        if gt not in EMPLOYEE_IDS:
            continue
        pred = row["pred_emp_id"] if row["det_bbox"] else "MISS"
        if pred not in labels:
            pred = "Unknown"
        matrix[gt][pred] += 1

    out_rows = []
    for gt in EMPLOYEE_IDS:
        row = {"gt_emp_id": gt}
        row.update(matrix[gt])
        out_rows.append(row)

    write_csv(path, out_rows, ["gt_emp_id"] + labels)


def write_markdown_summary(path: Path, summary_rows: list[dict]) -> None:
    total_expected = sum(int(r["expected_people_total"]) for r in summary_rows)
    total_detected = sum(int(r["detected_people_total"]) for r in summary_rows)
    total_matched = sum(int(r["matched_people_total"]) for r in summary_rows)
    total_correct = sum(int(r["reid_correct"]) for r in summary_rows)
    total_unknown = sum(int(r["unknown_or_low_conf_count"]) for r in summary_rows)
    recall = total_matched / total_expected if total_expected else 0.0
    reid_acc = total_correct / total_matched if total_matched else 0.0

    confusion = Counter()
    for row in summary_rows:
        for item in str(row["common_confusions"]).split(";"):
            item = item.strip()
            if not item or ":" not in item or "->" not in item:
                continue
            pair, count = item.rsplit(":", 1)
            confusion[pair] += int(count)

    lines = [
        "# Synthetic Multi-Person AI Test Summary",
        "",
        f"- expected_people_total: {total_expected}",
        f"- detected_people_total: {total_detected}",
        f"- detection_recall: {recall:.4f}",
        f"- reid_correct: {total_correct}",
        f"- reid_accuracy_on_matched: {reid_acc:.4f}",
        f"- unknown_or_low_conf_count: {total_unknown}",
        "",
        "## Per Case",
        "",
        "| case | type | frames | expected | detected | recall | reid_acc | unknown | confusions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case_name']} | {row['source_type']} | {row['frames_processed']} | "
            f"{row['expected_people_total']} | {row['detected_people_total']} | "
            f"{float(row['detection_recall']):.4f} | {float(row['reid_accuracy_on_matched']):.4f} | "
            f"{row['unknown_or_low_conf_count']} | {row['common_confusions']} |"
        )

    lines.extend(["", "## Common Confusions", ""])
    if confusion:
        for pair, count in confusion.most_common(10):
            lines.append(f"- {pair}: {count}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_top_scores(value: str) -> list[tuple[str, float]]:
    scores = []
    for item in str(value).split(";"):
        if not item or ":" not in item:
            continue
        emp_id, score = item.rsplit(":", 1)
        scores.append((emp_id, to_float(score)))
    return scores


def write_reid_debug_report(path: Path, rows: list[dict], summary_rows: list[dict], conf_threshold: float) -> None:
    detected = [r for r in rows if r["det_bbox"]]
    matched = [r for r in detected if r["gt_emp_id"] and r["gt_emp_id"] != "MISS"]
    unknown = [r for r in matched if r["pred_emp_id"] == "Unknown"]
    accepted = [r for r in matched if r["pred_emp_id"] not in ("Unknown", "MISS", "")]

    reason_counts = Counter(r["reject_reason"] or "unknown_reason_not_found" for r in unknown)
    methods = Counter(r["reid_call_method"] or "unknown" for r in detected)

    best_scores = [to_float(r["best_score"]) for r in unknown]
    second_scores = [to_float(r["second_score"]) for r in unknown]
    margins = [to_float(r["margin"]) for r in unknown]
    accepted_scores = [to_float(r["best_score"]) for r in accepted]
    accepted_margins = [to_float(r["margin"]) for r in accepted]

    high_score_low_margin = [
        r for r in unknown
        if to_float(r["best_score"]) >= REID_MATCH_THRESHOLD and to_float(r["margin"]) < REID_MATCH_MARGIN
    ]
    low_score = [r for r in unknown if to_float(r["best_score"]) < REID_MATCH_THRESHOLD]

    rank_sums: dict[int, list[float]] = defaultdict(list)
    for row in unknown:
        for idx, (_, score) in enumerate(parse_top_scores(row["top5_scores"]), start=1):
            rank_sums[idx].append(score)

    top5_avg = []
    for idx in range(1, 6):
        values = rank_sums.get(idx, [])
        if values:
            top5_avg.append(f"rank{idx}={float(np.mean(values)):.4f}")

    crop_sizes = []
    for row in detected:
        shape = row.get("crop_shape", "")
        if "x" not in shape:
            continue
        w, h = shape.split("x", 1)
        crop_sizes.append((int(to_float(w)), int(to_float(h))))

    if crop_sizes:
        widths = [w for w, _ in crop_sizes]
        heights = [h for _, h in crop_sizes]
        crop_summary = (
            f"count={len(crop_sizes)}, "
            f"width min/mean/max={min(widths)}/{float(np.mean(widths)):.1f}/{max(widths)}, "
            f"height min/mean/max={min(heights)}/{float(np.mean(heights)):.1f}/{max(heights)}"
        )
    else:
        crop_summary = "No valid crop shape recorded."

    total_expected = sum(int(r["expected_people_total"]) for r in summary_rows)
    total_detected = sum(int(r["detected_people_total"]) for r in summary_rows)
    total_matched = sum(int(r["matched_people_total"]) for r in summary_rows)
    total_correct = sum(int(r["reid_correct"]) for r in summary_rows)

    mean_unknown_best = float(np.mean(best_scores)) if best_scores else 0.0
    mean_unknown_second = float(np.mean(second_scores)) if second_scores else 0.0
    mean_unknown_margin = float(np.mean(margins)) if margins else 0.0
    mean_accepted_best = float(np.mean(accepted_scores)) if accepted_scores else 0.0
    mean_accepted_margin = float(np.mean(accepted_margins)) if accepted_margins else 0.0

    lines = [
        "# Re-ID Debug Report",
        "",
        "## Scope",
        "",
        "- Pipeline tested: YOLO person detection -> crop BGR bbox -> ReIDEngine.get_embedding() -> ReIDEngine.match_score() per gallery ID.",
        "- This script does not call InsightFace, camera, fusion, database, or training code.",
        "",
        "## Summary",
        "",
        f"- expected_people_total: {total_expected}",
        f"- detected_people_total: {total_detected}",
        f"- matched_people_total: {total_matched}",
        f"- reid_correct: {total_correct}",
        f"- matched_unknown_count: {len(unknown)}",
        f"- accepted_matched_count: {len(accepted)}",
        "",
        "## Direct Answers",
        "",
        "1. Re-ID Unknown appears in this test script after `rank_reid()` returns `pred_emp_id=None`; `prediction_row()` serializes that detected crop as `Unknown`.",
        f"2. Threshold/margin used: REID_MATCH_THRESHOLD={REID_MATCH_THRESHOLD:.6f}, REID_MATCH_MARGIN={REID_MATCH_MARGIN:.6f}.",
        f"3. `--conf-threshold` is YOLO-only in `detect_people()`; current value was {conf_threshold:.3f}. It does not change Re-ID threshold or margin.",
        f"4. Unknown top-5 average scores: {', '.join(top5_avg) if top5_avg else 'No unknown top-5 scores recorded.'}",
        f"5. Score high but rejected by margin: {len(high_score_low_margin)} / {len(unknown)} Unknown matched crops.",
        f"6. Score below threshold: {len(low_score)} / {len(unknown)} Unknown matched crops. Unknown best-score mean={mean_unknown_best:.4f}, second mean={mean_unknown_second:.4f}, margin mean={mean_unknown_margin:.4f}.",
        f"7. Re-ID crop size summary: {crop_summary}. Inspect saved crops in `review_outputs/reid_crops_debug` for color/person/detail issues.",
        f"8. API call method used: {', '.join(f'{k}={v}' for k, v in methods.items())}. Input is BGR crop; `ReIDEngine.get_embedding()` resizes and converts BGR->RGB internally.",
        "9. Next fix should be chosen after comparing this report with `source_image_reid_sanity.md`: if source images pass but synthetic crops fail, debug synthetic cutout/scale/detection crops; if both fail, debug gallery/model/mapping/threshold.",
        "",
        "## Reject Reasons",
        "",
    ]

    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Score Distributions",
        "",
        f"- Unknown best_score mean: {mean_unknown_best:.6f}",
        f"- Unknown second_score mean: {mean_unknown_second:.6f}",
        f"- Unknown margin mean: {mean_unknown_margin:.6f}",
        f"- Accepted best_score mean: {mean_accepted_best:.6f}",
        f"- Accepted margin mean: {mean_accepted_margin:.6f}",
        "",
        "## Notes",
        "",
        "- `api_returned_unknown` should only appear if this script later switches to `ReIDEngine.identify()`. Current method computes ranked scores directly for debug visibility.",
        "- `script_forced_unknown` means the script had a ranked result but no specific threshold/margin branch explained it.",
        "- Ground-truth misses are written as `MISS`, not `Unknown`, so Unknown counts mean detected crops rejected by Re-ID logic.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["images", "videos", "all"], default="all")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all:
        raise SystemExit("Use --all to process the synthetic test set.")

    synthetic_root = DEFAULT_ROOT
    images_dir = synthetic_root / "images"
    videos_dir = synthetic_root / "videos"
    gt_dir = synthetic_root / "ground_truth"
    review_dir = synthetic_root / "review_outputs"
    crop_debug_dir = review_dir / "reid_crops_debug"
    reports_dir = synthetic_root / "reports"
    review_dir.mkdir(parents=True, exist_ok=True)
    crop_debug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] YOLO weights: {YOLO_WEIGHTS}")
    yolo = YOLO(str(YOLO_WEIGHTS))
    if args.device:
        yolo.to(args.device)

    print("[INFO] Loading Body Re-ID engine and gallery")
    reid = ReIDEngine()
    gallery = Gallery()
    gallery_data = gallery.all()
    if not gallery_data:
        raise RuntimeError("Gallery is empty. Expected python_cv/data/gallery/gallery.pkl")
    print(f"[INFO] Gallery employees: {sorted(gallery_data.keys())}")

    all_rows = []
    summary_rows = []

    if args.mode in ("images", "all"):
        for image_path in sorted(images_dir.glob("*.jpg")):
            gt_path = gt_dir / f"{image_path.stem}.json"
            if not gt_path.exists():
                print(f"[WARN] Missing GT for image: {image_path.name}")
                continue
            rows, summary = process_image_case(
                image_path,
                gt_path,
                review_dir,
                crop_debug_dir,
                yolo,
                reid,
                gallery_data,
                args.conf_threshold,
                args.iou_threshold,
                args.show,
            )
            all_rows.extend(rows)
            summary_rows.append(summary)
            print(f"[OK] image {image_path.name}: recall={summary['detection_recall']:.3f} reid_acc={summary['reid_accuracy_on_matched']:.3f}")

    if args.mode in ("videos", "all"):
        for video_path in sorted(videos_dir.glob("*.mp4")):
            gt_path = gt_dir / f"{video_path.stem}.json"
            if not gt_path.exists():
                print(f"[WARN] Missing GT for video: {video_path.name}")
                continue
            rows, summary = process_video_case(
                video_path,
                gt_path,
                review_dir,
                crop_debug_dir,
                yolo,
                reid,
                gallery_data,
                args.conf_threshold,
                args.iou_threshold,
                args.max_frames,
                args.save_video,
                args.show,
            )
            all_rows.extend(rows)
            summary_rows.append(summary)
            print(f"[OK] video {video_path.name}: frames={summary['frames_processed']} recall={summary['detection_recall']:.3f} reid_acc={summary['reid_accuracy_on_matched']:.3f}")

    prediction_fields = [
        "case_name",
        "source_type",
        "frame_idx",
        "gt_emp_id",
        "gt_bbox",
        "det_bbox",
        "match_iou",
        "track_id",
        "pred_emp_id",
        "reid_score",
        "reid_second_id",
        "reid_second_score",
        "reid_margin",
        "best_emp_id",
        "best_score",
        "second_emp_id",
        "second_score",
        "margin",
        "top5_scores",
        "reid_threshold_used",
        "reid_margin_used",
        "reject_reason",
        "reid_call_method",
        "crop_shape",
        "crop_debug_path",
        "det_conf",
        "correct",
    ]
    summary_fields = [
        "case_name",
        "source_type",
        "frames_processed",
        "expected_people_total",
        "detected_people_total",
        "matched_people_total",
        "detection_recall",
        "reid_correct",
        "reid_accuracy_on_matched",
        "unknown_or_low_conf_count",
        "common_confusions",
    ]

    write_csv(reports_dir / "per_frame_predictions.csv", all_rows, prediction_fields)
    write_csv(reports_dir / "synthetic_test_summary.csv", summary_rows, summary_fields)
    write_confusion_matrix(reports_dir / "confusion_matrix.csv", all_rows)
    write_markdown_summary(reports_dir / "synthetic_test_summary.md", summary_rows)
    write_reid_debug_report(reports_dir / "reid_debug_report.md", all_rows, summary_rows, args.conf_threshold)

    if args.show:
        cv2.destroyAllWindows()

    print(f"[DONE] Review outputs: {review_dir}")
    print(f"[DONE] Reports: {reports_dir}")


if __name__ == "__main__":
    main()
