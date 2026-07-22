"""
Run final warehouse-access demo from a local video.

This script keeps the production architecture intact:
- local video replaces live camera input
- YOLO detects people
- project Tracker/ByteTrack assigns track IDs
- FusionLayer handles simulated RFID + entry crossing matching
- current body Re-ID gallery is used read-only
- CSV/JSON logs and annotated MP4 are written to outputs/final_demo/

Example:
  python tools/run_final_demo_from_video.py `
    --video "D:\\VID_20260507_014519.mp4" `
    --simulate-rfid "NV001@2.0,NV002@5.0"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]  # python_cv/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_VIDEO = Path("D:/VID_20260507_014519.mp4")
OUTPUT_DIR = ROOT_DIR / "outputs" / "final_demo"
ZONE_NAME = "zone1"
ZONE_NUMBER = 1
DEFAULT_YOLO_CONF = 0.4

cv2 = None
Detections = None
YOLO = None
EMPLOYEE_NAMES: dict[str, str] = {}
UID_MAP: dict[str, str] = {}
YOLO_CLASSES = [0]
YOLO_CONF = DEFAULT_YOLO_CONF
YOLO_IMG_SIZE = 640
YOLO_WEIGHTS = ROOT_DIR.parent / "yolo11n-seg.pt"
fusion_module = None
FusionLayer = None
Gallery = None
ReIDEngine = None
Tracker = None


def load_runtime_dependencies():
    global cv2, Detections, YOLO
    global EMPLOYEE_NAMES, UID_MAP, YOLO_CLASSES, YOLO_CONF, YOLO_IMG_SIZE, YOLO_WEIGHTS
    global fusion_module, FusionLayer, Gallery, ReIDEngine, Tracker

    try:
        import cv2 as _cv2
        from supervision import Detections as _Detections
        from ultralytics import YOLO as _YOLO

        from config.settings import (
            EMPLOYEE_NAMES as _EMPLOYEE_NAMES,
            UID_MAP as _UID_MAP,
            YOLO_CLASSES as _YOLO_CLASSES,
            YOLO_CONF as _YOLO_CONF,
            YOLO_IMG_SIZE as _YOLO_IMG_SIZE,
            YOLO_WEIGHTS as _YOLO_WEIGHTS,
        )
        import fusion.fusion_layer as _fusion_module
        from fusion.fusion_layer import FusionLayer as _FusionLayer
        from reid.gallery import Gallery as _Gallery
        from reid.reid_engine import ReIDEngine as _ReIDEngine
        from tracking.tracker import Tracker as _Tracker

    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"[ERROR] Missing Python dependency: {missing}\n"
            "Install dependencies first, for example:\n"
            "  pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    Detections = _Detections
    YOLO = _YOLO
    EMPLOYEE_NAMES = _EMPLOYEE_NAMES
    UID_MAP = _UID_MAP
    YOLO_CLASSES = _YOLO_CLASSES
    YOLO_CONF = _YOLO_CONF
    YOLO_IMG_SIZE = _YOLO_IMG_SIZE
    YOLO_WEIGHTS = _YOLO_WEIGHTS
    fusion_module = _fusion_module
    FusionLayer = _FusionLayer
    Gallery = _Gallery
    ReIDEngine = _ReIDEngine
    Tracker = _Tracker


@dataclass
class SimulatedRFID:
    employee_id: str
    second: float
    uid: str
    fired: bool = False


class ReadOnlyGallery:
    """Gallery adapter that lets FusionLayer read current gallery without saving to disk."""

    def __init__(self, source: Gallery):
        self._data = source.all()

    def get(self, employee_id: str) -> list:
        return list(self._data.get(employee_id, []))

    def all(self) -> dict:
        return {employee_id: list(embeds) for employee_id, embeds in self._data.items()}

    def employees(self) -> list:
        return list(self._data.keys())

    def update(self, employee_id: str, embedding: np.ndarray):
        # Production FusionLayer updates the live gallery after a match. For final demo
        # video we intentionally keep gallery.pkl unchanged.
        return None


class DemoDatabase:
    """Small in-memory DB facade for FusionLayer and exportable demo logs."""

    def __init__(self):
        self.current_video_sec = 0.0
        self.records: list[dict[str, Any]] = []
        self.presence: dict[str, dict[str, Any]] = {}
        self.messages: deque[dict[str, Any]] = deque(maxlen=30)

    def _append(self, kind: str, level: str, text: str, **data):
        row = {
            "kind": kind,
            "level": level,
            "video_sec": round(float(self.current_video_sec), 3),
            "message": text,
            **data,
        }
        self.records.append(row)
        self.messages.append(row)
        return row

    def log_rfid_event(
        self,
        uid: str,
        employee_id: str | None,
        timestamp: str,
        device: str = "",
        zone: int = 1,
    ):
        label = employee_id or "UNKNOWN"
        self._append(
            "rfid_event",
            "info",
            f"RFID {label}",
            uid=uid,
            employee_id=employee_id,
            timestamp=timestamp,
            device=device,
            zone=zone,
        )

    def log_unknown_uid(self, uid: str, timestamp: datetime):
        self.log_rfid_event(uid, None, timestamp.isoformat(), zone=ZONE_NUMBER)

    def log_entry(
        self,
        employee_id: str,
        track_id,
        entry_time: datetime,
        fusion_score: float,
        zone: int = 1,
    ):
        name = EMPLOYEE_NAMES.get(employee_id, employee_id)
        self._append(
            "entry",
            "ok",
            f"ENTRY {employee_id} {name} score={fusion_score:.3f}",
            employee_id=employee_id,
            employee_name=name,
            track_id=str(track_id),
            entry_time=entry_time.isoformat(),
            fusion_score=float(fusion_score),
            zone=zone,
        )

    def log_anomaly(
        self,
        atype: str,
        employee_id: str | None,
        track_id,
        detail: str,
        timestamp: datetime,
    ):
        self._append(
            "anomaly",
            "warn",
            f"ANOMALY {atype} {track_id or ''}".strip(),
            anomaly_type=atype,
            employee_id=employee_id,
            track_id=None if track_id is None else str(track_id),
            detail=detail,
            timestamp=timestamp.isoformat(),
        )

    def update_presence(
        self,
        employee_id: str,
        current_zone: int,
        track_key,
        source: str = "cv",
        status: str = "active",
    ):
        self.presence[employee_id] = {
            "employee_id": employee_id,
            "employee_name": EMPLOYEE_NAMES.get(employee_id, employee_id),
            "current_zone": int(current_zone),
            "track_key": str(track_key),
            "source": source,
            "status": status,
            "video_sec": round(float(self.current_video_sec), 3),
        }

    def remove_presence(self, employee_id: str):
        self.presence.pop(employee_id, None)

    def close(self):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final RFID+CV demo from a local video without RFID hardware."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input video path.",
    )
    parser.add_argument(
        "--simulate-rfid",
        default="",
        help='Simulated events, e.g. "NV001@2.0,NV002@5.0". Omit for no_rfid_intruder mode.',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for annotated video and CSV/JSON logs.",
    )
    parser.add_argument(
        "--entry-line-axis",
        choices=("x", "y"),
        default="x",
        help="Use x for a vertical line, y for a horizontal line.",
    )
    parser.add_argument(
        "--entry-line-ratio",
        type=float,
        default=0.50,
        help="Entry line position as a ratio of frame width/height when --entry-line-px is omitted.",
    )
    parser.add_argument(
        "--entry-line-px",
        type=int,
        default=None,
        help="Entry line position in pixels.",
    )
    parser.add_argument(
        "--crossing-cooldown",
        type=float,
        default=2.0,
        help="Seconds before the same track can trigger another crossing.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=None,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--reid-every",
        type=int,
        default=10,
        help="Refresh Re-ID ranking for each track every N processed frames.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame. Use 1 for full video.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional limit for quick tests.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated frames while processing; press q to stop.",
    )
    return parser.parse_args()


def parse_simulated_rfid(text: str) -> list[SimulatedRFID]:
    if not text.strip():
        return []

    uid_by_employee = {employee_id: uid for uid, employee_id in UID_MAP.items()}
    events: list[SimulatedRFID] = []

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        if "@" not in item:
            raise ValueError(f"Bad simulated RFID item: {item!r}. Expected EMPLOYEE@SECOND.")

        employee_id, second_text = item.split("@", 1)
        employee_id = employee_id.strip().upper()

        if employee_id not in uid_by_employee:
            known = ", ".join(sorted(uid_by_employee))
            raise ValueError(f"No UID_MAP entry for {employee_id}. Known employees: {known}")

        events.append(
            SimulatedRFID(
                employee_id=employee_id,
                second=float(second_text),
                uid=uid_by_employee[employee_id],
            )
        )

    return sorted(events, key=lambda x: x.second)


def configure_demo_fusion_runtime():
    fusion_module.ENABLE_MQTT = True
    fusion_module.RFID_EXPIRE_WINDOW = 60 * 60.0
    fusion_module.INTRUDER_WINDOW = 60 * 60.0


def rank_body(fusion: FusionLayer, crop_bgr: np.ndarray) -> dict[str, Any]:
    try:
        return fusion._rank_body(crop_bgr)
    except Exception:
        return {
            "best_id": None,
            "best_score": 0.0,
            "second_id": None,
            "second_score": 0.0,
            "margin": 0.0,
            "identified": None,
        }


def clip_bbox(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crossed_line(previous: float | None, current: float, line_px: int) -> bool:
    if previous is None:
        return False

    return (previous < line_px <= current) or (previous > line_px >= current)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]):
    y = max(18, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    db: DemoDatabase,
    mode: str,
    video_sec: float,
    frame_idx: int,
    overlay_seconds: float = 8.0,
):
    h, w = frame.shape[:2]
    status = f"FINAL DEMO | {mode} | t={video_sec:6.2f}s | frame={frame_idx}"
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(frame, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

    y = 58
    for row in list(db.messages)[-5:]:
        age = video_sec - float(row.get("video_sec", video_sec))
        if age > overlay_seconds:
            continue

        color = (0, 220, 80) if row.get("level") == "ok" else (0, 220, 255)
        if row.get("kind") == "anomaly":
            color = (0, 80, 255)

        text = str(row.get("message", ""))[:95]
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 24


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flush_unmatched_demo_events(fusion: FusionLayer, db: DemoDatabase, final_timestamp: datetime):
    for crossing in list(fusion._crossings):
        if not crossing.get("assigned"):
            db.current_video_sec = float(crossing.get("video_sec", db.current_video_sec))
            db.log_anomaly(
                "no_rfid_intruder",
                None,
                crossing.get("track_key"),
                "No matching simulated RFID event for this crossing",
                crossing.get("ts", final_timestamp),
            )
            crossing["assigned"] = True

    for employee_id, queue in list(fusion._pending.items()):
        if queue:
            rfid = queue[-1]
            db.log_anomaly(
                "rfid_no_crossing",
                employee_id,
                None,
                "Simulated RFID event did not match any video crossing",
                rfid.get("ts", final_timestamp),
            )
            queue.clear()


def main() -> int:
    args = parse_args()
    load_runtime_dependencies()

    args.frame_stride = max(1, int(args.frame_stride))
    args.reid_every = max(1, int(args.reid_every))
    if args.yolo_conf is None:
        args.yolo_conf = YOLO_CONF

    video_path = args.video.expanduser()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 2

    simulated_rfid = parse_simulated_rfid(args.simulate_rfid)
    mode = "simulated_rfid" if simulated_rfid else "no_rfid_intruder"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = args.output_dir / f"final_demo_annotated_{run_id}.mp4"
    events_csv = args.output_dir / f"final_demo_events_{run_id}.csv"
    tracks_csv = args.output_dir / f"final_demo_tracks_{run_id}.csv"
    summary_json = args.output_dir / f"final_demo_summary_{run_id}.json"

    configure_demo_fusion_runtime()

    print(f"[INFO] Input video : {video_path}")
    print(f"[INFO] Output dir  : {args.output_dir}")
    print(f"[INFO] Mode        : {mode}")

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

    writer_fps = fps / args.frame_stride
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, writer_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[ERROR] Cannot open output video writer: {output_video}")
        return 2

    line_px = args.entry_line_px
    if line_px is None:
        base = width if args.entry_line_axis == "x" else height
        line_px = int(max(0.0, min(1.0, args.entry_line_ratio)) * base)

    print(f"[INFO] Video       : {width}x{height} @ {fps:.2f} FPS, frames={frame_count}")
    print(f"[INFO] Entry line  : {args.entry_line_axis}={line_px}")

    db = DemoDatabase()
    reid = ReIDEngine()
    gallery = ReadOnlyGallery(Gallery())
    fusion = FusionLayer(reid, gallery, db, face_engine=None)
    model = YOLO(str(YOLO_WEIGHTS))
    tracker = Tracker()

    track_prev_position: dict[int, float] = {}
    track_last_crossing: dict[int, float] = {}
    track_rank_cache: dict[int, dict[str, Any]] = {}
    track_rows: list[dict[str, Any]] = []

    base_timestamp = datetime.now().astimezone()
    raw_frame_idx = 0
    processed_frame_idx = 0
    start_wall = time.time()
    stopped_by_user = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            raw_frame_idx += 1
            video_sec = (raw_frame_idx - 1) / fps
            if args.max_seconds is not None and video_sec > args.max_seconds:
                break

            if (raw_frame_idx - 1) % args.frame_stride != 0:
                continue

            processed_frame_idx += 1
            frame_timestamp = base_timestamp + timedelta(seconds=video_sec)
            db.current_video_sec = video_sec

            for event in simulated_rfid:
                if not event.fired and video_sec >= event.second:
                    event.fired = True
                    event_timestamp = base_timestamp + timedelta(seconds=event.second)
                    db.current_video_sec = event.second
                    fusion.on_rfid_event(
                        {
                            "uid": event.uid,
                            "timestamp": event_timestamp.isoformat(),
                            "dt": event_timestamp,
                            "device": "video-sim",
                            "zone": ZONE_NUMBER,
                        }
                    )
                    db.current_video_sec = video_sec

            results = model(
                frame,
                classes=YOLO_CLASSES,
                conf=args.yolo_conf,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
            )[0]
            detections = Detections.from_ultralytics(results)
            tracked = tracker.update(detections)
            annotated = frame.copy()

            if args.entry_line_axis == "x":
                cv2.line(annotated, (line_px, 0), (line_px, height), (0, 0, 220), 2)
                line_label_pos = (min(width - 180, line_px + 8), 62)
            else:
                cv2.line(annotated, (0, line_px), (width, line_px), (0, 0, 220), 2)
                line_label_pos = (10, max(82, line_px - 8))
            cv2.putText(
                annotated,
                f"ENTRY LINE {args.entry_line_axis}={line_px}",
                line_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 220),
                2,
                cv2.LINE_AA,
            )

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

                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                center_pos = cx if args.entry_line_axis == "x" else cy
                previous_pos = track_prev_position.get(track_id)

                should_refresh_reid = (
                    track_id not in track_rank_cache
                    or processed_frame_idx % args.reid_every == 0
                )
                if should_refresh_reid:
                    track_rank_cache[track_id] = rank_body(fusion, crop)

                rank = track_rank_cache.get(track_id, {})
                best_id = rank.get("best_id")
                best_score = float(rank.get("best_score") or 0.0)
                margin = float(rank.get("margin") or 0.0)

                if crossed_line(previous_pos, center_pos, line_px):
                    last_crossing = track_last_crossing.get(track_id, -1e9)
                    if video_sec - last_crossing >= args.crossing_cooldown:
                        track_last_crossing[track_id] = video_sec
                        track_key = f"{ZONE_NAME}:{track_id}"
                        db._append(
                            "crossing",
                            "info",
                            f"CROSSING {track_key}",
                            track_id=track_key,
                            timestamp=frame_timestamp.isoformat(),
                            best_id=best_id,
                            best_score=best_score,
                            margin=margin,
                        )
                        before_count = len(fusion._crossings)
                        fusion.on_entry_crossing(ZONE_NAME, track_id, crop, frame_timestamp)
                        if len(fusion._crossings) > before_count:
                            fusion._crossings[-1]["video_sec"] = video_sec

                        if not simulated_rfid:
                            db.log_anomaly(
                                "no_rfid_intruder",
                                None,
                                track_key,
                                "No simulated RFID event was provided",
                                frame_timestamp,
                            )
                            if fusion._crossings:
                                fusion._crossings[-1]["assigned"] = True

                track_prev_position[track_id] = center_pos

                confirmed = fusion.is_confirmed(track_id, zone=ZONE_NAME)
                confirmed_label = fusion.get_label(track_id, zone=ZONE_NAME)
                employee_label = confirmed_label if confirmed else (best_id or "Unknown")
                color = (0, 220, 80) if confirmed else (0, 165, 255)

                label = f"T{track_id} {employee_label} score={best_score:.3f}"
                if confirmed:
                    label = f"T{track_id} CONFIRMED {employee_label} score={best_score:.3f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                draw_label(annotated, label, x1, y1 - 8, color)

                track_rows.append(
                    {
                        "frame": raw_frame_idx,
                        "processed_frame": processed_frame_idx,
                        "video_sec": round(float(video_sec), 3),
                        "track_id": track_id,
                        "track_key": f"{ZONE_NAME}:{track_id}",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": round(cx, 2),
                        "center_y": round(cy, 2),
                        "best_id": best_id,
                        "best_score": best_score,
                        "second_id": rank.get("second_id"),
                        "second_score": rank.get("second_score"),
                        "margin": margin,
                        "confirmed": bool(confirmed),
                        "confirmed_label": confirmed_label if confirmed else "",
                    }
                )

            draw_overlay(annotated, db, mode, video_sec, raw_frame_idx)
            writer.write(annotated)

            if args.show:
                cv2.imshow("Final Demo From Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break

            if processed_frame_idx % 50 == 0:
                print(
                    f"[INFO] processed={processed_frame_idx} "
                    f"frame={raw_frame_idx} t={video_sec:.1f}s"
                )

    finally:
        final_ts = base_timestamp + timedelta(seconds=db.current_video_sec)
        if simulated_rfid:
            flush_unmatched_demo_events(fusion, db, final_ts)

        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    write_csv(events_csv, db.records)
    write_csv(tracks_csv, track_rows)

    summary = {
        "run_id": run_id,
        "mode": mode,
        "input_video": str(video_path),
        "output_video": str(output_video),
        "events_csv": str(events_csv),
        "tracks_csv": str(tracks_csv),
        "stopped_by_user": stopped_by_user,
        "processed_frames": processed_frame_idx,
        "raw_frames_seen": raw_frame_idx,
        "elapsed_wall_sec": round(time.time() - start_wall, 3),
        "entry_line_axis": args.entry_line_axis,
        "entry_line_px": line_px,
        "simulated_rfid": [
            {"employee_id": e.employee_id, "second": e.second, "fired": e.fired}
            for e in simulated_rfid
        ],
        "counts": {
            "records": len(db.records),
            "rfid_events": sum(1 for r in db.records if r.get("kind") == "rfid_event"),
            "crossings": sum(1 for r in db.records if r.get("kind") == "crossing"),
            "entries": sum(1 for r in db.records if r.get("kind") == "entry"),
            "anomalies": sum(1 for r in db.records if r.get("kind") == "anomaly"),
            "track_rows": len(track_rows),
        },
        "presence": list(db.presence.values()),
        "events": db.records,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Final demo video/logs created:")
    print(f"  video : {output_video}")
    print(f"  events: {events_csv}")
    print(f"  tracks: {tracks_csv}")
    print(f"  json  : {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
