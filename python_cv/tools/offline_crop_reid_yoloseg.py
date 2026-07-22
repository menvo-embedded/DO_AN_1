# tools/offline_crop_reid_yoloseg.py
# Offline crop person từ raw video bằng YOLO-seg, phục vụ dataset Re-ID.
#
# Input:
#   RAW_VIDEO_ROOT
#
# Output review riêng:
#   REVIEW_CROP_ROOT
#
# Output mỗi video:
#   bbox_crops     : ảnh crop người thường, đạt chuẩn Re-ID
#   blur_crops     : ảnh crop người + nền blur bằng segmentation mask, đạt chuẩn Re-ID
#   rejected_crops : ảnh bị loại, lưu để kiểm tra lý do
#   debug_frames   : optional, ảnh full frame có bbox
#   metadata.csv   : thông tin truy vết
#
# Lưu ý:
# - Không gán nhãn tự động.
# - Không đụng dataset chính.
# - Bạn review ảnh OK rồi tự copy vào:
#     DATASET_CROPS_ROOT\NV001...
#
# Test 1 session:
#   D:\UV4\anaconda3\python.exe .\tools\offline_crop_reid_yoloseg.py --session raw_3zone_test_20s --every 5 --max-frames 200
#
# Chạy toàn bộ:
#   D:\UV4\anaconda3\python.exe .\tools\offline_crop_reid_yoloseg.py --all --every 25

import sys
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_VIDEO_ROOT, REVIEW_CROP_ROOT, YOLO_WEIGHTS


# ============================================================
# PATH CONFIG
# ============================================================

RAW_ROOT = RAW_VIDEO_ROOT
OUTPUT_ROOT = REVIEW_CROP_ROOT

# Ưu tiên model seg tốt hơn cho crop offline.
# Nếu chưa có yolo11s-seg.pt trong project thì fallback về YOLO_WEIGHTS hiện tại.
CROP_YOLO_WEIGHTS = ROOT_DIR / "yolo11s-seg.pt"

if not CROP_YOLO_WEIGHTS.exists():
    CROP_YOLO_WEIGHTS = YOLO_WEIGHTS


# ============================================================
# YOLO / CROP CONFIG
# ============================================================

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}

PERSON_CLASS_ID = 0

# Crop offline nên ưu tiên chất lượng hơn tốc độ
YOLO_CONF = 0.30
YOLO_IMGSZ = 1280

# ============================================================
# RE-ID QUALITY FILTER - STRICT MODE
# ============================================================
# Chỉ lưu ảnh đủ tiêu chuẩn vào bbox_crops / blur_crops.
# Ảnh nhỏ/quá xấu sẽ đưa vào rejected_crops để bạn kiểm tra.

# Kích thước crop tối thiểu.
# 96x154, 113x174 sẽ bị loại vì chưa đủ tốt cho train Re-ID chính.
MIN_CROP_W = 90
MIN_CROP_H = 220
MIN_AREA = 25000

# Tỉ lệ người đứng hợp lý: height / width
# Quá thấp: crop ngang/lấy nửa người/ngồi quá thấp.
# Quá cao: crop quá mảnh/cắt thiếu thân.
MIN_HW_RATIO = 1.35
MAX_HW_RATIO = 3.80

# Không lấy người bị dính sát biên vì thường bị cụt đầu/chân/thân.
ALLOW_BORDER_CROP = False
BORDER_MARGIN = 8

# Mask người phải chiếm đủ vùng crop.
# Nếu mask quá nhỏ nghĩa là crop nhiều nền hoặc segmentation lỗi.
MIN_MASK_AREA_RATIO = 0.28

# Padding quanh người để không cắt sát tay/chân/đầu.
PAD_RATIO_X = 0.08
PAD_RATIO_Y = 0.08

# Blur nền trong crop
BLUR_KERNEL = 31
MASK_THRESHOLD = 0.50
MASK_FEATHER = 9

# Ảnh bị loại sẽ lưu vào rejected_crops để kiểm tra, không dùng train.
SAVE_REJECTED = True

# Optional debug full-frame
SAVE_FULL_DEBUG_EVERY = 0
# 0 = không lưu debug frame
# ví dụ 200 = mỗi 200 frame lưu 1 ảnh full frame có bbox/mask

JPEG_QUALITY = 95


def parse_zone(video_path: Path) -> str:
    name = video_path.name.lower()

    if "zone1" in name:
        return "zone1"

    if "zone2" in name:
        return "zone2"

    if "zone3" in name:
        return "zone3"

    return "unknown_zone"


def list_videos(session: str | None, all_sessions: bool):
    if all_sessions:
        return sorted([
            p for p in RAW_ROOT.rglob("*")
            if p.suffix.lower() in VIDEO_EXTS
        ])

    if session is None:
        raise ValueError("Cần truyền --session hoặc dùng --all")

    session_dir = RAW_ROOT / session

    if not session_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy session: {session_dir}")

    return sorted([
        p for p in session_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    ])


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def pad_box(x1, y1, x2, y2, frame_w, frame_h):
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * PAD_RATIO_X)
    pad_y = int(bh * PAD_RATIO_Y)

    return clamp_box(
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
        frame_w,
        frame_h,
    )


def is_reid_valid_crop(x1, y1, x2, y2, frame_w, frame_h, mask_full=None):
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_CROP_W:
        return False, "too_narrow"

    if crop_h < MIN_CROP_H:
        return False, "too_short"

    if crop_w * crop_h < MIN_AREA:
        return False, "too_small_area"

    hw_ratio = crop_h / max(1, crop_w)

    if hw_ratio < MIN_HW_RATIO:
        return False, "bad_ratio_too_wide"

    if hw_ratio > MAX_HW_RATIO:
        return False, "bad_ratio_too_tall"

    if not ALLOW_BORDER_CROP:
        if (
            x1 <= BORDER_MARGIN
            or y1 <= BORDER_MARGIN
            or x2 >= frame_w - BORDER_MARGIN
            or y2 >= frame_h - BORDER_MARGIN
        ):
            return False, "touch_border"

    if mask_full is not None:
        mask_crop = mask_full[y1:y2, x1:x2]

        if mask_crop is None or mask_crop.size == 0:
            return False, "empty_mask"

        mask_area_ratio = float((mask_crop > MASK_THRESHOLD).sum()) / float(crop_w * crop_h)

        if mask_area_ratio < MIN_MASK_AREA_RATIO:
            return False, f"low_mask_ratio_{mask_area_ratio:.2f}"

    return True, "ok"


def make_blur_crop(frame, mask_full, crop_box):
    x1, y1, x2, y2 = crop_box

    crop = frame[y1:y2, x1:x2].copy()

    if crop is None or crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]

    if mask_full is None:
        return crop

    mask_crop = mask_full[y1:y2, x1:x2]

    if mask_crop.shape[:2] != (crop_h, crop_w):
        mask_crop = cv2.resize(mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    mask_crop = (mask_crop > MASK_THRESHOLD).astype(np.float32)

    # Feather mask để viền người mềm hơn, bớt răng cưa.
    if MASK_FEATHER > 0:
        k = MASK_FEATHER if MASK_FEATHER % 2 == 1 else MASK_FEATHER + 1
        mask_crop = cv2.GaussianBlur(mask_crop, (k, k), 0)
        mask_crop = np.clip(mask_crop, 0.0, 1.0)

    mask_3 = np.repeat(mask_crop[:, :, None], 3, axis=2)

    # Blur background trong crop
    k = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    blurred = cv2.GaussianBlur(crop, (k, k), 0)

    out = crop.astype(np.float32) * mask_3 + blurred.astype(np.float32) * (1.0 - mask_3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def draw_debug(frame, items):
    out = frame.copy()

    for item in items:
        x1, y1, x2, y2 = item["box"]
        conf = item["conf"]
        reason = item["reason"]

        color = (0, 255, 0) if reason == "ok" else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            out,
            f"{conf:.2f} {reason}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def get_masks_from_result(result, frame_h, frame_w):
    """
    Trả về list mask full-frame dạng float 0..1.
    Nếu model không có mask thì return [].
    """
    if result.masks is None or result.masks.data is None:
        return []

    masks = result.masks.data.cpu().numpy()
    full_masks = []

    for m in masks:
        m_resized = cv2.resize(
            m.astype(np.float32),
            (frame_w, frame_h),
            interpolation=cv2.INTER_LINEAR,
        )
        full_masks.append(m_resized)

    return full_masks


def save_rejected_crop(
    frame,
    reject_dir: Path,
    zone: str,
    video_stem: str,
    frame_idx: int,
    fps: float,
    person_idx: int,
    reason: str,
    box,
):
    if not SAVE_REJECTED:
        return

    x1, y1, x2, y2 = box
    reject_crop = frame[y1:y2, x1:x2].copy()

    if reject_crop is None or reject_crop.size == 0:
        return

    time_sec = frame_idx / fps if fps and fps > 0 else 0

    reject_name = (
        f"{zone}_{video_stem}_"
        f"f{frame_idx:08d}_"
        f"t{time_sec:08.2f}_"
        f"p{person_idx:02d}_"
        f"{reason}.jpg"
    )

    cv2.imwrite(
        str(reject_dir / reject_name),
        reject_crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )


def process_video(model, video_path: Path, out_root: Path, every: int, max_frames: int | None):
    zone = parse_zone(video_path)
    session_name = video_path.parent.name
    video_stem = video_path.stem

    video_out_root = out_root / session_name / zone / video_stem

    bbox_dir = video_out_root / "bbox_crops"
    blur_dir = video_out_root / "blur_crops"
    reject_dir = video_out_root / "rejected_crops"
    debug_dir = video_out_root / "debug_frames"

    bbox_dir.mkdir(parents=True, exist_ok=True)
    blur_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_REJECTED:
        reject_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_FULL_DEBUG_EVERY > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = video_out_root / "metadata.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[FAIL] Cannot open video: {video_path}")
        return {
            "video": str(video_path),
            "frames": 0,
            "saved_bbox": 0,
            "saved_blur": 0,
            "rejected": 0,
            "status": "open_failed",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print()
    print("========== PROCESS VIDEO ==========")
    print("Video :", video_path)
    print("Zone  :", zone)
    print("Size  :", f"{frame_w}x{frame_h}")
    print("FPS   :", fps)
    print("Frames:", total_frames)
    print("Every :", every)
    print("BBox  :", bbox_dir)
    print("Blur  :", blur_dir)
    print("Reject:", reject_dir)
    print("===================================")

    frame_idx = 0
    read_count = 0
    saved_bbox = 0
    saved_blur = 0
    rejected_count = 0
    skipped = {}

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "bbox_file",
            "blur_file",
            "session",
            "zone",
            "video_file",
            "frame_idx",
            "time_sec",
            "person_idx",
            "conf",
            "reason",
            "x1",
            "y1",
            "x2",
            "y2",
            "crop_w",
            "crop_h",
            "source_w",
            "source_h",
            "has_mask",
            "model",
        ])

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            read_count += 1

            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            result = model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )[0]

            debug_items = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()

                full_masks = get_masks_from_result(result, frame_h, frame_w)

                person_idx = 0

                for det_idx, (box, conf, c) in enumerate(zip(xyxy, confs, cls)):
                    if int(c) != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = box.astype(int).tolist()
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, frame_w, frame_h)

                    px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, frame_w, frame_h)

                    mask_full = None

                    if det_idx < len(full_masks):
                        mask_full = full_masks[det_idx]

                    valid, reason = is_reid_valid_crop(
                        px1,
                        py1,
                        px2,
                        py2,
                        frame_w,
                        frame_h,
                        mask_full=mask_full,
                    )

                    debug_items.append({
                        "box": (px1, py1, px2, py2),
                        "conf": float(conf),
                        "reason": reason,
                    })

                    if not valid:
                        skipped[reason] = skipped.get(reason, 0) + 1
                        rejected_count += 1

                        save_rejected_crop(
                            frame=frame,
                            reject_dir=reject_dir,
                            zone=zone,
                            video_stem=video_stem,
                            frame_idx=frame_idx,
                            fps=fps,
                            person_idx=person_idx,
                            reason=reason,
                            box=(px1, py1, px2, py2),
                        )

                        person_idx += 1
                        continue

                    bbox_crop = frame[py1:py2, px1:px2].copy()

                    if bbox_crop is None or bbox_crop.size == 0:
                        skipped["empty_crop"] = skipped.get("empty_crop", 0) + 1
                        rejected_count += 1
                        person_idx += 1
                        continue

                    crop_h, crop_w = bbox_crop.shape[:2]

                    time_sec = frame_idx / fps if fps and fps > 0 else 0

                    base_name = (
                        f"{zone}_{video_stem}_"
                        f"f{frame_idx:08d}_"
                        f"t{time_sec:08.2f}_"
                        f"p{person_idx:02d}_"
                        f"conf{float(conf):.2f}"
                    )

                    bbox_path = bbox_dir / f"{base_name}_bbox.jpg"
                    blur_path = blur_dir / f"{base_name}_blur.jpg"

                    ok_bbox = cv2.imwrite(
                        str(bbox_path),
                        bbox_crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )

                    if not ok_bbox:
                        skipped["write_bbox_failed"] = skipped.get("write_bbox_failed", 0) + 1
                        rejected_count += 1
                        person_idx += 1
                        continue

                    saved_bbox += 1

                    blur_crop = make_blur_crop(
                        frame=frame,
                        mask_full=mask_full,
                        crop_box=(px1, py1, px2, py2),
                    )

                    has_mask = mask_full is not None

                    ok_blur = False

                    if blur_crop is not None:
                        ok_blur = cv2.imwrite(
                            str(blur_path),
                            blur_crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                        )

                    if ok_blur:
                        saved_blur += 1
                        blur_file_str = str(blur_path)
                    else:
                        blur_file_str = ""

                    writer.writerow([
                        str(bbox_path),
                        blur_file_str,
                        session_name,
                        zone,
                        video_path.name,
                        frame_idx,
                        round(time_sec, 3),
                        person_idx,
                        round(float(conf), 4),
                        reason,
                        px1,
                        py1,
                        px2,
                        py2,
                        crop_w,
                        crop_h,
                        frame_w,
                        frame_h,
                        int(has_mask),
                        str(CROP_YOLO_WEIGHTS),
                    ])

                    person_idx += 1

            if SAVE_FULL_DEBUG_EVERY > 0 and frame_idx % SAVE_FULL_DEBUG_EVERY == 0:
                dbg = draw_debug(frame, debug_items)
                dbg_path = debug_dir / f"debug_f{frame_idx:08d}.jpg"
                cv2.imwrite(str(dbg_path), dbg)

            if frame_idx % (every * 50) == 0:
                print(
                    f"[PROGRESS] {video_path.name} "
                    f"frame={frame_idx}/{total_frames} "
                    f"bbox={saved_bbox} blur={saved_blur} rejected={rejected_count}"
                )

            frame_idx += 1

    cap.release()

    print(
        f"[DONE] {video_path.name} | read={read_count} | "
        f"bbox={saved_bbox} | blur={saved_blur} | rejected={rejected_count}"
    )

    if skipped:
        print("[SKIPPED]", skipped)

    return {
        "video": str(video_path),
        "frames": read_count,
        "saved_bbox": saved_bbox,
        "saved_blur": saved_blur,
        "rejected": rejected_count,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Tên session trong raw_videos_full",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Xử lý toàn bộ session/video",
    )

    parser.add_argument(
        "--every",
        type=int,
        default=25,
        help="Lấy 1 frame mỗi N frame. 25 frame với video 25FPS = khoảng 1 ảnh/giây.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame/video để test.",
    )

    args = parser.parse_args()

    print("========== OFFLINE REID YOLO-SEG CROP ==========")
    print("RAW_ROOT   :", RAW_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("YOLO       :", CROP_YOLO_WEIGHTS)
    print("CONF       :", YOLO_CONF)
    print("IMGSZ      :", YOLO_IMGSZ)
    print("EVERY      :", args.every)
    print("STRICT     :", f"min={MIN_CROP_W}x{MIN_CROP_H}, area>={MIN_AREA}, ratio={MIN_HW_RATIO}-{MAX_HW_RATIO}")
    print("===============================================")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.session, args.all)

    print(f"[INFO] Found {len(videos)} videos")

    if not videos:
        return

    model = YOLO(str(CROP_YOLO_WEIGHTS))

    summary = []

    for video_path in videos:
        try:
            s = process_video(
                model=model,
                video_path=video_path,
                out_root=OUTPUT_ROOT,
                every=max(1, args.every),
                max_frames=args.max_frames,
            )
            summary.append(s)

        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")
            summary.append({
                "video": str(video_path),
                "frames": 0,
                "saved_bbox": 0,
                "saved_blur": 0,
                "rejected": 0,
                "status": f"error: {e}",
            })

    print()
    print("========== SUMMARY ==========")

    total_bbox = 0
    total_blur = 0
    total_rejected = 0

    for s in summary:
        print(
            f"{s['status']:12s} | "
            f"bbox={s['saved_bbox']:6d} | "
            f"blur={s['saved_blur']:6d} | "
            f"rejected={s['rejected']:6d} | "
            f"{s['video']}"
        )

        total_bbox += int(s["saved_bbox"])
        total_blur += int(s["saved_blur"])
        total_rejected += int(s["rejected"])

    print("-----------------------------")
    print("TOTAL BBOX    :", total_bbox)
    print("TOTAL BLUR    :", total_blur)
    print("TOTAL REJECTED:", total_rejected)
    print("OUTPUT        :", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
