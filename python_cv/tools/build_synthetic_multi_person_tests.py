import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


EMPLOYEE_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_CASES = [
    "case_01_2people_far",
    "case_02_2people_close",
    "case_03_3people",
    "case_04_partial_occlusion",
    "case_05_small_far",
    "case_06_crossing_paths",
    "case_07_group_entry",
    "case_08_hard_mixed",
]


@dataclass
class SourceCandidate:
    emp_id: str
    path: Path
    width: int
    height: int
    aspect: float
    blur: float


@dataclass
class PersonCutout:
    emp_id: str
    path: Path
    image_bgr: np.ndarray
    alpha: np.ndarray
    blur: float
    mask_aspect: float
    area_ratio: float
    quality: float

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])


@dataclass
class PersonSpec:
    emp_id: str
    source_idx: int
    start_frame: int
    end_frame: int
    x0: float
    y0: float
    x1: float
    y1: float
    target_h: int
    z: int = 0
    phase: float = 0.0
    wobble_x: float = 8.0
    wobble_y: float = 3.0


def laplacian_var(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_seg_weights() -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    candidates = [
        root_dir / "yolo11n-seg.pt",
        root_dir.parent / "yolo11n-seg.pt",
        root_dir / "models" / "yolo" / "yolo11n-seg.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find YOLO segmentation weights. Expected yolo11n-seg.pt in "
        "python_cv/ or repo root."
    )


def prefilter_candidate(path: Path, emp_id: str, min_blur: float) -> SourceCandidate | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if h < 160 or w < 60:
        return None

    aspect = h / max(w, 1)
    if aspect < 1.2 or aspect > 4.5:
        return None

    blur = laplacian_var(image)
    if blur < min_blur:
        return None

    return SourceCandidate(
        emp_id=emp_id,
        path=path,
        width=w,
        height=h,
        aspect=aspect,
        blur=blur,
    )


def refine_person_mask_with_grabcut(image: np.ndarray, yolo_mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = yolo_mask.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return None

    kernel = np.ones((5, 5), np.uint8)
    probable_fg = mask_u8.astype(bool)
    sure_bg = ~cv2.dilate(mask_u8, kernel, iterations=2).astype(bool)

    grab_mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    grab_mask[sure_bg] = cv2.GC_BGD
    grab_mask[probable_fg] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            grab_mask,
            None,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None

    refined = np.logical_or(grab_mask == cv2.GC_FGD, grab_mask == cv2.GC_PR_FGD)
    refined = np.logical_and(refined, yolo_mask)

    if int(refined.sum()) == 0:
        return None

    return refined


def keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    clean = np.zeros_like(mask_u8)
    clean[labels == largest_label] = 255
    return clean


def make_cutout(candidate: SourceCandidate, model: YOLO) -> PersonCutout | None:
    image = cv2.imread(str(candidate.path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    result = model.predict(
        image,
        classes=[0],
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    if result.masks is None or result.masks.data is None:
        return None

    masks = result.masks.data.detach().cpu().numpy()
    if masks.size == 0:
        return None

    best_mask = None
    best_area = 0

    for raw_mask in masks:
        mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = mask >= 0.50
        area = int(binary.sum())
        if area > best_area:
            best_area = area
            best_mask = binary

    if best_mask is None or best_area == 0:
        return None

    image_area = h * w
    area_ratio = best_area / max(image_area, 1)
    if area_ratio < 0.14 or area_ratio > 0.92:
        return None

    ys, xs = np.where(best_mask)
    if xs.size == 0 or ys.size == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_w = x2 - x1
    box_h = y2 - y1
    if box_h < 130 or box_w < 42:
        return None

    box_aspect = box_h / max(box_w, 1)
    if box_aspect < 2.05 or box_aspect > 5.2:
        return None

    pad = max(6, int(max(box_w, box_h) * 0.025))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    cut_image = image[y1p:y2p, x1p:x2p].copy()
    cut_mask = best_mask[y1p:y2p, x1p:x2p].astype(np.uint8) * 255
    cut_mask = keep_largest_component(cut_mask)

    kernel = np.ones((3, 3), np.uint8)
    cut_mask = cv2.morphologyEx(cut_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cut_mask = cv2.erode(cut_mask, kernel, iterations=1)
    cut_mask = cv2.GaussianBlur(cut_mask, (7, 7), 0)
    alpha = cut_mask.astype(np.float32) / 255.0

    if float(alpha.mean()) < 0.10:
        return None

    quality = min(candidate.blur, 220.0)
    quality += 100.0 if 2.55 <= box_aspect <= 4.7 else 0.0
    quality -= abs(area_ratio - 0.48) * 80.0

    return PersonCutout(
        emp_id=candidate.emp_id,
        path=candidate.path,
        image_bgr=cut_image,
        alpha=alpha,
        blur=candidate.blur,
        mask_aspect=box_aspect,
        area_ratio=area_ratio,
        quality=quality,
    )


def load_source_cutouts(
    dataset_root: Path,
    max_source_per_id: int,
    model: YOLO,
) -> dict[str, list[PersonCutout]]:
    sources: dict[str, list[PersonCutout]] = {}
    min_needed = 5
    target_per_id = max(min_needed, max_source_per_id)
    min_blur = 18.0

    for emp_id in EMPLOYEE_IDS:
        folder = dataset_root / emp_id
        candidates: list[SourceCandidate] = []

        if not folder.exists():
            print(f"[WARN] Missing dataset folder: {folder}")
            sources[emp_id] = []
            continue

        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            candidate = prefilter_candidate(path, emp_id, min_blur=min_blur)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda x: x.blur, reverse=True)

        cutouts: list[PersonCutout] = []
        scan_limit = min(len(candidates), max(80, target_per_id * 5))
        for candidate in candidates[:scan_limit]:
            cutout = make_cutout(candidate, model)
            if cutout is None:
                continue
            cutouts.append(cutout)

        cutouts.sort(key=lambda x: x.quality, reverse=True)
        sources[emp_id] = cutouts[:target_per_id]
        print(
            f"[INFO] {emp_id}: {len(candidates)} prefiltered crops, "
            f"{len(cutouts)} YOLO-seg cutouts"
        )

    missing = [emp_id for emp_id, rows in sources.items() if len(rows) < min_needed]
    if missing:
        raise RuntimeError(
            "Not enough valid YOLO-seg person cutouts for: "
            + ", ".join(missing)
            + ". Need at least 5 per ID."
        )

    return sources


def prepare_output_dirs(output_root: Path, clean_output: bool) -> dict[str, Path]:
    dirs = {
        "images": output_root / "images",
        "videos": output_root / "videos",
        "ground_truth": output_root / "ground_truth",
        "source_samples": output_root / "source_samples",
        "backgrounds": output_root / "backgrounds",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for key in ["images", "videos", "ground_truth", "source_samples"]:
            if dirs[key].exists():
                shutil.rmtree(dirs[key])

    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    return dirs


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - width) // 2)
    y0 = max(0, (nh - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def build_default_background(width: int, height: int) -> np.ndarray:
    bg = np.full((height, width, 3), (205, 208, 205), dtype=np.uint8)
    floor_y = int(height * 0.66)

    cv2.rectangle(bg, (0, floor_y), (width, height), (186, 190, 187), -1)
    cv2.line(bg, (0, floor_y), (width, floor_y), (150, 154, 151), 2)

    for y in range(floor_y + 38, height, 54):
        cv2.line(bg, (0, y), (width, y), (174, 178, 175), 1)
    for x in range(100, width, 160):
        cv2.line(bg, (x, height), (x + 90, floor_y), (176, 180, 177), 1)

    shelf_color = (135, 142, 138)
    box_color = (178, 158, 122)
    for shelf_x in [70, width - 330]:
        cv2.rectangle(bg, (shelf_x, 130), (shelf_x + 230, 360), shelf_color, 3)
        for y in [205, 285]:
            cv2.line(bg, (shelf_x, y), (shelf_x + 230, y), shelf_color, 2)
        for idx in range(4):
            bx = shelf_x + 25 + (idx % 2) * 92
            by = 150 + (idx // 2) * 95
            cv2.rectangle(bg, (bx, by), (bx + 58, by + 45), box_color, -1)
            cv2.rectangle(bg, (bx, by), (bx + 58, by + 45), (145, 130, 100), 1)

    return cv2.GaussianBlur(bg, (3, 3), 0)


def load_background(background_dir: Path, width: int, height: int, rng: random.Random) -> np.ndarray:
    candidates = [
        p for p in background_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if candidates:
        path = rng.choice(sorted(candidates))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None and image.size > 0:
            print(f"[INFO] Using background: {path}")
            return resize_cover(image, width, height)

    bg = build_default_background(width, height)
    generated = background_dir / f"generated_default_{width}x{height}.jpg"
    cv2.imwrite(str(generated), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[INFO] No usable background found; generated: {generated}")
    return bg


def save_source_samples(sources: dict[str, list[PersonCutout]], output_dir: Path, per_id: int = 5) -> None:
    for emp_id, rows in sources.items():
        emp_dir = output_dir / emp_id
        emp_dir.mkdir(parents=True, exist_ok=True)
        for idx, cutout in enumerate(rows[:per_id]):
            shutil.copy2(cutout.path, emp_dir / f"{idx:02d}_source_{cutout.path.name}")

            alpha_u8 = np.clip(cutout.alpha * 255.0, 0, 255).astype(np.uint8)
            rgba = cv2.cvtColor(cutout.image_bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha_u8
            cv2.imwrite(str(emp_dir / f"{idx:02d}_cutout_{cutout.path.stem}.png"), rgba)


def interpolate(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def choose_cutout(sources: dict[str, list[PersonCutout]], emp_id: str, idx: int) -> PersonCutout:
    rows = sources[emp_id]
    return rows[idx % len(rows)]


def paste_person(
    frame: np.ndarray,
    cutout: PersonCutout,
    center_x: float,
    bottom_y: float,
    target_h: int,
) -> list[int] | None:
    h, w = cutout.image_bgr.shape[:2]
    scale = target_h / h
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    person = cv2.resize(cutout.image_bgr, (target_w, target_h), interpolation=interp)
    alpha = cv2.resize(cutout.alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0)

    nonzero = alpha > 0.03
    if not np.any(nonzero):
        return None

    ys, xs = np.where(nonzero)
    px1, px2 = int(xs.min()), int(xs.max()) + 1
    py1, py2 = int(ys.min()), int(ys.max()) + 1

    x1 = int(round(center_x - target_w / 2))
    y1 = int(round(bottom_y - target_h))
    x2 = x1 + target_w
    y2 = y1 + target_h

    fh, fw = frame.shape[:2]
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(fw, x2)
    cy2 = min(fh, y2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    shadow_y = min(fh - 1, int(round(bottom_y - 5)))
    shadow_w = max(18, int((px2 - px1) * 0.34))
    cv2.ellipse(
        frame,
        (int(round(center_x)), shadow_y),
        (shadow_w, 9),
        0,
        0,
        360,
        (126, 129, 126),
        -1,
    )

    sx1 = cx1 - x1
    sy1 = cy1 - y1
    sx2 = sx1 + (cx2 - cx1)
    sy2 = sy1 + (cy2 - cy1)

    roi = frame[cy1:cy2, cx1:cx2].astype(np.float32)
    src = person[sy1:sy2, sx1:sx2].astype(np.float32)
    a = alpha[sy1:sy2, sx1:sx2].astype(np.float32)[:, :, None]
    blended = src * a + roi * (1.0 - a)
    frame[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

    bx1 = max(0, x1 + px1)
    by1 = max(0, y1 + py1)
    bx2 = min(fw, x1 + px2)
    by2 = min(fh, y1 + py2)
    if bx2 <= bx1 or by2 <= by1:
        return None
    return [int(bx1), int(by1), int(bx2), int(by2)]


def position_for(spec: PersonSpec, frame_idx: int) -> tuple[float, float]:
    if spec.end_frame <= spec.start_frame:
        t = 0.0
    else:
        t = (frame_idx - spec.start_frame) / (spec.end_frame - spec.start_frame)
        t = max(0.0, min(1.0, t))

    x = interpolate(spec.x0, spec.x1, t)
    y = interpolate(spec.y0, spec.y1, t)
    x += math.sin((frame_idx * 0.17) + spec.phase) * spec.wobble_x
    y += math.cos((frame_idx * 0.13) + spec.phase) * spec.wobble_y
    return x, y


def make_case_specs(case_name: str, width: int, height: int, fps: int, duration: float) -> list[PersonSpec]:
    total = int(round(fps * duration))
    floor = int(height * 0.86)

    if case_name == "case_01_2people_far":
        return [
            PersonSpec("NV001", 0, 0, total - 1, width * 0.23, floor, width * 0.25, floor - 8, 410, 1, 0.0),
            PersonSpec("NV005", 0, 0, total - 1, width * 0.75, floor, width * 0.73, floor - 4, 410, 1, 1.4),
        ]
    if case_name == "case_02_2people_close":
        return [
            PersonSpec("NV002", 0, 0, total - 1, width * 0.47, floor, width * 0.49, floor, 430, 1, 0.3),
            PersonSpec("NV004", 0, 0, total - 1, width * 0.54, floor - 2, width * 0.52, floor - 4, 430, 2, 1.1),
        ]
    if case_name == "case_03_3people":
        return [
            PersonSpec("NV001", 1, 0, total - 1, width * 0.25, floor, width * 0.27, floor - 5, 390, 1, 0.0),
            PersonSpec("NV003", 0, 0, total - 1, width * 0.50, floor + 4, width * 0.50, floor - 5, 415, 2, 1.0),
            PersonSpec("NV005", 1, 0, total - 1, width * 0.75, floor, width * 0.73, floor - 2, 390, 1, 2.0),
        ]
    if case_name == "case_04_partial_occlusion":
        return [
            PersonSpec("NV003", 1, 0, total - 1, width * 0.50, floor - 12, width * 0.51, floor - 10, 380, 1, 0.6),
            PersonSpec("NV005", 2, 0, total - 1, width * 0.56, floor + 10, width * 0.54, floor + 8, 440, 2, 1.7),
        ]
    if case_name == "case_05_small_far":
        return [
            PersonSpec("NV001", 2, 0, total - 1, width * 0.35, int(height * 0.68), width * 0.37, int(height * 0.68), 245, 1, 0.2, 4, 2),
            PersonSpec("NV004", 1, 0, total - 1, width * 0.64, int(height * 0.68), width * 0.62, int(height * 0.68), 245, 1, 1.0, 4, 2),
        ]
    if case_name == "case_06_crossing_paths":
        return [
            PersonSpec("NV002", 1, 0, total - 1, width * 0.20, floor, width * 0.78, floor - 4, 395, 2, 0.1, 5, 2),
            PersonSpec("NV005", 3, 0, total - 1, width * 0.78, floor - 8, width * 0.22, floor + 2, 395, 1, 1.9, 5, 2),
        ]
    if case_name == "case_07_group_entry":
        return [
            PersonSpec("NV001", 3, 0, total - 1, width * 0.30, floor, width * 0.34, floor - 5, 390, 1, 0.0),
            PersonSpec("NV003", 2, int(3 * fps), total - 1, -120, floor, width * 0.53, floor, 400, 2, 1.0),
            PersonSpec("NV004", 2, int(6 * fps), total - 1, width + 120, floor, width * 0.72, floor - 8, 390, 1, 2.1),
        ]
    if case_name == "case_08_hard_mixed":
        return [
            PersonSpec("NV001", 4, 0, total - 1, width * 0.30, int(height * 0.69), width * 0.32, int(height * 0.69), 260, 1, 0.4, 5, 2),
            PersonSpec("NV002", 2, 0, total - 1, width * 0.52, floor + 10, width * 0.50, floor + 8, 445, 3, 1.6, 7, 3),
            PersonSpec("NV005", 4, 0, total - 1, width * 0.58, floor - 5, width * 0.60, floor - 5, 370, 2, 2.4, 6, 2),
        ]

    raise ValueError(f"Unknown case: {case_name}")


def render_case(
    case_name: str,
    sources: dict[str, list[PersonCutout]],
    dirs: dict[str, Path],
    background: np.ndarray,
    width: int,
    height: int,
    fps: int,
    duration: float,
) -> None:
    total_frames = int(round(fps * duration))
    specs = make_case_specs(case_name, width, height, fps, duration)
    video_path = dirs["videos"] / f"{case_name}.mp4"
    image_path = dirs["images"] / f"{case_name}.jpg"
    gt_path = dirs["ground_truth"] / f"{case_name}.json"

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {video_path}")

    cutouts = {
        (spec.emp_id, spec.source_idx): choose_cutout(sources, spec.emp_id, spec.source_idx)
        for spec in specs
    }

    frames_gt = []
    representative = total_frames // 2
    representative_frame = None

    for frame_idx in range(total_frames):
        frame = background.copy()
        objects = []

        for spec in sorted(specs, key=lambda s: s.z):
            if frame_idx < spec.start_frame or frame_idx > spec.end_frame:
                continue

            cutout = cutouts[(spec.emp_id, spec.source_idx)]
            center_x, bottom_y = position_for(spec, frame_idx)
            bbox = paste_person(frame, cutout, center_x=center_x, bottom_y=bottom_y, target_h=spec.target_h)
            if bbox is None:
                continue

            objects.append({
                "emp_id": spec.emp_id,
                "bbox": bbox,
                "source_image": str(cutout.path),
            })

        writer.write(frame)
        frames_gt.append({"frame_idx": frame_idx, "objects": objects})

        if frame_idx == representative:
            representative_frame = frame.copy()

    writer.release()

    if representative_frame is None:
        representative_frame = background.copy()
    cv2.imwrite(str(image_path), representative_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    gt = {
        "case_name": case_name,
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "duration_sec": duration,
        "frames": frames_gt,
    }
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"[OK] {case_name}: {video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate all synthetic test cases.")
    parser.add_argument("--dataset-root", default="D:/warehouse_dataset/dataset_crops")
    parser.add_argument("--output-root", default="D:/warehouse_dataset/synthetic_tests")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--max-source-per-id", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all:
        raise SystemExit("Use --all to generate the 8 requested synthetic cases.")

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dirs = prepare_output_dirs(output_root, args.clean_output)
    seg_weights = find_seg_weights()
    print(f"[INFO] Loading YOLO segmentation model: {seg_weights}")
    seg_model = YOLO(str(seg_weights))

    sources = load_source_cutouts(dataset_root, args.max_source_per_id, seg_model)
    save_source_samples(sources, dirs["source_samples"])
    background = load_background(dirs["backgrounds"], args.width, args.height, rng)

    for case_name in VIDEO_CASES:
        render_case(
            case_name=case_name,
            sources=sources,
            dirs=dirs,
            background=background,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
        )

    print("[DONE] Synthetic multi-person review set created at:", output_root)


if __name__ == "__main__":
    main()
