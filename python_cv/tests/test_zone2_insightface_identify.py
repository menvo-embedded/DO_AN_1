# tests/test_zone2_insightface_identify.py
# Test nhận diện khuôn mặt Zone 2 bằng InsightFace gallery
#
# Chạy:
# cd C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv
# D:\UV4\anaconda3\python.exe .\tests\test_zone2_insightface_identify.py

import sys
import time
import re
import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# PROJECT ROOT
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP


# ============================================================
# CONFIG - CHỈNH TOÀN BỘ Ở ĐÂY
# ============================================================
ROTATE_MODE = None

MODEL_NAME = "buffalo_sc"
USE_GPU = True

DET_SIZE = (640, 640)
DET_THRESH = 0.50

DETECT_EVERY_N_FRAMES = 1

MIN_FACE_W = 25
MIN_FACE_H = 25
MAX_FACE_RATIO = 2.2

# Matching config
FACE_THRESHOLD = 0.38
FACE_MARGIN = 0.09
TOPK_MEAN = 5

# Nếu gallery chỉ có 1 người, margin sẽ không bắt buộc
REQUIRE_MARGIN_WHEN_ONLY_ONE_EMP = False

# Lưu debug
AUTO_SAVE_RESULT = True
SAVE_COOLDOWN_SEC = 2.0

GALLERY_PATH = ROOT_DIR / "data" / "face_gallery" / "insightface_gallery.pkl"
OUTPUT_DIR = ROOT_DIR / "outputs" / "debug_frames" / "insightface_zone2_identify"

WINDOW_NAME = "Zone 2 InsightFace Identify"
MAX_READ_FAILS = 60


# ============================================================
# UTILS
# ============================================================
def mask_rtsp_url(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def rotate_frame(frame):
    if ROTATE_MODE is None:
        return frame
    if ROTATE_MODE == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ROTATE_MODE == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ROTATE_MODE == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm < 1e-12:
        return emb
    return emb / norm


def open_zone2_camera():
    print("[INFO] Opening Zone 2 RTSP camera...")
    print(f"[INFO] CAM_ZONE2_RTSP = {mask_rtsp_url(CAM_ZONE2_RTSP)}")

    cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[WARN] CAP_FFMPEG failed, retry default backend...")
        cap.release()
        cap = cv2.VideoCapture(CAM_ZONE2_RTSP)

    if not cap.isOpened():
        raise RuntimeError("Không mở được Zone 2 RTSP. Kiểm tra CAM_ZONE2_RTSP.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def init_insightface():
    print("[INFO] Loading InsightFace...")
    print(f"[INFO] MODEL_NAME = {MODEL_NAME}")
    print(f"[INFO] USE_GPU = {USE_GPU}")
    print(f"[INFO] DET_SIZE = {DET_SIZE}")
    print(f"[INFO] DET_THRESH = {DET_THRESH}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]

    try:
        app = FaceAnalysis(
            name=MODEL_NAME,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
    except TypeError:
        try:
            app = FaceAnalysis(
                name=MODEL_NAME,
                providers=providers,
            )
        except TypeError:
            app = FaceAnalysis(name=MODEL_NAME)

    ctx_id = 0 if USE_GPU else -1

    try:
        app.prepare(
            ctx_id=ctx_id,
            det_size=DET_SIZE,
            det_thresh=DET_THRESH,
        )
    except TypeError:
        app.prepare(
            ctx_id=ctx_id,
            det_size=DET_SIZE,
        )

    print("[INFO] InsightFace ready.")
    return app


def load_gallery():
    if not GALLERY_PATH.exists():
        raise FileNotFoundError(f"Không thấy gallery: {GALLERY_PATH}")

    with open(GALLERY_PATH, "rb") as f:
        gallery = pickle.load(f)

    employees = gallery.get("employees", {})

    clean_gallery = {}

    for emp_id, item in employees.items():
        name = item.get("name", emp_id)
        embeddings = item.get("embeddings", [])

        clean_embs = []
        for emb in embeddings:
            clean_embs.append(normalize_embedding(emb))

        if len(clean_embs) > 0:
            clean_gallery[emp_id] = {
                "name": name,
                "embeddings": clean_embs,
            }

    if len(clean_gallery) == 0:
        raise RuntimeError("Gallery rỗng, chưa có embedding nào.")

    print("[GALLERY] Loaded:")
    for emp_id, item in clean_gallery.items():
        print(f"          {emp_id} | {item['name']} | {len(item['embeddings'])} embeddings")

    return clean_gallery


def clip_bbox(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))

    return x1, y1, x2, y2


def check_face_box(bbox):
    x1, y1, x2, y2 = bbox

    face_w = x2 - x1
    face_h = y2 - y1

    if face_w <= 0 or face_h <= 0:
        return False, "invalid"

    if face_w < MIN_FACE_W or face_h < MIN_FACE_H:
        return False, f"small {face_w}x{face_h}"

    ratio = max(face_w / face_h, face_h / face_w)
    if ratio > MAX_FACE_RATIO:
        return False, f"bad_ratio {ratio:.2f}"

    return True, "ok"


def choose_best_face(faces, frame_w, frame_h):
    candidates = []

    for face in faces:
        bbox = clip_bbox(face.bbox, frame_w, frame_h)
        ok, _ = check_face_box(bbox)

        if not ok:
            continue

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            continue

        embedding = normalize_embedding(embedding)

        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(face.det_score)
        rank_score = score * area

        candidates.append(
            {
                "face": face,
                "bbox": bbox,
                "score": score,
                "area": area,
                "embedding": embedding,
                "rank_score": rank_score,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    return candidates[0]


def match_face(query_embedding, gallery):
    """
    So query embedding với toàn bộ gallery.
    Score mỗi nhân viên = top-k mean cosine similarity.
    """
    results = []

    for emp_id, item in gallery.items():
        embs = item["embeddings"]

        sims = [float(np.dot(query_embedding, emb)) for emb in embs]
        sims_sorted = sorted(sims, reverse=True)

        topk = sims_sorted[: min(TOPK_MEAN, len(sims_sorted))]
        topk_mean = float(np.mean(topk))
        max_score = float(sims_sorted[0])

        results.append(
            {
                "employee_id": emp_id,
                "name": item["name"],
                "score": topk_mean,
                "max_score": max_score,
                "top_scores": topk,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0]
    second = results[1] if len(results) > 1 else None

    best_score = best["score"]
    second_score = second["score"] if second else -1.0
    margin = best_score - second_score if second else 999.0

    if len(results) == 1 and not REQUIRE_MARGIN_WHEN_ONLY_ONE_EMP:
        is_match = best_score >= FACE_THRESHOLD
    else:
        is_match = best_score >= FACE_THRESHOLD and margin >= FACE_MARGIN

    if is_match:
        return {
            "status": "MATCH",
            "employee_id": best["employee_id"],
            "name": best["name"],
            "score": best_score,
            "second_score": second_score,
            "margin": margin,
            "ranking": results,
        }

    return {
        "status": "UNKNOWN",
        "employee_id": "Unknown",
        "name": "Unknown",
        "score": best_score,
        "second_score": second_score,
        "margin": margin,
        "ranking": results,
    }


def save_result(frame_draw, frame_raw, bbox, match_result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    label = match_result["employee_id"]
    prefix = f"zone2_identify_{label}_{ts}_{ms:03d}"

    x1, y1, x2, y2 = bbox
    crop = frame_raw[y1:y2, x1:x2]

    full_path = OUTPUT_DIR / f"{prefix}_full_bbox.jpg"
    crop_path = OUTPUT_DIR / f"{prefix}_crop.jpg"

    cv2.imwrite(str(full_path), frame_draw)
    cv2.imwrite(str(crop_path), crop)

    print("[SAVE] Identify result:")
    print(f"       {full_path}")
    print(f"       {crop_path}")


def draw_overlay(frame, raw_count, valid_count, fps, last_result_text):
    lines = [
        "ZONE 2 INSIGHTFACE IDENTIFY",
        f"Raw faces: {raw_count} | Valid face: {valid_count}",
        f"Model: {MODEL_NAME} | Det size: {DET_SIZE} | Det thresh: {DET_THRESH}",
        f"Face threshold: {FACE_THRESHOLD} | Margin: {FACE_MARGIN} | TopK: {TOPK_MEAN}",
        f"Rotate: {ROTATE_MODE} | GPU: {USE_GPU} | FPS: {fps:.1f}",
        "Q/ESC: quit | S: save current full frame",
    ]

    if last_result_text:
        lines.append(last_result_text)

    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


def save_snapshot(frame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    path = OUTPUT_DIR / f"zone2_manual_snapshot_{ts}_{ms:03d}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"[SAVE] Snapshot: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    if not CAM_ZONE2_RTSP:
        raise RuntimeError("CAM_ZONE2_RTSP rỗng. Kiểm tra .env/settings.py")

    gallery = load_gallery()
    app = init_insightface()
    cap = open_zone2_camera()

    print("[INFO] Zone 2 InsightFace identify started.")
    print("[INFO] Gallery:", GALLERY_PATH)
    print("[INFO] Q/ESC: thoát | S: lưu ảnh hiện tại.")

    read_fails = 0
    frame_idx = 0

    last_faces = []
    last_save_time = 0.0
    last_result_text = ""

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            read_fails += 1
            print(f"[WARN] Cannot read frame. fail={read_fails}/{MAX_READ_FAILS}")

            if read_fails >= MAX_READ_FAILS:
                print("[WARN] Too many read fails. Reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = open_zone2_camera()
                read_fails = 0

            time.sleep(0.03)
            continue

        read_fails = 0
        frame_idx += 1

        frame = rotate_frame(frame)
        frame_raw = frame.copy()
        frame_draw = frame.copy()

        frame_h, frame_w = frame.shape[:2]

        now = time.time()
        dt = now - prev_time
        prev_time = now

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            try:
                last_faces = app.get(frame_raw)
            except Exception as e:
                print(f"[WARN] InsightFace get() error: {e}")
                last_faces = []

        raw_count = len(last_faces)

        best_face = choose_best_face(last_faces, frame_w, frame_h)
        valid_count = 1 if best_face is not None else 0

        if best_face is not None:
            bbox = best_face["bbox"]
            x1, y1, x2, y2 = bbox
            det_score = best_face["score"]
            embedding = best_face["embedding"]

            match_result = match_face(embedding, gallery)

            emp_id = match_result["employee_id"]
            name = match_result["name"]
            score = match_result["score"]
            second_score = match_result["second_score"]
            margin = match_result["margin"]
            status = match_result["status"]

            if status == "MATCH":
                color = (0, 255, 0)
                label = f"{emp_id} | {name} | sim={score:.3f}"
            else:
                color = (0, 0, 255)
                label = f"Unknown | best={score:.3f}"

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_draw,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_draw,
                f"det={det_score:.2f} second={second_score:.3f} margin={margin:.3f}",
                (x1, min(frame_h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            kps = getattr(best_face["face"], "kps", None)
            if kps is not None:
                for px, py in kps.astype(int):
                    cv2.circle(frame_draw, (int(px), int(py)), 2, (0, 255, 255), -1)

            top_text = []
            for r in match_result["ranking"][:3]:
                top_text.append(f"{r['employee_id']}:{r['score']:.3f}")
            last_result_text = "Top: " + " | ".join(top_text)

            print(
                f"[IDENTIFY] {status} => {emp_id} | {name} | "
                f"score={score:.3f} second={second_score:.3f} margin={margin:.3f}"
            )

            if AUTO_SAVE_RESULT:
                current_time = time.time()
                if current_time - last_save_time >= SAVE_COOLDOWN_SEC:
                    save_result(frame_draw, frame_raw, bbox, match_result)
                    last_save_time = current_time

        draw_overlay(frame_draw, raw_count, valid_count, fps, last_result_text)

        cv2.imshow(WINDOW_NAME, frame_draw)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("[INFO] Quit.")
            break

        if key in [ord("s"), ord("S")]:
            save_snapshot(frame_draw)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
