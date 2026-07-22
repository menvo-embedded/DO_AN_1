# tests/enroll_insightface_gallery.py
# Enroll Face Gallery bằng InsightFace/SCRFD cho Zone 2 IP Camera
#
# Ví dụ chạy:
# D:\UV4\anaconda3\python.exe .\tests\enroll_insightface_gallery.py --id NV005 --name "Toi" --target 20

import sys
import time
import re
import pickle
import argparse
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

AUTO_SAVE_FACE = True
SAVE_COOLDOWN_SEC = 1.2

TARGET_EMBEDDINGS = 20

MIN_FACE_W = 25
MIN_FACE_H = 25
MAX_FACE_RATIO = 2.2

# Tránh lưu quá nhiều frame gần giống nhau.
# Nếu mặt mới quá giống embedding vừa lưu thì bỏ qua.
SKIP_TOO_SIMILAR = True
MAX_DUPLICATE_SIM = 0.985

# Nếu trong khung hình có nhiều người, chọn mặt lớn nhất + score cao nhất.
CHOOSE_LARGEST_FACE = True

# ROI lọc vùng enroll. Mặc định tắt.
# Nếu muốn chỉ lấy mặt ở vùng bàn/giữa frame, bật USE_ROI=True rồi chỉnh ROI.
USE_ROI = False
ROI_X1 = 250
ROI_Y1 = 80
ROI_X2 = 620
ROI_Y2 = 460

GALLERY_DIR = ROOT_DIR / "data" / "face_gallery"
GALLERY_PATH = GALLERY_DIR / "insightface_gallery.pkl"

OUTPUT_DIR = ROOT_DIR / "outputs" / "debug_frames" / "insightface_gallery_enroll"
WINDOW_NAME = "Enroll InsightFace Gallery"

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


def open_zone2_camera():
    print("[INFO] Opening Zone 2 RTSP camera...")
    print(f"[INFO] CAM_ZONE2_RTSP = {mask_rtsp_url(CAM_ZONE2_RTSP)}")

    cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[WARN] CAP_FFMPEG failed, retry default backend...")
        cap.release()
        cap = cv2.VideoCapture(CAM_ZONE2_RTSP)

    if not cap.isOpened():
        raise RuntimeError("Không mở được Zone 2 RTSP. Kiểm tra .env/settings.py hoặc camera IP.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def init_insightface():
    print("[INFO] Loading InsightFace...")
    print(f"[INFO] MODEL_NAME = {MODEL_NAME}")
    print(f"[INFO] USE_GPU = {USE_GPU}")
    print(f"[INFO] DET_SIZE = {DET_SIZE}")
    print(f"[INFO] DET_THRESH = {DET_THRESH}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]

    # Tương thích nhiều version InsightFace
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
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    if not GALLERY_PATH.exists():
        return {
            "version": "insightface_gallery_v1",
            "model_name": MODEL_NAME,
            "embedding_dim": 512,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "employees": {},
        }

    with open(GALLERY_PATH, "rb") as f:
        gallery = pickle.load(f)

    if "employees" not in gallery:
        gallery["employees"] = {}

    return gallery


def save_gallery(gallery):
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    gallery["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(GALLERY_PATH, "wb") as f:
        pickle.dump(gallery, f)

    print(f"[GALLERY] Saved: {GALLERY_PATH}")


def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)

    if norm < 1e-12:
        return emb

    return emb / norm


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


def bbox_center_in_roi(bbox):
    if not USE_ROI:
        return True

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


def choose_best_face(faces, frame_w, frame_h):
    candidates = []

    for face in faces:
        bbox = clip_bbox(face.bbox, frame_w, frame_h)
        x1, y1, x2, y2 = bbox

        ok, reason = check_face_box(bbox)
        if not ok:
            continue

        if not bbox_center_in_roi(bbox):
            continue

        score = float(face.det_score)
        area = max(0, x2 - x1) * max(0, y2 - y1)

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            continue

        embedding = normalize_embedding(embedding)

        rank_score = score * area

        candidates.append(
            {
                "face": face,
                "bbox": bbox,
                "score": score,
                "area": area,
                "rank_score": rank_score,
                "embedding": embedding,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    return candidates[0]


def is_too_similar(new_embedding, session_embeddings):
    if not SKIP_TOO_SIMILAR:
        return False, 0.0

    if len(session_embeddings) == 0:
        return False, 0.0

    sims = [float(np.dot(new_embedding, old_emb)) for old_emb in session_embeddings]
    max_sim = max(sims)

    if max_sim >= MAX_DUPLICATE_SIM:
        return True, max_sim

    return False, max_sim


def save_debug_face(frame_draw, frame_raw, bbox, embedding, employee_id, saved_count, score):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emp_dir = OUTPUT_DIR / employee_id
    emp_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    prefix = f"{employee_id}_{saved_count:03d}_{ts}_{ms:03d}"

    x1, y1, x2, y2 = bbox
    crop = frame_raw[y1:y2, x1:x2]

    full_path = emp_dir / f"{prefix}_full_bbox.jpg"
    crop_path = emp_dir / f"{prefix}_crop.jpg"
    emb_path = emp_dir / f"{prefix}_embedding.npy"

    cv2.imwrite(str(full_path), frame_draw)
    cv2.imwrite(str(crop_path), crop)
    np.save(str(emb_path), embedding)

    print("[SAVE] Enroll sample:")
    print(f"       employee={employee_id} score={score:.3f} emb={embedding.shape}")
    print(f"       {full_path}")
    print(f"       {crop_path}")
    print(f"       {emb_path}")


def add_embedding_to_gallery(gallery, employee_id, employee_name, embedding, score):
    if employee_id not in gallery["employees"]:
        gallery["employees"][employee_id] = {
            "name": employee_name,
            "embeddings": [],
            "scores": [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    item = gallery["employees"][employee_id]
    item["name"] = employee_name
    item["embeddings"].append(np.asarray(embedding, dtype=np.float32))
    item["scores"].append(float(score))
    item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def get_employee_total(gallery, employee_id):
    if employee_id not in gallery["employees"]:
        return 0
    return len(gallery["employees"][employee_id].get("embeddings", []))


def draw_overlay(frame, employee_id, employee_name, raw_count, valid_count, saved_session, total_count, fps, last_msg):
    lines = [
        "ENROLL INSIGHTFACE FACE GALLERY",
        f"Employee: {employee_id} | {employee_name}",
        f"Raw faces: {raw_count} | Valid candidate: {valid_count}",
        f"Saved this session: {saved_session}/{TARGET_EMBEDDINGS} | Total in gallery: {total_count}",
        f"Model: {MODEL_NAME} | Det size: {DET_SIZE} | Det thresh: {DET_THRESH}",
        f"Rotate: {ROTATE_MODE} | GPU: {USE_GPU} | FPS: {fps:.1f}",
        "Q/ESC: quit | S: save current best face manually",
    ]

    if USE_ROI:
        lines.append(f"ROI ON: ({ROI_X1},{ROI_Y1})-({ROI_X2},{ROI_Y2})")

    if last_msg:
        lines.append(last_msg)

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


def draw_roi(frame):
    if not USE_ROI:
        return

    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "ROI",
        (ROI_X1, max(20, ROI_Y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    global TARGET_EMBEDDINGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Employee ID, ví dụ NV005")
    parser.add_argument("--name", required=True, help='Tên nhân viên, ví dụ "Toi"')
    parser.add_argument("--target", type=int, default=TARGET_EMBEDDINGS, help="Số embedding cần thu trong session")
    args = parser.parse_args()

    employee_id = args.id.strip()
    employee_name = args.name.strip()
    target = int(args.target)

    if not employee_id:
        raise RuntimeError("Employee ID rỗng.")
    if not employee_name:
        raise RuntimeError("Employee name rỗng.")

    if not CAM_ZONE2_RTSP:
        raise RuntimeError("CAM_ZONE2_RTSP rỗng. Kiểm tra .env hoặc config/settings.py")

    TARGET_EMBEDDINGS = target

    gallery = load_gallery()

    print("========== ENROLL INFO ==========")
    print("Employee ID:", employee_id)
    print("Employee Name:", employee_name)
    print("Target:", TARGET_EMBEDDINGS)
    print("Gallery:", GALLERY_PATH)
    print("Current total:", get_employee_total(gallery, employee_id))

    app = init_insightface()
    cap = open_zone2_camera()

    print("[INFO] Enroll started.")
    print("[INFO] Nhìn lên camera, xoay nhẹ trái/phải, gần/xa một chút để gallery đa dạng.")
    print("[INFO] Q/ESC: thoát | S: lưu thủ công candidate hiện tại.")

    read_fails = 0
    frame_idx = 0

    last_faces = []
    last_best = None
    last_save_time = 0.0
    last_msg = ""

    saved_session = 0
    session_embeddings = []

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

        best = choose_best_face(last_faces, frame_w, frame_h)
        last_best = best

        valid_count = 1 if best is not None else 0

        draw_roi(frame_draw)

        if best is not None:
            x1, y1, x2, y2 = best["bbox"]
            score = best["score"]
            embedding = best["embedding"]

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_draw,
                f"BEST face score={score:.2f} emb={embedding.shape[0]}d",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            kps = getattr(best["face"], "kps", None)
            if kps is not None:
                for px, py in kps.astype(int):
                    cv2.circle(frame_draw, (int(px), int(py)), 2, (0, 255, 255), -1)

            too_similar, max_sim = is_too_similar(embedding, session_embeddings)

            if AUTO_SAVE_FACE and not too_similar:
                current_time = time.time()

                if current_time - last_save_time >= SAVE_COOLDOWN_SEC:
                    add_embedding_to_gallery(gallery, employee_id, employee_name, embedding, score)
                    save_gallery(gallery)

                    saved_session += 1
                    session_embeddings.append(embedding)

                    total_count = get_employee_total(gallery, employee_id)

                    save_debug_face(
                        frame_draw=frame_draw,
                        frame_raw=frame_raw,
                        bbox=best["bbox"],
                        embedding=embedding,
                        employee_id=employee_id,
                        saved_count=total_count,
                        score=score,
                    )

                    last_save_time = current_time
                    last_msg = f"Last saved: {time.strftime('%H:%M:%S')} | sim={max_sim:.3f}"

            elif too_similar:
                last_msg = f"Skip similar face | sim={max_sim:.3f}"

        total_count = get_employee_total(gallery, employee_id)

        draw_overlay(
            frame_draw,
            employee_id=employee_id,
            employee_name=employee_name,
            raw_count=raw_count,
            valid_count=valid_count,
            saved_session=saved_session,
            total_count=total_count,
            fps=fps,
            last_msg=last_msg,
        )

        cv2.imshow(WINDOW_NAME, frame_draw)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("[INFO] Quit.")
            break

        if key in [ord("s"), ord("S")]:
            if last_best is None:
                print("[WARN] Manual save skipped: no valid face candidate.")
                last_msg = "Manual save skipped: no face"
            else:
                embedding = last_best["embedding"]
                score = last_best["score"]

                add_embedding_to_gallery(gallery, employee_id, employee_name, embedding, score)
                save_gallery(gallery)

                saved_session += 1
                session_embeddings.append(embedding)

                total_count = get_employee_total(gallery, employee_id)

                save_debug_face(
                    frame_draw=frame_draw,
                    frame_raw=frame_raw,
                    bbox=last_best["bbox"],
                    embedding=embedding,
                    employee_id=employee_id,
                    saved_count=total_count,
                    score=score,
                )

                last_msg = f"Manual saved: {time.strftime('%H:%M:%S')}"

        if saved_session >= TARGET_EMBEDDINGS:
            print("[DONE] Target reached.")
            print(f"[DONE] {employee_id} saved this session: {saved_session}")
            print(f"[DONE] Total in gallery: {get_employee_total(gallery, employee_id)}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()