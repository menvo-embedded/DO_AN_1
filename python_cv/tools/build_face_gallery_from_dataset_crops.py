# tools/build_face_gallery_from_dataset_crops.py
# Build Face Gallery từ dataset crop sạch:
# DATASET_CROPS_ROOT\NV001 ... NV005
#
# Mixed/hard:
# DATASET_CROPS_ROOT\mixed\NV001 ... NV005
# chỉ dùng để evaluation, không đưa vào gallery chính.

import sys
import time
import pickle
import random
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# PROJECT ROOT
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# CONFIG
# ============================================================
from config.settings import (
    DATASET_CROPS_ROOT,
    FACE_DET_SIZE,
    FACE_DET_THRESH,
    FACE_GALLERY_PATH,
    FACE_MODEL_NAME,
)

DATASET_ROOT = DATASET_CROPS_ROOT

GALLERY_OUT = FACE_GALLERY_PATH
DEBUG_DIR = ROOT_DIR / "outputs" / "debug_frames" / "build_face_gallery_from_dataset_crops"

MODEL_NAME = FACE_MODEL_NAME
USE_GPU = True
DET_SIZE = FACE_DET_SIZE
DET_THRESH = FACE_DET_THRESH

EMPLOYEE_NAMES = {
    "NV001": "Bo Man",
    "NV002": "Me Mai",
    "NV003": "Anh Minh",
    "NV004": "Chi Dung",
    "NV005": "Toi",
}

EMPLOYEE_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Chỉ lấy tối đa số embedding tốt nhất mỗi người để gallery không quá nặng
MAX_GALLERY_PER_EMP = 120

# Chia clean data: phần lớn build gallery, một phần test clean
CLEAN_TEST_RATIO = 0.20
RANDOM_SEED = 42

# Lọc bbox mặt
MIN_FACE_W = 18
MIN_FACE_H = 18
MAX_FACE_RATIO = 2.4

# Matching
TOPK_MEAN = 5
FACE_THRESHOLD = 0.38
FACE_MARGIN = 0.09


# ============================================================
# UTILS
# ============================================================
def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)

    if norm < 1e-12:
        return emb

    return emb / norm


def init_insightface():
    print("[INFO] Loading InsightFace...")
    print(f"[INFO] MODEL_NAME = {MODEL_NAME}")
    print(f"[INFO] USE_GPU = {USE_GPU}")
    print(f"[INFO] DET_SIZE = {DET_SIZE}")
    print(f"[INFO] DET_THRESH = {DET_THRESH}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )

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


def list_images(folder: Path):
    paths = []

    if not folder.exists():
        return paths

    for ext in IMAGE_EXTS:
        paths.extend(folder.glob(f"*{ext}"))
        paths.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(set(paths))


def clip_bbox(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))

    return x1, y1, x2, y2


def check_face_box(bbox):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, "invalid"

    if w < MIN_FACE_W or h < MIN_FACE_H:
        return False, f"small_{w}x{h}"

    ratio = max(w / h, h / w)
    if ratio > MAX_FACE_RATIO:
        return False, f"bad_ratio_{ratio:.2f}"

    return True, "ok"


def extract_best_face(app, image_path: Path):
    img = cv2.imread(str(image_path))

    if img is None:
        return None, "read_fail"

    h, w = img.shape[:2]

    try:
        faces = app.get(img)
    except Exception as e:
        return None, f"app_get_error_{e}"

    candidates = []

    for face in faces:
        bbox = clip_bbox(face.bbox, w, h)

        ok, reason = check_face_box(bbox)
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
        det_score = float(face.det_score)
        quality = det_score * area

        candidates.append(
            {
                "image": img,
                "bbox": bbox,
                "embedding": embedding,
                "det_score": det_score,
                "area": area,
                "quality": quality,
                "image_path": image_path,
            }
        )

    if not candidates:
        return None, "no_valid_face"

    candidates.sort(key=lambda x: x["quality"], reverse=True)
    return candidates[0], "ok"


def save_debug(emp_id, image_path, face_info, split_name):
    out_dir = DEBUG_DIR / split_name / emp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    img = face_info["image"].copy()
    x1, y1, x2, y2 = face_info["bbox"]

    crop = img[y1:y2, x1:x2]

    safe_stem = image_path.stem.replace(" ", "_")
    full_path = out_dir / f"{safe_stem}_full.jpg"
    crop_path = out_dir / f"{safe_stem}_face.jpg"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{emp_id} det={face_info['det_score']:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(full_path), img)
    cv2.imwrite(str(crop_path), crop)


def split_clean_paths(paths, test_ratio):
    paths = list(paths)
    random.shuffle(paths)

    if len(paths) <= 5:
        return paths, []

    test_count = max(1, int(len(paths) * test_ratio))
    test_paths = paths[:test_count]
    train_paths = paths[test_count:]

    return train_paths, test_paths


def collect_face_infos(app, emp_id, paths, split_name):
    face_infos = []
    failed = []

    for p in paths:
        face_info, status = extract_best_face(app, p)

        if status != "ok":
            failed.append((emp_id, str(p), status))
            continue

        save_debug(emp_id, p, face_info, split_name)
        face_infos.append(face_info)

    return face_infos, failed


def build_gallery(app, dataset_root):
    gallery = {
        "version": "insightface_gallery_dataset_crops_v1",
        "model_name": MODEL_NAME,
        "det_size": tuple(DET_SIZE),
        "det_thresh": float(DET_THRESH),
        "embedding_dim": 512,
        "source": str(dataset_root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {},
        "employees": {},
    }

    clean_test_set = {}
    all_failed = []

    print("\n========== BUILD GALLERY FROM CLEAN ROOT ==========")

    for emp_id in EMPLOYEE_IDS:
        emp_name = EMPLOYEE_NAMES.get(emp_id, emp_id)
        clean_dir = dataset_root / emp_id

        all_paths = list_images(clean_dir)

        train_paths, test_paths = split_clean_paths(all_paths, CLEAN_TEST_RATIO)
        clean_test_set[emp_id] = test_paths

        print(f"\n[EMP] {emp_id} | {emp_name}")
        print(f"      clean total={len(all_paths)} train={len(train_paths)} clean_test={len(test_paths)}")

        face_infos, failed = collect_face_infos(app, emp_id, train_paths, "gallery_train")
        all_failed.extend(failed)

        # Sort theo chất lượng mặt rồi lấy tối đa MAX_GALLERY_PER_EMP
        face_infos.sort(key=lambda x: x["quality"], reverse=True)
        selected = face_infos[:MAX_GALLERY_PER_EMP]

        embeddings = [x["embedding"] for x in selected]
        scores = [x["det_score"] for x in selected]

        gallery["employees"][emp_id] = {
            "name": emp_name,
            "embeddings": embeddings,
            "scores": scores,
            "source_total": len(all_paths),
            "train_total": len(train_paths),
            "faces_detected": len(face_infos),
            "selected_count": len(embeddings),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        gallery["counts"][emp_id] = len(embeddings)

        print(f"      faces detected in train={len(face_infos)}")
        print(f"      selected gallery embeddings={len(embeddings)}")

    return gallery, clean_test_set, all_failed


def save_gallery(gallery, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gallery["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if path.name == "insightface_gallery.pkl":
            backup_path = path.parent / f"insightface_gallery_backup_{timestamp}.pkl"
        else:
            backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        print(f"[BACKUP] Existing gallery backed up: {backup_path}")

    with open(path, "wb") as f:
        pickle.dump(gallery, f)

    print(f"\n[SAVE] Gallery saved: {path}")


def match_embedding(query_embedding, gallery):
    query_embedding = normalize_embedding(query_embedding)

    ranking = []

    for emp_id, item in gallery["employees"].items():
        embs = item.get("embeddings", [])

        if not embs:
            continue

        sims = [float(np.dot(query_embedding, normalize_embedding(e))) for e in embs]
        sims_sorted = sorted(sims, reverse=True)

        topk = sims_sorted[: min(TOPK_MEAN, len(sims_sorted))]
        topk_mean = float(np.mean(topk))

        ranking.append(
            {
                "employee_id": emp_id,
                "name": item.get("name", emp_id),
                "score": topk_mean,
                "max_score": float(sims_sorted[0]),
                "top_scores": topk,
            }
        )

    if not ranking:
        return {
            "status": "NO_GALLERY",
            "employee_id": "Unknown",
            "name": "Unknown",
            "score": 0.0,
            "second_score": -1.0,
            "margin": 0.0,
            "ranking": [],
        }

    ranking.sort(key=lambda x: x["score"], reverse=True)

    best = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None

    best_score = best["score"]
    second_score = second["score"] if second else -1.0
    margin = best_score - second_score if second else 999.0

    if len(ranking) == 1:
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
            "ranking": ranking,
        }

    return {
        "status": "UNKNOWN",
        "employee_id": "Unknown",
        "name": "Unknown",
        "score": best_score,
        "second_score": second_score,
        "margin": margin,
        "ranking": ranking,
    }


def evaluate_paths(app, gallery, test_paths_by_emp, split_name):
    total = 0
    correct = 0
    unknown = 0
    wrong = 0
    skipped = 0

    per_emp = {}

    print(f"\n========== EVALUATION: {split_name} ==========")

    for true_emp_id in EMPLOYEE_IDS:
        paths = test_paths_by_emp.get(true_emp_id, [])

        per_emp[true_emp_id] = {
            "total": 0,
            "correct": 0,
            "unknown": 0,
            "wrong": 0,
            "skipped": 0,
        }

        for p in paths:
            face_info, status = extract_best_face(app, p)

            if status != "ok":
                skipped += 1
                per_emp[true_emp_id]["skipped"] += 1
                print(f"[SKIP] true={true_emp_id} file={p.name} reason={status}")
                continue

            save_debug(true_emp_id, p, face_info, split_name)

            result = match_embedding(face_info["embedding"], gallery)
            pred_id = result["employee_id"]

            total += 1
            per_emp[true_emp_id]["total"] += 1

            if pred_id == true_emp_id:
                correct += 1
                per_emp[true_emp_id]["correct"] += 1
                flag = "OK"
            elif pred_id == "Unknown":
                unknown += 1
                per_emp[true_emp_id]["unknown"] += 1
                flag = "UNKNOWN"
            else:
                wrong += 1
                per_emp[true_emp_id]["wrong"] += 1
                flag = "WRONG"

            print(
                f"[{flag}] true={true_emp_id} pred={pred_id} "
                f"score={result['score']:.3f} second={result['second_score']:.3f} "
                f"margin={result['margin']:.3f} file={p.name}"
            )

    acc = correct / total if total > 0 else 0.0

    print(f"\n===== SUMMARY {split_name} =====")
    print(f"Total valid test : {total}")
    print(f"Correct          : {correct}")
    print(f"Unknown          : {unknown}")
    print(f"Wrong            : {wrong}")
    print(f"Skipped no face  : {skipped}")
    print(f"Accuracy         : {acc:.4f}")

    print(f"\n===== PER EMPLOYEE {split_name} =====")
    for emp_id, s in per_emp.items():
        emp_total = s["total"]
        emp_acc = s["correct"] / emp_total if emp_total > 0 else 0.0
        print(
            f"{emp_id}: total={emp_total} correct={s['correct']} "
            f"unknown={s['unknown']} wrong={s['wrong']} skipped={s['skipped']} acc={emp_acc:.4f}"
        )


def collect_mixed_test_set(dataset_root):
    mixed_root = dataset_root / "mixed"
    mixed_test_set = {}

    for emp_id in EMPLOYEE_IDS:
        mixed_dir = mixed_root / emp_id
        mixed_test_set[emp_id] = list_images(mixed_dir)

    return mixed_test_set


def print_dataset_summary(dataset_root):
    print("========== DATASET SUMMARY ==========")

    for emp_id in EMPLOYEE_IDS:
        clean_count = len(list_images(dataset_root / emp_id))
        mixed_count = len(list_images(dataset_root / "mixed" / emp_id))

        print(f"{emp_id}: clean={clean_count} | mixed/hard={mixed_count}")


# ============================================================
# MAIN
# ============================================================
def main():
    global MAX_GALLERY_PER_EMP, CLEAN_TEST_RATIO

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=str(DATASET_ROOT))
    parser.add_argument("--gallery_out", default=str(GALLERY_OUT))
    parser.add_argument("--max_per_emp", type=int, default=MAX_GALLERY_PER_EMP)
    parser.add_argument("--clean_test_ratio", type=float, default=CLEAN_TEST_RATIO)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    gallery_out = Path(args.gallery_out)
    max_per_emp = int(args.max_per_emp)
    clean_test_ratio = float(args.clean_test_ratio)

    MAX_GALLERY_PER_EMP = max_per_emp
    CLEAN_TEST_RATIO = clean_test_ratio

    random.seed(RANDOM_SEED)

    if not dataset_root.exists():
        raise RuntimeError(f"Không thấy dataset_root: {dataset_root}")

    print_dataset_summary(dataset_root)

    app = init_insightface()

    gallery, clean_test_set, failed = build_gallery(app, dataset_root)
    save_gallery(gallery, gallery_out)

    print("\n========== FAILED TRAIN IMAGES ==========")
    print(f"Total failed: {len(failed)}")
    for emp_id, path, reason in failed[:80]:
        print(f"{emp_id} | {reason} | {path}")
    if len(failed) > 80:
        print(f"... còn {len(failed) - 80} ảnh failed nữa")

    evaluate_paths(app, gallery, clean_test_set, "clean_test")

    mixed_test_set = collect_mixed_test_set(dataset_root)
    evaluate_paths(app, gallery, mixed_test_set, "mixed_hard_test")

    print("\n[DONE]")
    print(f"Gallery output : {gallery_out}")
    print(f"Debug output   : {DEBUG_DIR}")
    print(f"Gallery model  : {MODEL_NAME}")
    print(f"DET_SIZE       : {DET_SIZE}")
    print(f"DET_THRESH     : {DET_THRESH}")


if __name__ == "__main__":
    main()
