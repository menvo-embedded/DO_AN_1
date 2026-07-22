# reid/face_insightface_engine.py
# InsightFace Engine dùng phụ trợ cho Zone 2
# Input: frame hoáº·c person crop
# Output: employee_id / Unknown + score

import time
import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils.logger import get_logger


log = get_logger("face")


class InsightFaceEngine:
    def __init__(
        self,
        gallery_path="data/face_gallery/insightface_gallery.pkl",
        model_name="buffalo_l",
        use_gpu=True,
        det_size=(640, 640),
        det_thresh=0.35,
        face_threshold=0.38,
        face_margin=0.09,
        topk_mean=5,
        min_face_w=25,
        min_face_h=25,
        max_face_ratio=2.2,
    ):
        self.gallery_path = Path(gallery_path)
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.det_thresh = det_thresh

        self.face_threshold = face_threshold
        self.face_margin = face_margin
        self.topk_mean = topk_mean

        self.min_face_w = min_face_w
        self.min_face_h = min_face_h
        self.max_face_ratio = max_face_ratio

        self.app = None
        self.gallery = {}
        self.gallery_metadata = {}

        self._load_model()
        self._load_gallery()

    # =========================
    # INIT
    # =========================
    def _load_model(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]

        # SCRFD yêu cầu det_size chia hết cho 32, nếu không sẽ lỗi broadcast shape.
        # Tự làm tròn về bội số 32 gần nhất (tối thiểu 32) để tránh footgun cấu hình.
        def _snap32(v):
            return max(32, int(round(v / 32.0)) * 32)
        snapped = (_snap32(self.det_size[0]), _snap32(self.det_size[1]))
        if snapped != tuple(self.det_size):
            log.warning(
                f"det_size={self.det_size} không chia hết 32 (SCRFD yêu cầu) "
                f"→ tự điều chỉnh thành {snapped}"
            )
            self.det_size = snapped

        log.info(
            f"InsightFace runtime config: model_name={self.model_name} "
            f"det_size={self.det_size} det_thresh={self.det_thresh} "
            f"providers={providers}"
        )

        self.app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )

        ctx_id = 0 if self.use_gpu else -1

        try:
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=self.det_size,
                det_thresh=self.det_thresh,
            )
        except TypeError:
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=self.det_size,
            )

        log.info("InsightFace ready")

    def _load_gallery(self):
        if not self.gallery_path.exists():
            log.warning(f"Face gallery not found: {self.gallery_path}")
            self.gallery = {}
            return

        with open(self.gallery_path, "rb") as f:
            raw_gallery = pickle.load(f)

        self._validate_gallery_metadata(raw_gallery)

        employees = raw_gallery.get("employees", {})
        clean_gallery = {}

        for emp_id, item in employees.items():
            name = item.get("name", emp_id)
            embeddings = item.get("embeddings", [])

            clean_embs = []
            for emb in embeddings:
                clean_embs.append(self._normalize(emb))

            if len(clean_embs) > 0:
                clean_gallery[emp_id] = {
                    "name": name,
                    "embeddings": clean_embs,
                }

        self.gallery = clean_gallery

        log.info(
            f"Face gallery loaded: model_name={self.gallery_metadata.get('model_name')} "
            f"det_size={self.gallery_metadata.get('det_size')} "
            f"det_thresh={self.gallery_metadata.get('det_thresh')} "
            f"embedding_dim={self.gallery_metadata.get('embedding_dim')}"
        )
        for emp_id, item in self.gallery.items():
            log.info(f"Face gallery employee: {emp_id} | {item['name']} | {len(item['embeddings'])} embeddings")

    def _validate_gallery_metadata(self, raw_gallery):
        if not isinstance(raw_gallery, dict):
            raise RuntimeError(
                f"Invalid face gallery format: {self.gallery_path}. Rebuild gallery."
            )

        gallery_model = raw_gallery.get("model_name")
        gallery_det_size = raw_gallery.get("det_size")
        gallery_det_thresh = raw_gallery.get("det_thresh")
        gallery_dim = raw_gallery.get("embedding_dim")

        self.gallery_metadata = {
            "version": raw_gallery.get("version"),
            "model_name": gallery_model,
            "det_size": gallery_det_size,
            "det_thresh": gallery_det_thresh,
            "embedding_dim": gallery_dim,
            "created_at": raw_gallery.get("created_at"),
            "counts": raw_gallery.get("counts"),
        }

        if not gallery_model:
            raise RuntimeError(
                f"Face gallery has no model_name metadata: {self.gallery_path}. "
                f"Runtime uses {self.model_name}. Rebuild gallery."
            )

        if gallery_model != self.model_name:
            raise RuntimeError(
                f"Face gallery built with {gallery_model} but runtime uses {self.model_name}. "
                f"Rebuild gallery."
            )

        if gallery_dim is not None and int(gallery_dim) != 512:
            raise RuntimeError(
                f"Face gallery embedding_dim={gallery_dim}, expected 512. Rebuild gallery."
            )

    # =========================
    # BASIC UTILS
    # =========================
    def _normalize(self, embedding):
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)

        if norm < 1e-12:
            return emb

        return emb / norm

    def _clip_bbox(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w - 1))
        y2 = max(0, min(int(y2), frame_h - 1))

        return x1, y1, x2, y2

    def _check_face_box(self, bbox):
        x1, y1, x2, y2 = bbox

        face_w = x2 - x1
        face_h = y2 - y1

        if face_w <= 0 or face_h <= 0:
            return False

        if face_w < self.min_face_w or face_h < self.min_face_h:
            return False

        ratio = max(face_w / face_h, face_h / face_w)
        if ratio > self.max_face_ratio:
            return False

        return True

    # =========================
    # FACE DETECTION
    # =========================
    def detect_faces(self, image_bgr):
        if image_bgr is None:
            return []

        try:
            faces = self.app.get(image_bgr)
        except Exception as e:
            print(f"[FaceEngine] app.get error: {e}")
            return []

        frame_h, frame_w = image_bgr.shape[:2]
        results = []

        for face in faces:
            bbox = self._clip_bbox(face.bbox, frame_w, frame_h)

            if not self._check_face_box(bbox):
                continue

            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)

            if embedding is None:
                continue

            embedding = self._normalize(embedding)

            x1, y1, x2, y2 = bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)

            results.append(
                {
                    "bbox": bbox,
                    "score": float(face.det_score),
                    "embedding": embedding,
                    "area": area,
                    "face": face,
                }
            )

        results.sort(key=lambda x: x["score"] * x["area"], reverse=True)
        return results

    # =========================
    # MATCHING
    # =========================
    def match_embedding(self, query_embedding):
        if not self.gallery:
            return {
                "status": "NO_GALLERY",
                "employee_id": "Unknown",
                "name": "Unknown",
                "score": 0.0,
                "second_score": -1.0,
                "margin": 0.0,
                "ranking": [],
            }

        query_embedding = self._normalize(query_embedding)

        ranking = []

        for emp_id, item in self.gallery.items():
            sims = [float(np.dot(query_embedding, emb)) for emb in item["embeddings"]]
            sims_sorted = sorted(sims, reverse=True)

            topk = sims_sorted[: min(self.topk_mean, len(sims_sorted))]
            topk_mean = float(np.mean(topk))
            max_score = float(sims_sorted[0])

            ranking.append(
                {
                    "employee_id": emp_id,
                    "name": item["name"],
                    "score": topk_mean,
                    "max_score": max_score,
                    "top_scores": topk,
                }
            )

        ranking.sort(key=lambda x: x["score"], reverse=True)

        best = ranking[0]
        second = ranking[1] if len(ranking) > 1 else None

        best_score = best["score"]
        second_score = second["score"] if second else -1.0
        margin = best_score - second_score if second else 999.0

        if len(ranking) == 1:
            is_match = best_score >= self.face_threshold
        else:
            is_match = best_score >= self.face_threshold and margin >= self.face_margin

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

    def identify_image(self, image_bgr):
        """
        Input: frame/crop BGR
        Output: dict nháº­n diá»‡n máº·t tá»‘t nháº¥t
        """
        faces = self.detect_faces(image_bgr)

        if len(faces) == 0:
            return {
                "status": "NO_FACE",
                "employee_id": "Unknown",
                "name": "Unknown",
                "score": 0.0,
                "second_score": -1.0,
                "margin": 0.0,
                "bbox": None,
                "faces": [],
            }

        best_face = faces[0]
        match = self.match_embedding(best_face["embedding"])

        match["bbox"] = best_face["bbox"]
        match["det_score"] = best_face["score"]
        match["faces"] = faces

        return match
