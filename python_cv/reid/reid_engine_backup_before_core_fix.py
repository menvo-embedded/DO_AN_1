import numpy as np
import torch
import torch.nn as nn
import cv2
import torchreid
import onnxruntime as ort
from pathlib import Path
from torchvision import models as tv_models

from config.settings import REID_MODEL_NAME, REID_INPUT_SIZE, REID_MATCH_THRESHOLD
from utils.logger import get_logger

log = get_logger("reid")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FACE_DET_MODEL = "models/insightface/buffalo_sc/det_500m.onnx"
FACE_REC_MODEL = "models/insightface/buffalo_sc/w600k_mbf.onnx"
FINETUNED_PATH = "models/reid_resnet50_v3_engine.pth"
NUM_CLASSES    = 5

# Body:Face ratio trong hybrid embedding
BODY_W = 0.6
FACE_W = 0.4


class ReIDEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading {REID_MODEL_NAME} on {self.device}")

        # OSNet (dung de fallback)
        self.model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1000,
            pretrained=True,
        )
        self.model.eval().to(self.device)

        # Fine-tuned ResNet50 v3
        self.finetuned_model = None
        self._load_finetuned()

        # AdaFace
        self.face_det = None
        self.face_rec = None
        self._load_adaface()

        # OpenCV face detector (de crop mat tu toan than)
        self.cv_face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        log.info("ReIDEngine ready (ResNet50-FT + AdaFace + HaarCascade)")

    def _load_finetuned(self):
        path = Path(FINETUNED_PATH)
        if not path.exists():
            log.warning(f"Fine-tuned model not found: {path}")
            return
        try:
            base = tv_models.resnet50(weights=None)
            base.fc = nn.Identity()
            state = torch.load(str(path), map_location=self.device)
            new_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ["module.", "model.", "backbone."]:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                new_state[nk] = v
            missing, _ = base.load_state_dict(new_state, strict=False)
            real_missing = [k for k in missing if not k.startswith("fc.")]
            if real_missing:
                log.warning(f"Missing keys: {real_missing[:5]}")
            base.eval().to(self.device)
            self.finetuned_model = base
            log.info("Fine-tuned ResNet50 v3 loaded OK (2048d embedding)")
        except Exception as e:
            log.warning(f"Fine-tuned load failed: {e}")

    def _load_adaface(self):
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            det_path = Path(FACE_DET_MODEL)
            rec_path = Path(FACE_REC_MODEL)
            if det_path.exists() and rec_path.exists():
                self.face_det = ort.InferenceSession(
                    str(det_path), sess_options=opts, providers=providers)
                self.face_rec = ort.InferenceSession(
                    str(rec_path), sess_options=opts, providers=providers)
                log.info("AdaFace loaded on GPU")
            else:
                log.warning("AdaFace models not found")
        except Exception as e:
            log.warning(f"AdaFace load failed: {e}")

    def _crop_face_from_body(self, body_bgr: np.ndarray) -> np.ndarray | None:
        """
        Crop khuon mat tu anh toan than.
        Thu HaarCascade truoc (nhanh), neu ko thay thi crop 1/3 tren cua body.
        """
        if body_bgr is None or body_bgr.size == 0:
            return None

        h, w = body_bgr.shape[:2]

        # 1. Thu HaarCascade
        gray  = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cv_face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        if len(faces) > 0:
            # Lay mat lon nhat
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            pad = int(0.2 * min(fw, fh))
            x1 = max(0, fx - pad)
            y1 = max(0, fy - pad)
            x2 = min(w, fx + fw + pad)
            y2 = min(h, fy + fh + pad)
            face_crop = body_bgr[y1:y2, x1:x2]
            if face_crop.size > 0:
                return face_crop

        # 2. Fallback: crop 1/3 tren (vung dau/mat)
        top_h = max(30, h // 3)
        face_crop = body_bgr[:top_h, :]
        if face_crop.size > 0:
            return face_crop

        return None

    @torch.no_grad()
    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            # ── Body embedding (ResNet50 fine-tuned) ─────────────────────────
            body_emb = None
            if self.finetuned_model is not None:
                img2 = cv2.resize(crop_bgr, (224, 224))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img2 = (img2 - _MEAN) / _STD
                t2   = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)
                ft_feat = self.finetuned_model(t2)
                ft_feat = ft_feat / (ft_feat.norm(dim=1, keepdim=True) + 1e-8)
                body_emb = ft_feat.squeeze(0).cpu().numpy()
            else:
                # Fallback OSNet
                w, h = REID_INPUT_SIZE
                img = cv2.resize(crop_bgr, (w, h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = (img - _MEAN) / _STD
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                feat = self.model(tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
                body_emb = feat.squeeze(0).cpu().numpy()

            # ── Face embedding (AdaFace) ──────────────────────────────────────
            if self.face_det is not None:
                face_crop = self._crop_face_from_body(crop_bgr)
                if face_crop is not None:
                    face_emb = self._get_face_embedding(face_crop)
                    if face_emb is not None:
                        log.info(f"Face detected — hybrid embedding")
                        min_dim  = min(len(body_emb), len(face_emb))
                        combined = (body_emb[:min_dim] * BODY_W
                                    + face_emb[:min_dim] * FACE_W)
                        return combined / (np.linalg.norm(combined) + 1e-8)

            return body_emb / (np.linalg.norm(body_emb) + 1e-8)

        except Exception as e:
            log.error(f"Embedding error: {e}")
            return None

    def _get_face_embedding(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """AdaFace inference tren anh mat da crop san."""
        try:
            # Detection
            det_img = cv2.resize(face_bgr, (320, 320))
            det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            det_img = (det_img - 127.5) / 128.0
            det_img = det_img.transpose(2, 0, 1)[np.newaxis]
            det_input  = self.face_det.get_inputs()[0].name
            det_output = self.face_det.run(None, {det_input: det_img})
            scores = det_output[0][0] if len(det_output) > 0 else []
            if len(scores) == 0 or float(scores[0]) < 0.1:
                return None
            log.info(f"AdaFace score={float(scores[0]):.3f}")

            # Recognition
            rec_img = cv2.resize(face_bgr, (112, 112))
            rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            rec_img = (rec_img - 127.5) / 128.0
            rec_img = rec_img.transpose(2, 0, 1)[np.newaxis]
            rec_input = self.face_rec.get_inputs()[0].name
            face_feat = self.face_rec.run(None, {rec_input: rec_img})[0][0]
            return face_feat / (np.linalg.norm(face_feat) + 1e-8)
        except Exception:
            return None

    def match_score(self, query: np.ndarray, gallery_embeds: list) -> float:
        if not gallery_embeds or query is None:
            return 0.0
        mat     = np.stack(gallery_embeds)
        min_dim = min(mat.shape[1], len(query))
        scores  = mat[:, :min_dim] @ query[:min_dim]
        return float(scores.max())

    def identify(self, crop_bgr: np.ndarray, gallery: dict) -> str | None:
        emb = self.get_embedding(crop_bgr)
        if emb is None:
            return None
        best_id, best_score = None, REID_MATCH_THRESHOLD
        for emp_id, embeds in gallery.items():
            if not embeds:
                continue
            score = self.match_score(emb, embeds)
            if score > best_score:
                best_score, best_id = score, emp_id
        return best_id
