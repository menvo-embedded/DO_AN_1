import numpy as np
import torch
import torch.nn as nn
import cv2
import torchreid
from pathlib import Path
from torchvision import models as tv_models

from config.settings import (
    ROOT,
    REID_MODEL_NAME,
    REID_INPUT_SIZE,
    REID_MATCH_THRESHOLD,
    REID_MATCH_MARGIN,
    REID_TOPK_MEAN,
    REID_FINETUNED_PATH,
)
from utils.logger import get_logger

log = get_logger("reid")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)



class ReIDEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading ReIDEngine on {self.device}")

        # Fallback OSNet
        self.model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1000,
            pretrained=True,
        )
        self.model.eval().to(self.device)

        # Fine-tuned ResNet50 v3
        self.finetuned_model = None
        self._load_finetuned()

        if self.finetuned_model is not None:
            log.info("ReIDEngine ready: Body Re-ID only | Fine-tuned ResNet50 v3 | 2048D")
        else:
            log.warning("ReIDEngine ready: fallback OSNet")

    def _load_finetuned(self):
        path = Path(REID_FINETUNED_PATH)
        if not path.is_absolute():
            path = ROOT / path

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
            log.info("Fine-tuned ResNet50 v3 loaded OK")

        except Exception as e:
            log.warning(f"Fine-tuned load failed: {e}")
            self.finetuned_model = None

    @torch.no_grad()
    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Body Re-ID embedding thuần.
        Không trộn face embedding vào đây để tránh lỗi lẫn 512D/2048D.
        Face Recognition nên xử lý bằng engine riêng và fusion score riêng.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        try:
            if self.finetuned_model is not None:
                img = cv2.resize(crop_bgr, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = (img - _MEAN) / _STD

                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

                feat = self.finetuned_model(tensor)
                feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

                emb = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
                return emb / (np.linalg.norm(emb) + 1e-8)

            # fallback OSNet
            w, h = REID_INPUT_SIZE
            img = cv2.resize(crop_bgr, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - _MEAN) / _STD

            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            feat = self.model(tensor)
            feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

            emb = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)

        except Exception as e:
            log.error(f"Embedding error: {e}")
            return None

    def _prepare_gallery_matrix(self, query: np.ndarray, gallery_embeds: list) -> np.ndarray | None:
        """
        Lọc gallery embedding cùng dimension với query.
        Tránh lỗi do gallery cũ từng có embedding 512D lẫn 2048D.
        """
        if query is None or not gallery_embeds:
            return None

        q_dim = int(query.shape[0])
        valid = []

        for emb in gallery_embeds:
            if emb is None:
                continue

            arr = np.asarray(emb, dtype=np.float32).reshape(-1)

            if arr.shape[0] != q_dim:
                continue

            arr = arr / (np.linalg.norm(arr) + 1e-8)
            valid.append(arr)

        if not valid:
            return None

        return np.stack(valid)

    def match_score(self, query: np.ndarray, gallery_embeds: list) -> float:
        """
        Top-k mean cosine similarity.
        Ổn định hơn max-score vì giảm rủi ro 1 ảnh gallery bị ăn may.
        """
        if query is None or not gallery_embeds:
            return 0.0

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)

        mat = self._prepare_gallery_matrix(q, gallery_embeds)

        if mat is None:
            return 0.0

        scores = mat @ q

        if scores.size == 0:
            return 0.0

        k = min(REID_TOPK_MEAN, scores.size)
        topk = np.sort(scores)[-k:]

        return float(np.mean(topk))

    def identify(self, crop_bgr: np.ndarray, gallery: dict) -> str | None:
        """
        Identify bằng:
        - best_score >= REID_MATCH_THRESHOLD
        - best_score - second_score >= REID_MATCH_MARGIN
        """
        emb = self.get_embedding(crop_bgr)

        if emb is None:
            return None

        results = []

        for emp_id, embeds in gallery.items():
            if not embeds:
                continue

            score = self.match_score(emb, embeds)
            results.append((emp_id, score))

        if not results:
            return None

        results.sort(key=lambda x: x[1], reverse=True)

        best_id, best_score = results[0]
        second_score = results[1][1] if len(results) > 1 else 0.0
        margin = best_score - second_score

        log.info(
            f"REID identify: best={best_id} score={best_score:.3f} "
            f"second={second_score:.3f} margin={margin:.3f}"
        )

        if best_score >= REID_MATCH_THRESHOLD and margin >= REID_MATCH_MARGIN:
            return best_id

        return None
