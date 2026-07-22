import pickle
from collections import deque
from pathlib import Path
import numpy as np
from config.settings import GALLERY_DIR, REID_GALLERY_SIZE
from utils.logger import get_logger

log = get_logger("gallery")
GALLERY_FILE = GALLERY_DIR / "gallery.pkl"

class Gallery:
    def __init__(self):
        self._data: dict[str, deque] = {}
        self._load()

    def update(self, employee_id: str, embedding: np.ndarray):
        if employee_id not in self._data:
            self._data[employee_id] = deque(maxlen=REID_GALLERY_SIZE)
        self._data[employee_id].append(embedding)
        self._save()

    def get(self, employee_id: str) -> list:
        return list(self._data.get(employee_id, []))

    def all(self) -> dict:
        return {k: list(v) for k, v in self._data.items()}

    def employees(self) -> list:
        return list(self._data.keys())

    def size(self, employee_id: str) -> int:
        return len(self._data.get(employee_id, []))

    def _save(self):
        try:
            with open(GALLERY_FILE, "wb") as f:
                pickle.dump({k: list(v) for k, v in self._data.items()}, f)
        except Exception as e:
            log.error(f"Gallery save error: {e}")

    def _load(self):
        if GALLERY_FILE.exists():
            try:
                with open(GALLERY_FILE, "rb") as f:
                    raw = pickle.load(f)
                self._data = {k: deque(v, maxlen=REID_GALLERY_SIZE) for k, v in raw.items()}
                log.info(f"Gallery loaded: {list(self._data.keys())}")
            except Exception as e:
                log.warning(f"Gallery load failed ({e}) - starting fresh")
                self._data = {}
        else:
            log.info("No gallery file - starting fresh")
