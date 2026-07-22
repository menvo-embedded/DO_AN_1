import sys
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reid.reid_engine import ReIDEngine
from config.settings import GALLERY_DIR


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


if len(sys.argv) < 2:
    print("Usage: python scripts/diagnose_reid_scores.py <image_path>")
    raise SystemExit

img_path = Path(sys.argv[1])
if not img_path.exists():
    raise FileNotFoundError(img_path)

gallery_path = GALLERY_DIR / "gallery.pkl"
print("Image:", img_path)
print("Gallery:", gallery_path)

with open(gallery_path, "rb") as f:
    gallery = pickle.load(f)

print("Gallery IDs:", list(gallery.keys()))
for k, v in gallery.items():
    print(k, len(v))

img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError(f"Cannot read image: {img_path}")

reid = ReIDEngine()
query_emb = reid.get_embedding(img)

if query_emb is None:
    print("Cannot extract embedding.")
    raise SystemExit

rows = []

for emp_id, embs in gallery.items():
    scores = [cosine(query_emb, e) for e in embs if e is not None]

    if not scores:
        continue

    scores_sorted = sorted(scores, reverse=True)
    rows.append({
        "id": emp_id,
        "max": scores_sorted[0],
        "top5_mean": float(np.mean(scores_sorted[:5])),
        "mean": float(np.mean(scores_sorted)),
    })

rows = sorted(rows, key=lambda x: x["top5_mean"], reverse=True)

print("\n=== SCORE RANKING ===")
for r in rows:
    print(
        f"{r['id']} | "
        f"max={r['max']:.4f} | "
        f"top5_mean={r['top5_mean']:.4f} | "
        f"mean={r['mean']:.4f}"
    )