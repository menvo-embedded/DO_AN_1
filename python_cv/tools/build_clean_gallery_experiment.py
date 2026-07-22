"""
tools/build_clean_gallery_experiment.py
----------------------------------------
Tạo gallery thử nghiệm sạch cho NV003/NV005.
KHÔNG ghi đè gallery chính.

Usage:
    cd D:\\warehouse-access-rfid-cv\\python_cv
    D:\\UV4\\anaconda3\\python.exe tools\\build_clean_gallery_experiment.py
"""

import sys
import pickle
from collections import deque
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid.reid_engine import ReIDEngine
from config.settings import GALLERY_DIR, REID_GALLERY_SIZE

# ── Paths ─────────────────────────────────────────────────────────────────────
GALLERY_MAIN  = GALLERY_DIR / "gallery.pkl"
CANDIDATES    = Path("D:/warehouse_dataset/gallery_candidates")
NV003_DIR     = CANDIDATES / "NV003_clean"
NV005_DIR     = CANDIDATES / "NV005_clean"
OUTPUT_PKL    = CANDIDATES / "gallery_clean_experiment.pkl"
REPORT_MD     = CANDIDATES / "clean_gallery_experiment_report.md"

CANDIDATES.mkdir(parents=True, exist_ok=True)

MIN_H       = 160
MIN_W       = 60
MIN_BLUR    = 30.0
IMG_EXTS    = {".jpg", ".jpeg", ".png"}
ALL_IDS     = ["NV001", "NV002", "NV003", "NV004", "NV005"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_main_gallery() -> dict:
    if not GALLERY_MAIN.exists():
        raise FileNotFoundError(f"Gallery chính không tồn tại: {GALLERY_MAIN}")
    with open(GALLERY_MAIN, "rb") as f:
        raw = pickle.load(f)
    gallery = {}
    for k, v in raw.items():
        gallery[k] = list(v)
    return gallery


def build_embeddings_from_dir(
    folder: Path,
    reid: ReIDEngine,
    nv_id: str,
) -> tuple[list, list]:
    """
    Đọc ảnh từ folder, extract embedding qua ReIDEngine.
    Trả về (embeddings, skip_log).
    """
    if not folder.exists():
        print(f"[WARN] Folder không tồn tại: {folder}")
        return [], [f"Folder {folder} không tồn tại"]

    imgs = sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in IMG_EXTS and p.is_file()])

    embeddings = []
    skip_log   = []
    blurs      = []

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            skip_log.append(f"{p.name}: không đọc được")
            continue

        h, w = img.shape[:2]
        if h < MIN_H or w < MIN_W:
            skip_log.append(f"{p.name}: quá nhỏ ({w}x{h})")
            continue

        bl = blur_score(img)
        blurs.append(bl)

        if bl < MIN_BLUR:
            skip_log.append(f"{p.name}: quá mờ (blur={bl:.1f})")
            continue

        emb = reid.get_embedding(img)
        if emb is None:
            skip_log.append(f"{p.name}: get_embedding trả None")
            continue

        emb = np.array(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append(emb)

    return embeddings, skip_log, blurs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BUILD CLEAN GALLERY EXPERIMENT")
    print("=" * 60)

    reid = ReIDEngine()

    # Load gallery chính
    main_gallery = load_main_gallery()
    print(f"\n[Gallery chính] Loaded from: {GALLERY_MAIN}")
    for nv in ALL_IDS:
        n = len(main_gallery.get(nv, []))
        print(f"  {nv}: {n} embeddings")

    # Build embeddings NV003 clean
    print(f"\n[NV003] Đọc từ: {NV003_DIR}")
    nv3_embs, nv3_skip, nv3_blurs = build_embeddings_from_dir(NV003_DIR, reid, "NV003")
    print(f"  → {len(nv3_embs)} embeddings, {len(nv3_skip)} skipped")

    # Build embeddings NV005 clean
    print(f"\n[NV005] Đọc từ: {NV005_DIR}")
    nv5_embs, nv5_skip, nv5_blurs = build_embeddings_from_dir(NV005_DIR, reid, "NV005")
    print(f"  → {len(nv5_embs)} embeddings, {len(nv5_skip)} skipped")

    # Build experiment gallery
    exp_gallery = {}
    for nv in ALL_IDS:
        if nv == "NV003":
            exp_gallery[nv] = nv3_embs
        elif nv == "NV005":
            exp_gallery[nv] = nv5_embs
        else:
            exp_gallery[nv] = list(main_gallery.get(nv, []))

    # Save experiment gallery
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(exp_gallery, f)
    print(f"\n[SAVED] Gallery thử nghiệm: {OUTPUT_PKL}")

    # Verify
    print("\n[Verify] Experiment gallery:")
    for nv in ALL_IDS:
        n = len(exp_gallery.get(nv, []))
        src = "clean" if nv in ("NV003","NV005") else "main"
        print(f"  {nv}: {n} embeddings [{src}]")

    # ── Markdown report ───────────────────────────────────────────────────────
    def fmt_blurs(blurs):
        if not blurs:
            return "N/A"
        return f"min={min(blurs):.1f} mean={np.mean(blurs):.1f} max={max(blurs):.1f}"

    lines = [
        "# Clean Gallery Experiment Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Config",
        "",
        f"- Gallery chính: `{GALLERY_MAIN}`",
        f"- Output thử nghiệm: `{OUTPUT_PKL}`",
        f"- MIN_H={MIN_H}, MIN_W={MIN_W}, MIN_BLUR={MIN_BLUR}",
        "",
        "## Gallery chính (giữ nguyên)",
        "",
    ]
    for nv in ["NV001","NV002","NV004"]:
        lines.append(f"- {nv}: {len(main_gallery.get(nv,[]))} embeddings")

    lines += [
        "",
        "## NV003 Clean Build",
        "",
        f"- Folder: `{NV003_DIR}`",
        f"- Ảnh đọc được / embedding tạo được: {len(nv3_embs)}",
        f"- Ảnh bị skip: {len(nv3_skip)}",
        f"- Blur stats: {fmt_blurs(nv3_blurs)}",
    ]
    if len(nv3_embs) < 20:
        lines.append(f"- ⚠️ WARNING: Chỉ có {len(nv3_embs)} embeddings, nên >= 20")
    if nv3_skip:
        lines.append("\n**Skip log:**")
        for s in nv3_skip[:10]:
            lines.append(f"  - {s}")
        if len(nv3_skip) > 10:
            lines.append(f"  - ... và {len(nv3_skip)-10} ảnh khác")

    lines += [
        "",
        "## NV005 Clean Build",
        "",
        f"- Folder: `{NV005_DIR}`",
        f"- Ảnh đọc được / embedding tạo được: {len(nv5_embs)}",
        f"- Ảnh bị skip: {len(nv5_skip)}",
        f"- Blur stats: {fmt_blurs(nv5_blurs)}",
    ]
    if len(nv5_embs) < 20:
        lines.append(f"- ⚠️ WARNING: Chỉ có {len(nv5_embs)} embeddings, nên >= 20")
    if nv5_skip:
        lines.append("\n**Skip log:**")
        for s in nv5_skip[:10]:
            lines.append(f"  - {s}")
        if len(nv5_skip) > 10:
            lines.append(f"  - ... và {len(nv5_skip)-10} ảnh khác")

    lines += [
        "",
        "## Output",
        "",
        f"- `{OUTPUT_PKL}`",
        "",
        "## Next Step",
        "",
        "```",
        "D:\\UV4\\anaconda3\\python.exe tools\\test_with_clean_gallery_experiment.py",
        "```",
    ]

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {REPORT_MD}")
    print("\n[DONE] Chạy tiếp: python tools/test_with_clean_gallery_experiment.py")


if __name__ == "__main__":
    main()
