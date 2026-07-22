"""
tools/test_with_clean_gallery_experiment.py
--------------------------------------------
Test sanity check với gallery thử nghiệm sạch.
KHÔNG sửa gallery chính.

Usage:
    cd D:\\warehouse-access-rfid-cv\\python_cv
    D:\\UV4\\anaconda3\\python.exe tools\\test_with_clean_gallery_experiment.py
"""

import sys
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid.reid_engine import ReIDEngine
from config.settings import REID_MATCH_THRESHOLD, REID_MATCH_MARGIN

# ── Paths ─────────────────────────────────────────────────────────────────────
CANDIDATES   = Path("D:/warehouse_dataset/gallery_candidates")
EXP_GALLERY  = CANDIDATES / "gallery_clean_experiment.pkl"
GT_DIR       = Path("D:/warehouse_dataset/synthetic_tests/ground_truth")
OUT_MD       = CANDIDATES / "clean_gallery_sanity_result.md"
OUT_CSV      = CANDIDATES / "clean_gallery_sanity_result.csv"

ALL_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]
NAMES   = {"NV001":"Bo","NV002":"Me","NV003":"Anh","NV004":"Chi","NV005":"Toi"}

# Baseline từ report hiện tại
BASELINE = {
    "accuracy":  0.5789,
    "NV001": 1.0000,
    "NV002": 1.0000,
    "NV003": 0.0000,
    "NV004": 1.0000,
    "NV005": 0.0000,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_experiment_gallery() -> dict:
    if not EXP_GALLERY.exists():
        raise FileNotFoundError(
            f"Gallery thử nghiệm không tồn tại: {EXP_GALLERY}\n"
            "Chạy build_clean_gallery_experiment.py trước."
        )
    with open(EXP_GALLERY, "rb") as f:
        raw = pickle.load(f)
    return {k: [np.array(e, dtype=np.float32) for e in v]
            for k, v in raw.items()}


def rank_reid(reid: ReIDEngine, gallery: dict, crop_bgr: np.ndarray) -> dict:
    result = {
        "pred_emp_id": None,
        "best_emp_id": None,
        "best_score":  0.0,
        "second_emp_id": None,
        "second_score":  0.0,
        "margin":      0.0,
        "top5_scores": [],
        "reject_reason": "",
    }

    if crop_bgr is None or crop_bgr.size == 0:
        result["reject_reason"] = "invalid_crop"
        return result

    emb = reid.get_embedding(crop_bgr)
    if emb is None:
        result["reject_reason"] = "no_embedding"
        return result

    rows = []
    for emp_id, embeds in gallery.items():
        if not embeds:
            continue
        score = reid.match_score(emb, embeds)
        rows.append((emp_id, score))

    if not rows:
        result["reject_reason"] = "no_gallery"
        return result

    rows.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score   = rows[0]
    second_id, second_score = rows[1] if len(rows) > 1 else (None, 0.0)
    margin = float(best_score - second_score)

    result.update({
        "best_emp_id":   best_id,
        "best_score":    float(best_score),
        "second_emp_id": second_id,
        "second_score":  float(second_score),
        "margin":        margin,
        "top5_scores":   [(eid, float(sc)) for eid, sc in rows[:5]],
    })

    if best_score >= REID_MATCH_THRESHOLD and margin >= REID_MATCH_MARGIN:
        result["pred_emp_id"] = best_id
    elif best_score < REID_MATCH_THRESHOLD:
        result["reject_reason"] = "low_score"
    else:
        result["reject_reason"] = "low_margin"

    return result


def load_unique_source_images(gt_dir: Path) -> list[dict]:
    seen, out = set(), []
    for gt_path in sorted(gt_dir.glob("*.json")):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                emp_id = obj.get("emp_id", "")
                src    = obj.get("source_image", "")
                key    = (emp_id, src)
                if not emp_id or not src or key in seen:
                    continue
                seen.add(key)
                out.append({"emp_id": emp_id, "source_image": src,
                            "case_name": data.get("case_name", gt_path.stem)})
    return out


def fmt_top5(scores):
    if not scores:
        return ""
    return ";".join(f"{e}:{s:.6f}" for e, s in scores)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TEST WITH CLEAN GALLERY EXPERIMENT")
    print("=" * 60)

    reid    = ReIDEngine()
    gallery = load_experiment_gallery()

    print(f"\n[Gallery thử nghiệm] {EXP_GALLERY}")
    for nv in ALL_IDS:
        n   = len(gallery.get(nv, []))
        src = "clean" if nv in ("NV003","NV005") else "main"
        print(f"  {nv}: {n} embeddings [{src}]")

    if not GT_DIR.exists():
        print(f"[ERROR] Ground truth dir không tồn tại: {GT_DIR}")
        return

    items = load_unique_source_images(GT_DIR)
    print(f"\n[Source images] {len(items)} unique images từ ground truth")

    rows = []
    for item in items:
        img_path = Path(item["source_image"])
        img      = cv2.imread(str(img_path))

        if img is None:
            rows.append({
                "case_name": item["case_name"],
                "emp_id":    item["emp_id"],
                "source_image": str(img_path),
                "status":    "unreadable",
                "pred_emp_id": "Unknown",
                "correct":   False,
                "best_emp_id":   "",
                "best_score":    "0.000000",
                "second_emp_id": "",
                "second_score":  "0.000000",
                "margin":        "0.000000",
                "top5_scores":   "",
                "reject_reason": "unreadable",
                "threshold_used": f"{REID_MATCH_THRESHOLD:.6f}",
                "margin_used":    f"{REID_MATCH_MARGIN:.6f}",
            })
            continue

        result = rank_reid(reid, gallery, img)
        pred   = result["pred_emp_id"] or "Unknown"

        rows.append({
            "case_name":    item["case_name"],
            "emp_id":       item["emp_id"],
            "source_image": str(img_path),
            "status":       "ok",
            "pred_emp_id":  pred,
            "correct":      (pred == item["emp_id"]),
            "best_emp_id":   result["best_emp_id"] or "",
            "best_score":    f"{result['best_score']:.6f}",
            "second_emp_id": result["second_emp_id"] or "",
            "second_score":  f"{result['second_score']:.6f}",
            "margin":        f"{result['margin']:.6f}",
            "top5_scores":   fmt_top5(result["top5_scores"]),
            "reject_reason": result["reject_reason"],
            "threshold_used": f"{REID_MATCH_THRESHOLD:.6f}",
            "margin_used":    f"{REID_MATCH_MARGIN:.6f}",
        })

    # ── CSV ──────────────────────────────────────────────────────────────────
    fields = ["case_name","emp_id","source_image","status","pred_emp_id",
              "correct","best_emp_id","best_score","second_emp_id",
              "second_score","margin","top5_scores","reject_reason",
              "threshold_used","margin_used"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[CSV] {OUT_CSV}")

    # ── Analytics ─────────────────────────────────────────────────────────────
    valid   = [r for r in rows if r["status"] == "ok"]
    correct = [r for r in valid if r["correct"]]
    unknown = [r for r in valid if r["pred_emp_id"] == "Unknown"]
    acc     = len(correct) / len(valid) if valid else 0.0

    by_emp = defaultdict(list)
    for r in valid:
        by_emp[r["emp_id"]].append(r)

    confusions = Counter()
    for r in valid:
        if not r["correct"]:
            confusions[(r["emp_id"], r["pred_emp_id"])] += 1

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# Clean Gallery Experiment — Sanity Result",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Gallery:** `{EXP_GALLERY}`",
        f"**Threshold:** {REID_MATCH_THRESHOLD} | **Margin:** {REID_MATCH_MARGIN}",
        "",
        "## Summary",
        "",
        f"| Metric | Baseline | Experiment | Delta |",
        f"|--------|----------|------------|-------|",
    ]

    delta_acc = acc - BASELINE["accuracy"]
    delta_str = f"+{delta_acc:.4f}" if delta_acc >= 0 else f"{delta_acc:.4f}"
    lines.append(
        f"| accuracy | {BASELINE['accuracy']:.4f} | {acc:.4f} | {delta_str} |"
    )

    lines += [
        "",
        f"- source_images_total: {len(rows)}",
        f"- readable_images: {len(valid)}",
        f"- correct: {len(correct)}",
        f"- accuracy: {acc:.4f}",
        f"- unknown_count: {len(unknown)}",
        "",
        "## Per Employee",
        "",
        "| emp_id | total | correct | accuracy | unknown | baseline_acc | delta |",
        "|--------|------:|--------:|--------:|--------:|-------------:|-------|",
    ]

    for nv in ALL_IDS:
        emp_rows    = by_emp.get(nv, [])
        emp_correct = [r for r in emp_rows if r["correct"]]
        emp_unknown = [r for r in emp_rows if r["pred_emp_id"] == "Unknown"]
        emp_acc     = len(emp_correct) / len(emp_rows) if emp_rows else 0.0
        base_acc    = BASELINE.get(nv, 0.0)
        delta       = emp_acc - base_acc
        delta_s     = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        src_tag     = " [clean]" if nv in ("NV003","NV005") else ""
        lines.append(
            f"| {nv}{src_tag} | {len(emp_rows)} | {len(emp_correct)} | "
            f"{emp_acc:.4f} | {len(emp_unknown)} | {base_acc:.4f} | {delta_s} |"
        )

    lines += ["", "## Common Confusions", ""]
    if confusions:
        for (gt, pred), cnt in confusions.most_common(10):
            lines.append(f"- {gt} → {pred}: {cnt}")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Interpretation",
        "",
    ]

    if acc > BASELINE["accuracy"]:
        lines.append(
            f"✅ Clean gallery cải thiện accuracy: "
            f"{BASELINE['accuracy']:.4f} → {acc:.4f} (+{acc-BASELINE['accuracy']:.4f})"
        )
    else:
        lines.append(
            f"❌ Clean gallery KHÔNG cải thiện: "
            f"{BASELINE['accuracy']:.4f} → {acc:.4f}"
        )

    nv3 = by_emp.get("NV003", [])
    nv5 = by_emp.get("NV005", [])
    nv3_acc = len([r for r in nv3 if r["correct"]]) / len(nv3) if nv3 else 0.0
    nv5_acc = len([r for r in nv5 if r["correct"]]) / len(nv5) if nv5 else 0.0

    if nv3_acc > BASELINE["NV003"]:
        lines.append(f"✅ NV003: {BASELINE['NV003']:.4f} → {nv3_acc:.4f} — cải thiện")
    else:
        lines.append(f"❌ NV003: {BASELINE['NV003']:.4f} → {nv3_acc:.4f} — không cải thiện")

    if nv5_acc > BASELINE["NV005"]:
        lines.append(f"✅ NV005: {BASELINE['NV005']:.4f} → {nv5_acc:.4f} — cải thiện")
    else:
        lines.append(f"❌ NV005: {BASELINE['NV005']:.4f} → {nv5_acc:.4f} — không cải thiện")

    lines += [
        "",
        "## Next Step",
        "",
        "- Nếu accuracy tăng đáng kể → xem xét rebuild gallery chính với ảnh clean tương tự.",
        "- Nếu không tăng → vấn đề ở threshold hoặc dataset source images quá khó.",
        "- Nếu NV003/NV005 vẫn fail → hạ threshold 0.90 và test lại.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[MD]  {OUT_MD}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"RESULT: accuracy={acc:.4f} (baseline={BASELINE['accuracy']:.4f})")
    print(f"  NV003: {nv3_acc:.4f} (baseline={BASELINE['NV003']:.4f})")
    print(f"  NV005: {nv5_acc:.4f} (baseline={BASELINE['NV005']:.4f})")
    print(f"  unknown_count={len(unknown)}/{len(valid)}")
    print(f"{'='*60}")
    print("[DONE]")


if __name__ == "__main__":
    main()
