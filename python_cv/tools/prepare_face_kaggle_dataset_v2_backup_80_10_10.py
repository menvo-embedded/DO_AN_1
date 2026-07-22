import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path


PERSON_MAP = {
    "NV001_Mến": {
        "out_folder": "NV001_Men",
        "employee_id": "NV001",
        "display_name": "Men",
        "display_name_vi": "Mến",
        "label": 0,
    },
    "NV002_Đức": {
        "out_folder": "NV002_Duc",
        "employee_id": "NV002",
        "display_name": "Duc",
        "display_name_vi": "Đức",
        "label": 1,
    },
}

LAYER_MAP = {
    "L1_Dễ_chính_diện_rõ_mặt": {
        "out_folder": "L1_easy_front_clear",
        "layer_code": "L1",
        "layer_weight": 1.0,
    },
    "L2_Bình_thường_hơi_nghiêng": {
        "out_folder": "L2_normal_slight_angle",
        "layer_code": "L2",
        "layer_weight": 1.2,
    },
    "L3_Biến_thể_góc_mặt": {
        "out_folder": "L3_pose_variation",
        "layer_code": "L3",
        "layer_weight": 1.5,
    },
    "L4_Khó_hợp_lệ": {
        "out_folder": "L4_hard_valid",
        "layer_code": "L4",
        "layer_weight": 2.5,
    },
    "L5_Rất_khó_hợp_lệ": {
        "out_folder": "L5_extreme_hard_valid",
        "layer_code": "L5",
        "layer_weight": 3.5,
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_md5(path: Path, block_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def infer_is_synthetic(filename: str):
    upper = filename.upper()
    return int(
        upper.startswith("SYN_")
        or "SYN_L4" in upper
        or "SYN_L5" in upper
        or upper.startswith("FLIP_")
        or "FLIP_" in upper
    )


def infer_aug_type(filename: str, layer_code: str):
    upper = filename.upper()

    if "FLIP_" in upper:
        return "flip_pose_balance"

    if "SYN_L4" in upper:
        return "synthetic_hard_l4_blur_lowlight_noise_jpeg"

    if "SYN_L5" in upper:
        return "synthetic_extreme_l5_blur_lowlight_noise_jpeg"

    if layer_code in ["L1", "L2", "L3"]:
        return "real"

    return "unknown"


def infer_group_id(filename: str):
    """
    Chống leak data:
    Ảnh augment thường có tên SYN_L4_<source_stem>_aug0001.jpg
    hoặc FLIP_..._<source_stem>.jpg.
    Ta cố gắng gom source và augment vào cùng group.
    """
    stem = Path(filename).stem

    if stem.startswith("SYN_L4_"):
        stem = stem.replace("SYN_L4_", "", 1)
    if stem.startswith("SYN_L5_"):
        stem = stem.replace("SYN_L5_", "", 1)

    if "_aug" in stem:
        stem = stem.split("_aug")[0]

    if stem.startswith("FLIP_"):
        parts = stem.split("_")
        # bỏ prefix FLIP_xxx_TO_xxx nếu có
        # giữ phần cuối làm group tương đối
        if len(parts) > 6:
            stem = "_".join(parts[5:])

    return stem


def safe_filename(prefix: str, src: Path, idx: int):
    ext = ".jpg" if src.suffix.lower() not in IMAGE_EXTS else src.suffix.lower()
    clean_stem = (
        src.stem
        .replace(" ", "_")
        .replace("Đ", "D")
        .replace("đ", "d")
        .replace("ế", "e")
        .replace("ễ", "e")
        .replace("ệ", "e")
        .replace("ắ", "a")
        .replace("ă", "a")
        .replace("ư", "u")
        .replace("ơ", "o")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("á", "a")
        .replace("à", "a")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("ú", "u")
        .replace("ù", "u")
    )

    return f"{prefix}_{idx:06d}_{clean_stem}{ext}"


def collect_and_copy(src_root: Path, out_root: Path):
    rows = []
    idx = 0

    for person_vi, pinfo in PERSON_MAP.items():
        for layer_vi, linfo in LAYER_MAP.items():
            src_dir = src_root / person_vi / layer_vi
            dst_dir = out_root / pinfo["out_folder"] / linfo["out_folder"]
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                continue

            images = [
                p for p in src_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]

            for src in sorted(images):
                idx += 1
                prefix = f"{pinfo['employee_id']}_{linfo['layer_code']}"
                dst_name = safe_filename(prefix, src, idx)
                dst = dst_dir / dst_name

                shutil.copy2(src, dst)

                rel_path = dst.relative_to(out_root).as_posix()
                is_synthetic = infer_is_synthetic(src.name)
                aug_type = infer_aug_type(src.name, linfo["layer_code"])
                group_id = infer_group_id(src.name)

                rows.append({
                    "image_path": rel_path,
                    "person_folder": pinfo["out_folder"],
                    "employee_id": pinfo["employee_id"],
                    "display_name": pinfo["display_name"],
                    "display_name_vi": pinfo["display_name_vi"],
                    "label": pinfo["label"],
                    "layer": linfo["out_folder"],
                    "layer_code": linfo["layer_code"],
                    "layer_weight": linfo["layer_weight"],
                    "is_synthetic": is_synthetic,
                    "aug_type": aug_type,
                    "source_filename": src.name,
                    "source_group_id": group_id,
                    "md5": file_md5(dst),
                    "split": "",
                })

    return rows


def assign_splits_groupwise(rows, seed=42, train_ratio=0.8, val_ratio=0.1):
    """
    Chia train/val/test theo group để tránh leak:
    source image và ảnh augment từ nó vào cùng split.
    """
    random.seed(seed)

    group_map = {}
    for row in rows:
        key = (row["employee_id"], row["source_group_id"])
        group_map.setdefault(key, []).append(row)

    groups_by_emp = {}
    for key, group_rows in group_map.items():
        emp = key[0]
        groups_by_emp.setdefault(emp, []).append(group_rows)

    for emp, groups in groups_by_emp.items():
        random.shuffle(groups)

        n = len(groups)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, group_rows in enumerate(groups):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            for row in group_rows:
                row["split"] = split

    return rows


def write_manifest(rows, out_root: Path):
    manifest_path = out_root / "face_train_manifest_v2.csv"
    summary_path = out_root / "face_manifest_summary_v2.csv"
    label_map_path = out_root / "face_label_map_v2.json"
    config_path = out_root / "face_train_config_v2.json"

    fieldnames = [
        "image_path",
        "person_folder",
        "employee_id",
        "display_name",
        "display_name_vi",
        "label",
        "layer",
        "layer_code",
        "layer_weight",
        "is_synthetic",
        "aug_type",
        "source_filename",
        "source_group_id",
        "md5",
        "split",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (
            row["employee_id"],
            row["display_name_vi"],
            row["layer_code"],
            row["split"],
            row["is_synthetic"],
        )
        summary[key] = summary.get(key, 0) + 1

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "employee_id",
            "display_name_vi",
            "layer_code",
            "split",
            "is_synthetic",
            "count",
        ])

        for key, count in sorted(summary.items()):
            writer.writerow([*key, count])

    label_map = {}
    for _, pinfo in PERSON_MAP.items():
        label_map[str(pinfo["label"])] = {
            "employee_id": pinfo["employee_id"],
            "display_name": pinfo["display_name"],
            "display_name_vi": pinfo["display_name_vi"],
            "person_folder": pinfo["out_folder"],
        }

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    config = {
        "dataset_name": "face_reid_2people_v2",
        "active_people": {
            "NV001": "Mến",
            "NV002": "Đức",
        },
        "layer_weights": {
            "L1": 1.0,
            "L2": 1.2,
            "L3": 1.5,
            "L4": 2.5,
            "L5": 3.5,
        },
        "split_policy": "groupwise_by_source_group_id_to_avoid_augmentation_leakage",
        "recommended_training": {
            "backbone": "resnet50_or_mobilefacenet_or_iresnet",
            "loss": "cross_entropy_label_smoothing_or_arcface",
            "sampler": "weighted_sampler_by_layer_weight",
            "eval": [
                "accuracy_by_layer",
                "confusion_matrix",
                "threshold_score",
                "false_accept_false_reject",
            ],
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return manifest_path, summary_path, label_map_path, config_path


def zip_dataset(out_root: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_root.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_root))

    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default=r"D:\warehouse_dataset_v2\face_data_tieng_viet",
    )
    parser.add_argument(
        "--out-root",
        default=r"D:\warehouse_dataset_v2\face_train_final_en",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PREPARE FACE KAGGLE DATASET V2")
    print(f"SRC: {src_root}")
    print(f"OUT: {out_root}")
    print("=" * 90)

    rows = collect_and_copy(src_root, out_root)
    rows = assign_splits_groupwise(rows, seed=args.seed)

    manifest, summary, label_map, config = write_manifest(rows, out_root)

    print(f"Total images: {len(rows)}")
    print(f"Manifest: {manifest}")
    print(f"Summary: {summary}")
    print(f"Label map: {label_map}")
    print(f"Config: {config}")

    people_count = {}
    split_count = {}
    layer_count = {}

    for row in rows:
        people_count[row["employee_id"]] = people_count.get(row["employee_id"], 0) + 1
        split_count[row["split"]] = split_count.get(row["split"], 0) + 1
        layer_key = (row["employee_id"], row["layer_code"])
        layer_count[layer_key] = layer_count.get(layer_key, 0) + 1

    print("\nPeople count:")
    for k, v in sorted(people_count.items()):
        print(f"  {k}: {v}")

    print("\nSplit count:")
    for k, v in sorted(split_count.items()):
        print(f"  {k}: {v}")

    print("\nLayer count:")
    for k, v in sorted(layer_count.items()):
        print(f"  {k[0]} {k[1]}: {v}")

    if args.zip:
        zip_path = out_root.parent / "face_reid_2people_v2_kaggle.zip"
        zip_dataset(out_root, zip_path)
        print(f"\nZIP: {zip_path}")

    print("=" * 90)
    print("DONE")


if __name__ == "__main__":
    main()