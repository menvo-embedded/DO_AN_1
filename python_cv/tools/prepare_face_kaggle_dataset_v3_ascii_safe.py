import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PERSON_BY_PREFIX = {
    "NV001": {
        "out_folder": "NV001_Men",
        "employee_id": "NV001",
        "display_name": "Men",
        "display_name_vi": "Men",
        "label": 0,
    },
    "NV002": {
        "out_folder": "NV002_Duc",
        "employee_id": "NV002",
        "display_name": "Duc",
        "display_name_vi": "Duc",
        "label": 1,
    },
}

LAYER_BY_PREFIX = {
    "L1": {
        "out_folder": "L1_easy_front_clear",
        "layer_code": "L1",
        "layer_weight": 1.0,
    },
    "L2": {
        "out_folder": "L2_normal_slight_angle",
        "layer_code": "L2",
        "layer_weight": 1.2,
    },
    "L3": {
        "out_folder": "L3_pose_variation",
        "layer_code": "L3",
        "layer_weight": 1.5,
    },
    "L4": {
        "out_folder": "L4_hard_valid",
        "layer_code": "L4",
        "layer_weight": 2.5,
    },
    "L5": {
        "out_folder": "L5_extreme_hard_valid",
        "layer_code": "L5",
        "layer_weight": 3.5,
    },
}


def file_md5(path: Path, block_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_person_info(folder_name: str):
    for prefix, info in PERSON_BY_PREFIX.items():
        if folder_name.startswith(prefix):
            return prefix, info
    return None, None


def find_layer_info(folder_name: str):
    for prefix, info in LAYER_BY_PREFIX.items():
        if folder_name.startswith(prefix):
            return prefix, info
    return None, None


def infer_is_synthetic(filename: str):
    upper = filename.upper()
    return int(
        upper.startswith("SYN_")
        or "SYN_L4" in upper
        or "SYN_L5" in upper
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
    stem = Path(filename).stem

    if stem.startswith("SYN_L4_"):
        stem = stem.replace("SYN_L4_", "", 1)
    if stem.startswith("SYN_L5_"):
        stem = stem.replace("SYN_L5_", "", 1)

    if "_aug" in stem:
        stem = stem.split("_aug")[0]

    if stem.startswith("FLIP_"):
        parts = stem.split("_")
        if len(parts) > 6:
            stem = "_".join(parts[5:])

    return stem


def safe_filename(prefix: str, src: Path, idx: int):
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"

    clean = "".join(
        c if c.isalnum() or c in ["_", "-", "."] else "_"
        for c in src.stem
    )

    clean = clean.encode("ascii", errors="ignore").decode("ascii")
    if not clean:
        clean = "image"

    return f"{prefix}_{idx:06d}_{clean}{ext}"


def collect_and_copy(src_root: Path, out_root: Path):
    rows = []
    idx = 0

    person_dirs = [p for p in src_root.iterdir() if p.is_dir()]

    for person_dir in sorted(person_dirs):
        person_prefix, pinfo = find_person_info(person_dir.name)
        if pinfo is None:
            continue

        layer_dirs = [p for p in person_dir.iterdir() if p.is_dir()]

        for layer_dir in sorted(layer_dirs):
            layer_prefix, linfo = find_layer_info(layer_dir.name)
            if linfo is None:
                continue

            dst_dir = out_root / pinfo["out_folder"] / linfo["out_folder"]
            dst_dir.mkdir(parents=True, exist_ok=True)

            images = [
                p for p in layer_dir.rglob("*")
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


def assign_splits_groupwise(rows, seed=42, train_ratio=0.75, val_ratio=0.15):
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

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (
            row["employee_id"],
            row["layer_code"],
            row["split"],
            row["is_synthetic"],
        )
        summary[key] = summary.get(key, 0) + 1

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "employee_id",
            "layer_code",
            "split",
            "is_synthetic",
            "count",
        ])

        for key, count in sorted(summary.items()):
            writer.writerow([*key, count])

    label_map = {}
    for _, pinfo in PERSON_BY_PREFIX.items():
        label_map[str(pinfo["label"])] = {
            "employee_id": pinfo["employee_id"],
            "display_name": pinfo["display_name"],
            "person_folder": pinfo["out_folder"],
        }

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    config = {
        "dataset_name": "face_reid_2people_v2",
        "active_people": {
            "NV001": "Men",
            "NV002": "Duc",
        },
        "split_policy": "groupwise_by_source_group_id",
        "split_ratio": {
            "train": 0.75,
            "val": 0.15,
            "test": 0.10,
        },
        "layer_weights": {
            "L1": 1.0,
            "L2": 1.2,
            "L3": 1.5,
            "L4": 2.5,
            "L5": 3.5,
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
    parser.add_argument("--src-root", default=r"D:\warehouse_dataset_v2\face_data_tieng_viet")
    parser.add_argument("--out-root", default=r"D:\warehouse_dataset_v2\face_train_final_en")
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
    print("PREPARE FACE KAGGLE DATASET V3 ASCII SAFE")
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