"""
Plotwise — PlantVillage Dataset Preparation
Author: Limawapang L Jamir

Downloads PlantVillage from Kaggle and organises into the folder structure
that train_disease_model.py expects:

    data/
    ├── train/
    │   ├── Rice_Blast/
    │   ├── Rice_BacterialBlight/
    │   ├── Rice_BrownSpot/
    │   ├── Maize_GrayLeafSpot/
    │   ├── Maize_NorthernLeafBlight/
    │   ├── Maize_CommonRust/
    │   ├── Potato_EarlyBlight/
    │   ├── Potato_LateBlight/
    │   ├── Chilli_LeafCurl/
    │   └── Healthy/
    └── test/
        └── ... (same structure, 20% split)

Prerequisites:
    pip install kaggle
    Place your kaggle.json API token at ~/.kaggle/kaggle.json
    (Download from: https://www.kaggle.com/settings → API → Create New Token)

Usage:
    python ml/prepare_dataset.py
    python ml/prepare_dataset.py --limit 500   # cap per class (faster for testing)
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# ── Mapping: PlantVillage folder name → our class label ──────────────────────
# Dataset: vipoooool/new-plant-diseases-dataset (Kaggle)
# The dataset already has a train/valid split — we'll use train only and re-split.

CLASS_MAP = {
    # PlantVillage folder name                          → our label
    "Rice___Leaf_blast":                                 "Rice_Blast",
    "Rice___Bacterial_leaf_blight":                      "Rice_BacterialBlight",
    "Rice___Brown_spot":                                 "Rice_BrownSpot",
    "Corn_(maize)___Gray_leaf_spot":                     "Maize_GrayLeafSpot",
    "Corn_(maize)___Northern_Leaf_Blight":               "Maize_NorthernLeafBlight",
    "Corn_(maize)___Common_rust_":                       "Maize_CommonRust",
    "Potato___Early_blight":                             "Potato_EarlyBlight",
    "Potato___Late_blight":                              "Potato_LateBlight",
    "Pepper,_bell___Bacterial_spot":                     "Chilli_LeafCurl",   # closest proxy
    # Healthy images — pull from multiple crops for variety
    "Rice___healthy":                                    "Healthy",
    "Corn_(maize)___healthy":                            "Healthy",
    "Potato___healthy":                                  "Healthy",
    "Pepper,_bell___healthy":                            "Healthy",
}

VAL_SPLIT  = 0.2   # 80% train, 20% test
SEED       = 42

# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(dest: Path):
    """Download from Kaggle using the kaggle CLI."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading PlantVillage dataset from Kaggle (~1.5 GB)…")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "vipoooool/new-plant-diseases-dataset",
         "--unzip", "-p", str(dest)],
        capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed.\n"
            "Make sure kaggle is installed (pip install kaggle) and\n"
            "~/.kaggle/kaggle.json is present with your API key."
        )
    print(f"Dataset extracted to: {dest}")


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Walk the raw PlantVillage folders and collect image paths per target class.
    PlantVillage may be nested under New Plant Diseases Dataset/train/ or similar.
    """
    collected: dict[str, list[Path]] = {label: [] for label in set(CLASS_MAP.values())}

    # Find the actual train folder (Kaggle zip may be nested)
    train_roots = list(raw_dir.rglob("train"))
    if not train_roots:
        raise FileNotFoundError(
            f"Could not find a 'train' folder under {raw_dir}.\n"
            "The Kaggle zip may have extracted differently — check the folder structure."
        )
    train_root = train_roots[0]
    print(f"Found train root: {train_root}")

    for pv_folder, label in CLASS_MAP.items():
        src = train_root / pv_folder
        if not src.exists():
            print(f"  ⚠️  Not found: {src.name}")
            continue
        imgs = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        collected[label].extend(imgs)
        print(f"  {label:30s} ← {pv_folder:45s}  ({len(imgs)} images)")

    return collected


def build_split(collected: dict, out_dir: Path, limit: int):
    """Copy images into train/ and test/ folders with the 80/20 split."""
    random.seed(SEED)

    for label, paths in collected.items():
        if not paths:
            print(f"  ⚠️  No images collected for class: {label} — skipping")
            continue

        random.shuffle(paths)
        if limit:
            paths = paths[:limit]

        split_at   = int(len(paths) * (1 - VAL_SPLIT))
        train_imgs = paths[:split_at]
        test_imgs  = paths[split_at:]

        for subset, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dest = out_dir / subset / label
            dest.mkdir(parents=True, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, dest / src.name)

        print(f"  {label:30s}  train={len(train_imgs)}  test={len(test_imgs)}")


def main(args):
    base    = Path(__file__).resolve().parent.parent   # repo root
    raw_dir = base / "data" / "raw" / "plantvillage"
    out_dir = base / "data"

    # Step 1 — Download
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        download_dataset(raw_dir)
    else:
        print(f"Raw dataset already present at {raw_dir} — skipping download.")

    # Step 2 — Collect
    print("\nCollecting images…")
    collected = collect_images(raw_dir)
    total = sum(len(v) for v in collected.values())
    print(f"Total images collected: {total}\n")

    # Step 3 — Split and copy
    print("Building train/test split…")
    build_split(collected, out_dir, args.limit)

    print(f"\n✅  Dataset ready at {out_dir}/train  and  {out_dir}/test")
    print("\nNext step — train the model:")
    print("    python ml/train_disease_model.py --epochs 20 --batch 32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PlantVillage dataset for Plotwise")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images per class (0 = no limit). Use ~500 for a quick test run.")
    main(parser.parse_args())
