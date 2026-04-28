"""
Plotwise — PlantVillage Dataset Preparation Script
Author: Limawapang Jamir

Reads already-downloaded PlantVillage 'color' folder, renames classes to
match our model, and splits into train/test:

    data/
      train/   (80%)
      test/    (20%)

Usage:
    python ml/prepare_dataset.py
    (no arguments needed — paths are set below)
"""

import shutil
import random
from pathlib import Path

# ── CONFIGURE THESE TWO PATHS ────────────────────────────────────────────────
SRC = r"D:\projects\plotwised dataset\plantvillage dataset\color"
DST = r"D:\projects\plotwise\data"
# ─────────────────────────────────────────────────────────────────────────────

SPLIT = 0.8   # 80% train, 20% test
SEED  = 42

# Flexible mapping — tries both double (__) and triple (___) underscore variants
# since different Kaggle uploads use different conventions.
# Format:  "keyword fragments to match" → "our class name"
# Matching is done by normalising folder names (lowercase, strip punctuation).
FOLDER_MAP = {
    # Maize / Corn
    "corn_maize_cercospora":          "Maize_Cercospora_GrayLeafSpot",
    "corn_maize_common_rust":         "Maize_CommonRust",
    "corn_maize_northern_leaf":       "Maize_NorthernLeafBlight",
    "corn_maize_healthy":             "Healthy_Maize",

    # Potato
    "potato_early_blight":            "Potato_EarlyBlight",
    "potato_late_blight":             "Potato_LateBlight",
    "potato_healthy":                 "Healthy_Potato",

    # Pepper / Chilli
    "pepper_bell_bacterial_spot":     "Pepper_BacterialSpot",
    "pepper_bell_healthy":            "Healthy_Pepper",

    # Tomato
    "tomato_bacterial_spot":          "Tomato_BacterialSpot",
    "tomato_early_blight":            "Tomato_EarlyBlight",
    "tomato_late_blight":             "Tomato_LateBlight",
    "tomato_leaf_mold":               "Tomato_LeafMold",
    "tomato_septoria":                "Tomato_SeptoriaLeafSpot",
    "tomato_yellow_leaf_curl":        "Tomato_YellowLeafCurl",
    "tomato_healthy":                 "Healthy_Tomato",

    # Orange / Citrus
    "orange_haunglongbing":           "Orange_Haunglongbing",

    # Soybean
    "soybean_healthy":                "Soybean_Healthy",

    # Apple
    "apple_apple_scab":               "Apple_AppleScab",
    "apple_black_rot":                "Apple_BlackRot",

    # Grape
    "grape_black_rot":                "Grape_BlackRot",
    "grape_esca":                     "Grape_Esca",
}

# Rice and Chilli_LeafCurl are not in standard PlantVillage — skipped automatically.


def normalise(name: str) -> str:
    """Lowercase, replace spaces/commas/brackets/underscores → single underscore."""
    import re
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def build_match_table(src_path: Path) -> dict[str, Path]:
    """Return {normalised_folder_name: actual_path} for all subfolders."""
    return {normalise(f.name): f for f in src_path.iterdir() if f.is_dir()}


def prepare():
    random.seed(SEED)
    src_path = Path(SRC)
    dst_path = Path(DST)

    if not src_path.exists():
        raise FileNotFoundError(f"Source folder not found:\n  {src_path}\nUpdate SRC in this script.")

    match_table = build_match_table(src_path)

    print(f"\nSource : {src_path}")
    print(f"Output : {dst_path}")
    print(f"Found  : {len(match_table)} folders in source\n")

    matched = 0
    not_found = []

    for key, our_name in FOLDER_MAP.items():
        # Find a folder whose normalised name contains all key fragments
        hit = next((path for norm, path in match_table.items() if key in norm), None)

        if hit is None:
            not_found.append(key)
            continue

        images = [f for f in hit.iterdir()
                  if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        random.shuffle(images)

        cut         = int(len(images) * SPLIT)
        train_imgs  = images[:cut]
        test_imgs   = images[cut:]

        for split_name, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dest = dst_path / split_name / our_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  [OK] {our_name:42s}  train={len(train_imgs):4d}  test={len(test_imgs):4d}")
        matched += 1

    print(f"\nDone: {matched} classes prepared")

    if not_found:
        print(f"\nNot found (normal for Rice/Chilli — not in PlantVillage): {len(not_found)}")
        for nf in not_found:
            print(f"  - {nf}")

    total_train = sum(1 for f in (dst_path / "train").rglob("*") if f.is_file())
    total_test  = sum(1 for f in (dst_path / "test").rglob("*")  if f.is_file())
    print(f"\nTotal images — train: {total_train:,}   test: {total_test:,}")
    print(f"\nDataset ready at: {dst_path}")
    print("\nNext — upload 'data/' folder to Google Drive and run training on Colab:")
    print("    python ml/train_disease_model.py --data-dir /content/data --epochs 30 --batch 32")


if __name__ == "__main__":
    prepare()
