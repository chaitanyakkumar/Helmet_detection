"""
Script for augmenting training images using custom OpenCV and Albumentations transformations.
"""

import os
from utils import (
    list_images,
    read_rgb,
    save_image,
    apply_fog,
    apply_night,
    apply_sunny,
    get_albumentations,
    display_side_by_side
)

# Paths
DATASET_DIR = r"D:\Capstone project\data\helmet_dataset\images\train"
OUTPUT_DIR = r"D:\Capstone project\data\aug_outputs"
SHOW = False   
SAVE = True


def augment_dataset():
    """Main function to perform data augmentation and save augmented images."""
    files = list_images(DATASET_DIR)
    alb_augs = get_albumentations()

    print(f"Found {len(files)} images in {DATASET_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, path in enumerate(files):
        img = read_rgb(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        # --- Custom augmentations (from utils) ---
        custom_augments = {
            "Foggy": apply_fog(img),
            "Night": apply_night(img),
            "Sunny": apply_sunny(img),
        }

        # --- Albumentations augmentations ---
        alb_augments = {k: aug(image=img)["image"] for k, aug in alb_augs.items()}

        # Combine all augmentations
        all_augments = {**custom_augments, **alb_augments}

        for name, aug_img in all_augments.items():
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.jpg")
            if SAVE:
                save_image(out_path, aug_img)
            display_side_by_side(img, aug_img, name, show=SHOW)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(files)} images...")

    print(f"Augmentation complete. Augmented images saved to:\n{os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    augment_dataset()
