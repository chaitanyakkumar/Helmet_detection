"""
Utility functions for dataset preprocessing, augmentation, and YOLO formatting.
This module centralizes helper functions used across data_cleaning.py,
data_augmentation.py, and training scripts.
"""

import os
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import albumentations as A

# To make the code reproducible
RANDOM_SEED = 42
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def simplify_osf_label(label: str) -> str:
    return "no_helmet" if "NoHelmet" in label else "helmet"

def to_yolo_bbox(x, y, w, h, img_w, img_h):
    return ((x + w/2)/img_w,
            (y + h/2)/img_h,
            w/img_w,
            h/img_h)

def write_label_file(out_path, cls_id, bbox):
    with open(out_path, "w") as f:
        f.write(f"{cls_id} " + " ".join(f"{b:.6f}" for b in bbox) + "\n")

def list_images(folder: str):
    """Return a sorted list of valid image file paths from the folder."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return sorted(files)


def read_rgb(path: str):
    """Read image in RGB format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, img_rgb: np.ndarray):
    """Save an RGB image to disk (converted to BGR)."""
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def clamp(img: np.ndarray) -> np.ndarray:
    """Clamp pixel values to the valid [0, 255] range."""
    return np.clip(img, 0, 255).astype(np.uint8)


def display_side_by_side(before, after, title: str, show: bool = False):
    """Display original and augmented image side-by-side."""
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(before)
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(after)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def simplify_osf_label(label: str) -> str:
    """
    Simplify the class label name for helmet detection dataset.
    Returns 'no_helmet' if the label contains 'NoHelmet', otherwise 'helmet'.
    """
    return "no_helmet" if "NoHelmet" in label else "helmet"


def to_yolo_bbox(x, y, w, h, img_w, img_h):
    """
    Convert pixel coordinates (x, y, w, h) to YOLO-normalized format.
    Returns (x_center, y_center, width, height) normalized to image size.
    """
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h
    )

def write_label_file(out_path: str, cls_id: int, bbox):
    """
    Write YOLO label file (.txt) with class ID and bounding box coordinates.
    """
    with open(out_path, "w") as f:
        f.write(f"{cls_id} " + " ".join(f"{b:.6f}" for b in bbox) + "\n")

def fractal_noise(h, w, octaves=5, base_scale=48, blur_sigma=0.0):
    """Generate fractal noise pattern for realistic fog and rain textures."""
    out = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    for o in range(octaves):
        scale = max(4, base_scale // (2 ** o))
        small = np.random.rand(max(1, h // scale), max(1, w // scale)).astype(np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        if blur_sigma > 0:
            layer = cv2.GaussianBlur(layer, (0, 0), blur_sigma)
        out += amplitude * layer
        amplitude *= 0.5
    out -= out.min()
    out /= (out.max() + 1e-6)
    return out


def apply_fog(img_rgb, beta=1.1, airlight=(238, 240, 255)):
    """Apply fog/haze effect to an RGB image."""
    h, w = img_rgb.shape[:2]
    img = img_rgb.astype(np.float32)
    noise = fractal_noise(h, w, base_scale=56)
    t = np.exp(-beta * noise)
    A = np.array(airlight, dtype=np.float32)
    fog = img * t[..., None] + A * (1 - t[..., None])
    return clamp(fog)


def apply_night(img_rgb, darkness=0.4, blue_tint=1.1):
    """Apply nighttime effect by darkening and adding blue tint."""
    img = img_rgb.astype(np.float32)
    img *= darkness
    r, g, b = cv2.split(img)
    b *= blue_tint
    img = cv2.merge([r, g, clamp(b)])
    return clamp(img)


def apply_sunny(img_rgb, brightness=1.1, contrast=1.05):
    """Apply bright daylight effect with higher contrast."""
    img = cv2.convertScaleAbs(img_rgb, alpha=contrast, beta=(brightness - 1) * 50)
    return clamp(img)


def get_albumentations():
    """Return a dictionary of Albumentations-based augmentation transforms."""
    return {
        "Blur": A.MotionBlur(blur_limit=70, p=1),
        "Rainy": A.Compose([
            A.RandomRain(blur_value=8, brightness_coefficient=0.8, drop_length=15, p=1),
            A.MotionBlur(blur_limit=10, p=0.8)
        ]),
        "Foggy": A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=1),
        "Dark": A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), contrast_limit=0.3, p=1),
        "LowQuality": A.ImageCompression(quality_lower=15, quality_upper=40, p=1),
        "Glare": A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=150, p=1),
        "Sunny": A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=1),
        "Geometric": A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=1),
    }