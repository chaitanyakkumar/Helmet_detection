import os
import glob
import re
import time

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract
import IPython.display as display  # optional, for Jupyter

# -------------------------------------------------------------------------
# CONFIG – EDIT THESE PATHS
# -------------------------------------------------------------------------
BEST_MODEL_PATH = "/content/drive/MyDrive/helmet_runs/helmet_yolov8s/weights/best.pt"

# Image inference
VAL_IMG_DIR = "/content/drive/MyDrive/helmet_dataset/images/val"
IMG_OUT_DIR = "/content/infer_results/helmet_lp_preds"

# Video inference
VIDEO_INPUT_PATH = "/content/videoplayback (1).mp4"
VIDEO_OUTPUT_PATH = "/content/videoplayback_helmet_lp_ocr.mp4"
SAMPLED_FRAMES_DIR = "/content/infer_results/video_sampled_frames"

DEVICE = 0        # GPU id or 'cpu'
CONF_THRESH = 0.5 # detection confidence threshold
TARGET_FPS = 3.0  # process ~3 frames per second from video


# OCR HELPERS
def ocr_license_plate(plate_img_bgr: np.ndarray) -> str:
    """
    Run a simple OCR pipeline on a cropped license plate image.
    Returns cleaned plate text (A–Z, 0–9) or '' if nothing good.
    Only called when detected class is 'license_plate'.
    """
    if plate_img_bgr is None or plate_img_bgr.size == 0:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale to help OCR
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Denoise + binarize
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract config: treat as a single line, restrict to letters+digits
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = pytesseract.image_to_string(thresh, config=config)
    # Keep only A–Z and 0–9
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # Basic sanity: license plates usually have >=4 chars
    if 4 <= len(text) <= 12:
        return text
    return ""


def draw_box_and_maybe_ocr(img: np.ndarray, box, cls_name: str, conf: float) -> None:
    """
    Draw bounding box + label on img.
    If class is 'license_plate', also run OCR and update label with plate text.
    Modifies img in-place.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Choose color by class
    if cls_name == "helmet":
        color = (0, 255, 0)  # green
        label = f"helmet {conf:.2f}"
    elif cls_name == "no_helmet":
        color = (0, 0, 255)  # red
        label = f"no_helmet {conf:.2f}"
    elif cls_name == "license_plate":
        color = (255, 255, 0)  # cyan/yellow
        # ---- OCR ONLY for license_plate ----
        h, w, _ = img.shape
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        plate_crop = img[y1c:y2c, x1c:x2c]
        plate_text = ocr_license_plate(plate_crop)
        if plate_text:
            label = f"{plate_text} ({conf:.2f})"
        else:
            label = f"license_plate {conf:.2f}"
    else:
        color = (255, 255, 255)
        label = f"{cls_name} {conf:.2f}"

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    y_text = max(0, y1 - 10)
    cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw, y_text + baseline), color, -1)

    # Choose black text for bright colors, white otherwise
    text_color = (0, 0, 0) if color in [(255, 255, 0), (255, 255, 255)] else (255, 255, 255)

    cv2.putText(
        img,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
    )



# IMAGE INFERENCE
 
def run_image_inference(model: YOLO):
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    print(f"Running image inference on: {VAL_IMG_DIR}")
    results = model.predict(
        source=VAL_IMG_DIR,
        conf=CONF_THRESH,
        device=DEVICE,
        verbose=False
    )

    for i, r in enumerate(results):
        img = r.orig_img.copy()
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = model.names[cls_id]
            draw_box_and_maybe_ocr(img, box, cls_name, conf)

        out_path = os.path.join(IMG_OUT_DIR, f"pred_{i}.jpg")
        cv2.imwrite(out_path, img)

    # Optionally display a few images (useful in notebooks)
    pred_images = sorted(glob.glob(os.path.join(IMG_OUT_DIR, "*.jpg")))[:5]
    for img_path in pred_images:
        display.display(Image.open(img_path))

    print(f"Saved annotated images to: {IMG_OUT_DIR}")


# VIDEO INFERENCE WITH FRAME SAMPLING
def run_video_inference_sampled(model: YOLO):
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"Video not found: {VIDEO_INPUT_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_INPUT_PATH}")
        return

    os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
    os.makedirs(SAMPLED_FRAMES_DIR, exist_ok=True)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 24.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process only every Nth frame so we get ~TARGET_FPS frames per second
    frame_stride = max(1, int(round(orig_fps / TARGET_FPS)))
    print(f"Original FPS: {orig_fps:.2f}, target FPS: {TARGET_FPS}, stride: {frame_stride}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, TARGET_FPS, (width, height))

    frame_idx = 0
    kept_idx = 0
    times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run YOLO on every 'frame_stride'-th frame
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        start = time.time()
        result = model.predict(
            frame,
            imgsz=640,
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False
        )[0]
        times.append(time.time() - start)

        annotated = frame.copy()
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = model.names[cls_id]
            draw_box_and_maybe_ocr(annotated, box, cls_name, conf)

        out_writer.write(annotated)

        # Save sampled frame as image
        out_img_path = os.path.join(SAMPLED_FRAMES_DIR, f"frame_{kept_idx:05d}.jpg")
        cv2.imwrite(out_img_path, annotated)

        kept_idx += 1
        frame_idx += 1

    cap.release()
    out_writer.release()

    if times:
        avg_t = np.mean(times)
        eff_fps = 1.0 / avg_t
        print(f"Processed {kept_idx} sampled frames. Effective model FPS: {eff_fps:.2f}")
    else:
        print("No frames processed.")

    print(f"Saved annotated video to: {VIDEO_OUTPUT_PATH}")
    print(f"Saved sampled annotated frames to: {SAMPLED_FRAMES_DIR}")


# MAIN
if __name__ == "__main__":
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_PATH}")

    print(f"Loading YOLO model from: {BEST_MODEL_PATH}")
    model = YOLO(BEST_MODEL_PATH)  # model.names must be ['helmet','no_helmet','license_plate']

    run_image_inference(model)
    run_video_inference_sampled(model)