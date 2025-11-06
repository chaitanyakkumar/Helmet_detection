"""
Helmet vs No-Helmet YOLOv8 Training & Inference
- Multiple experiments with freezing/unfreezing layers
- Dynamic learning rate (cosine schedule)
- Final experiment: unfreeze detection head + last 10 backbone layers
"""

from ultralytics import YOLO
import cv2, os, glob, time
import numpy as np
from PIL import Image
import IPython.display as display
import matplotlib.pyplot as plt

# CONFIG
BASE_WEIGHTS = 'yolov8s.pt'
DATA_YAML = '/content/drive/MyDrive/helmet_dataset/data.yaml'

PROJECT = 'helmet_runs'
BASE_NAME = 'helmet_yolov8s'
SAVE_ROOT = '/content/drive/MyDrive/helmet_runs'

VAL_IMG_DIR = '/content/drive/MyDrive/helmet_dataset/images/val'
INFER_SAVE_DIR = '/content/infer_results/helmet_pred_custom'
os.makedirs(INFER_SAVE_DIR, exist_ok=True)

VIDEO_PATH = '/content/videoplayback (1).mp4'
NUM_FPS_FRAMES = 100

DEVICE = 0  # GPU id


# HELPER: FREEZE / UNFREEZE
def ocr_license_plate(plate_img_bgr: np.ndarray) -> str:
    """
    Run a simple OCR pipeline on a cropped license plate image.
    Returns cleaned plate text (A–Z, 0–9) or '' if nothing good.
    """
    if plate_img_bgr is None or plate_img_bgr.size == 0:
        return ""

    # Preprocess
    gray = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale to help OCR
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Denoise + binarize
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract config: single line, only letters+digits
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = pytesseract.image_to_string(thresh, config=config)
    # Clean: keep only A–Z, 0–9
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # Basic sanity check
    if 4 <= len(text) <= 12:
        return text
    return ""

def freeze_all_params(model):
    """Set requires_grad=False for all model parameters."""
    for p in model.model.parameters():
        p.requires_grad = False


def unfreeze_all_params(model):
    """Set requires_grad=True for all model parameters."""
    for p in model.model.parameters():
        p.requires_grad = True


def unfreeze_head_and_last_n_backbone_layers(model, n_backbone_layers=10):
    """
    For YOLOv8:
    - model.model is the top-level Ultralytics Model
    - model.model.model is typically a nn.ModuleList of layers (backbone + head)
    Strategy:
      1. Freeze everything
      2. Unfreeze detection head
      3. Unfreeze last N backbone layers
    """
    # 1. Freeze everything first
    freeze_all_params(model)

    # 2. Get ordered layers (backbone + head)
    layers = list(model.model.model)

    if len(layers) < 2:
        raise RuntimeError("Unexpected YOLOv8 model structure. layers < 2.")

    backbone_layers = layers[:-1]
    head_layer = layers[-1]  # detection head

    # 3. Unfreeze head
    for p in head_layer.parameters():
        p.requires_grad = True

    # 4. Unfreeze last N backbone layers
    n = min(n_backbone_layers, len(backbone_layers))
    for layer in backbone_layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


# TRAINING EXPERIMENTS
def run_experiments():
    """
    Run multiple training experiments with different freezing strategies.
    Uses cosine LR as a dynamic learning rate schedule.
    """
    experiments = [
        # EXP 1: Freeze first 10 layers (mostly backbone), train head + later layers
        {
            "name": f"{BASE_NAME}_freeze10",
            "epochs": 20,
            "train_kwargs": dict(
                data=DATA_YAML,
                imgsz=640,
                batch=16,
                lr0=1e-3,
                lrf=1e-2,         # final LR = lr0 * lrf (cosine end)
                cos_lr=True,      # dynamic LR
                device=DEVICE,
                project=PROJECT,
                pretrained=True,
                workers=2,
                save_dir=SAVE_ROOT,
                freeze=10,        # built-in: freeze first 10 layers
            ),
        },

        # EXP 2: Freeze fewer layers (freeze=5), allow more of backbone to fine-tune
        {
            "name": f"{BASE_NAME}_freeze5",
            "epochs": 20,
            "train_kwargs": dict(
                data=DATA_YAML,
                imgsz=640,
                batch=16,
                lr0=5e-4,         # smaller LR (deeper fine-tuning)
                lrf=1e-3,
                cos_lr=True,
                device=DEVICE,
                project=PROJECT,
                pretrained=True,
                workers=2,
                save_dir=SAVE_ROOT,
                freeze=5,
            ),
        },

        # EXP 3: Custom freeze: only head + last 10 backbone layers trainable
        # We handle the freezing manually via requires_grad.
        {
            "name": f"{BASE_NAME}_head_plus_last10",
            "epochs": 30,
            "train_kwargs": dict(
                data=DATA_YAML,
                imgsz=640,
                batch=16,
                lr0=3e-4,
                lrf=1e-3,
                cos_lr=True,
                device=DEVICE,
                project=PROJECT,
                pretrained=True,
                workers=2,
                save_dir=SAVE_ROOT,
                freeze=0,   # let our manual freezing control trainable params
            ),
            "custom_freeze": True,
        },
    ]

    last_run_name = None

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")

        # Fresh model for each experiment
        model = YOLO(BASE_WEIGHTS)

        # Optional custom freezing logic
        if exp.get("custom_freeze", False):
            print("Applying custom freeze: head + last 10 backbone layers...")
            unfreeze_head_and_last_n_backbone_layers(model, n_backbone_layers=10)

        # Train
        model.train(
            epochs=exp["epochs"],
            name=exp["name"],
            **exp["train_kwargs"],
        )

        last_run_name = exp["name"]

    return last_run_name


# INFERENCE UTILS
def load_best_model(run_name: str):
    """
    Load best weights from a given experiment run.
    Ultralytics default: runs are inside PROJECT/run_name/weights/best.pt
    """
    best_model_path = os.path.join(SAVE_ROOT, run_name, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    print(f"Loading best model from: {best_model_path}")
    return YOLO(best_model_path), best_model_path


def run_image_inference(model):
    """
    Run inference on validation images and save annotated outputs.
    """
    names = model.names  # class name mapping

    print(f"\nRunning image inference on: {VAL_IMG_DIR}")
    results = model.predict(source=VAL_IMG_DIR, conf=0.5, device=DEVICE, verbose=False)

    for i, r in enumerate(results):
        img = r.orig_img.copy()

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{names[cls_id]} {conf:.2f}"

            color = (0, 255, 0) if names[cls_id] == 'helmet' else (0, 0, 255)

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)

            # Label text
            cv2.putText(
                img, label, (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

        out_path = os.path.join(INFER_SAVE_DIR, f"pred_{i}.jpg")
        cv2.imwrite(out_path, img)

    # Display a few predictions
    pred_images = sorted(glob.glob(os.path.join(INFER_SAVE_DIR, '*.jpg')))[:5]
    for img_path in pred_images:
        display.display(Image.open(img_path))


def evaluate_model(model):
    """
    Run built-in evaluation (mAP, precision, recall) on the val set from data.yaml.
    """
    print("\nEvaluating model on validation set...")
    metrics = model.val(data=DATA_YAML, device=DEVICE)
    print(f"mAP@0.5:       {metrics.box.map50 * 100:.2f}%")
    print(f"Precision:     {metrics.box.mp * 100:.2f}%")
    print(f"Recall:        {metrics.box.mr * 100:.2f}%")
    print(f"mAP@0.5:0.95:  {metrics.box.map * 100:.2f}%")
    return metrics


def benchmark_video_fps(model):
    """
    Benchmark inference speed on a 1080p video for NUM_FPS_FRAMES frames.
    """
    print(f"\nBenchmarking inference FPS on video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    times = []

    for i in range(NUM_FPS_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        _ = model.predict(frame, imgsz=640, device=DEVICE, verbose=False)
        times.append(time.time() - start)

    cap.release()

    if len(times) == 0:
        print("No frames read from video, cannot compute FPS.")
        return None

    fps = 1.0 / np.mean(times)
    print(f"Inference speed: {fps:.2f} FPS on ~1080p video")
    return fps



if __name__ == "__main__":
    # 1. Run training experiments (including the final: head + last 10 backbone layers)
    last_run_name = run_experiments()

    # 2. Load best model from the last experiment (you can change to any run name)
    model, best_model_path = load_best_model(last_run_name)

    # 3. Image inference + visualization
    run_image_inference(model)

    # 4. Evaluation metrics
    evaluate_model(model)

    # 5. FPS benchmark on 1080p video
    benchmark_video_fps(model)
