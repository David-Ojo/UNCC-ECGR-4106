import os
import time
import csv
from ultralytics import YOLO
from pathlib import Path



DATA_ROOT = "../not_uploaded/Industrial Hazards.v1i.yolov8"
DATA_YAML = os.path.join(DATA_ROOT, "data.yaml")

RESULTS_DIR = "hazard_yolo_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "hazard_yolo_results.csv")

MODEL_WEIGHTS = "yolov8s.pt"
#   "yolov8n.pt"  -> smallest, fastest, best for edge deployment
#   "yolov8s.pt"  -> better accuracy, slower/larger

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0


PROJECT_NAME = RESULTS_DIR
RUN_NAME = "yolov8n_hazard_detection"




def save_results(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Dataset YAML:", DATA_YAML)
    print("Model weights:", MODEL_WEIGHTS)

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Could not find data.yaml at: {DATA_YAML}")

    model = YOLO(MODEL_WEIGHTS)

    total_start = time.time()

    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        pretrained=True,
        patience=15,
        plots=True,
        exist_ok=True
    )

    total_training_time = time.time() - total_start

    print("\nTraining complete.")
    print(f"Total training time: {total_training_time:.2f} seconds")

    save_dir = Path(model.trainer.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"

    print("Actual YOLO save directory:", save_dir)
    print("Best model path:", best_model_path)

    best_model = YOLO(str(best_model_path))

    val_start = time.time()

    metrics = best_model.val(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True
    )

    val_time = time.time() - val_start

    # Box detection metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("\nValidation/Test Detection Results")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"Validation time: {val_time:.2f} seconds")

    result_row = {
        "model": MODEL_WEIGHTS,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "total_training_time_sec": total_training_time,
        "validation_time_sec": val_time,
        "best_model_path": best_model_path
    }

    save_results(result_row)

    print("\nSaved summary results to:", RESULTS_CSV)


if __name__ == "__main__":
    main()