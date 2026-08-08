from ultralytics import YOLO

model_paths = {
    "YOLOv8n trained": r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bestn.pt",
    "YOLOv8s trained": r"runs/detect/hazard_yolo_results/yolov8s_hazard_detection/weights/bests.pt"
}

for model_name, weight_path in model_paths.items():
    model = YOLO(weight_path)

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    model_size_mb = total_params * 4 / (1024 ** 2)

    print("=" * 50)
    print(model_name)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size from params: {model_size_mb:.2f} MB")