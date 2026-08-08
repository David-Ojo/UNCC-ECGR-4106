from ultralytics import YOLO

models_to_check = {
    "YOLOv8n base": "yolov8n.pt",
    "YOLOv8s base": "yolov8s.pt",

    # Optional: replace these with your actual trained paths
    "YOLOv8n trained": r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bests.pt",
    "YOLOv8s trained": r"runs/detect/hazard_yolo_results/yolov8s_hazard_detection/weights/bestn.pt",
}

for name, path in models_to_check.items():
    print("\n" + "=" * 70)
    print(name)
    print("Path:", path)

    model = YOLO(path)

    # verbose=True prints layers, params, gradients, and GFLOPs
    model.info(verbose=True)