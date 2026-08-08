import time
from hazard_models import (
    load_yolo_model,
    run_yolo_inference,
    print_yolo_detections
)




IMAGE_PATH = "../not_uploaded/Industrial Hazards.v1i.yolov8/valid/images/Water_Leaks_water197_jpg.rf.689ed22d03bce195deb4fe72a097048a.jpg"

#YOLO_CHECKPOINT = r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bests.pt"
YOLO_CHECKPOINT = r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bestn.pt"

OUTPUT_DIR = "pipeline_outputs/yolo_only"

YOLO_CONF_THRESHOLD = 0.25


def main():
    print("Pipeline: YOLO detection + localization")
    print("Image:", IMAGE_PATH)

    yolo_model = load_yolo_model(YOLO_CHECKPOINT)

    total_start = time.time()

    detections, yolo_time = run_yolo_inference(
        yolo_model,
        IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=YOLO_CONF_THRESHOLD
    )

    total_time = time.time() - total_start

    print_yolo_detections(detections)

    print("\nTiming Summary")
    print(f"YOLO inference time: {yolo_time:.4f} sec")
    print(f"Total pipeline time: {total_time:.4f} sec")


if __name__ == "__main__":
    main()