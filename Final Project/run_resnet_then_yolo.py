import time
from hazard_models import (
    load_resnet18_model,
    load_yolo_model,
    classify_image,
    run_yolo_inference,
    print_yolo_detections
)




IMAGE_PATH = "../not_uploaded/Industrial Hazards.v1i.yolov8/valid/images/Water_Leaks_water197_jpg.rf.689ed22d03bce195deb4fe72a097048a.jpg"

RESNET_CHECKPOINT = "hazard_resnet_results/resnet_18_pretrained_frozen_backbone_best_model.pth"

YOLO_CHECKPOINT = r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/best.pt"

OUTPUT_DIR = "pipeline_outputs/resnet_then_yolo"

CLASSIFIER_CONF_THRESHOLD = 0.70
YOLO_CONF_THRESHOLD = 0.05


def main():
    print("Pipeline: ResNet-18 detection/classification -> YOLO localization")
    print("Image:", IMAGE_PATH)

    resnet_model = load_resnet18_model(RESNET_CHECKPOINT)
    yolo_model = load_yolo_model(YOLO_CHECKPOINT)

    total_start = time.time()

    predicted_class, classifier_conf, resnet_time = classify_image(
        resnet_model,
        IMAGE_PATH
    )

    print("\nResNet-18 result:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {classifier_conf:.3f}")
    print(f"ResNet inference time: {resnet_time:.4f} sec")

    if classifier_conf >= CLASSIFIER_CONF_THRESHOLD:
        print("\nResNet confidence passed threshold. Running YOLO...")

        detections, yolo_time = run_yolo_inference(
            yolo_model,
            IMAGE_PATH,
            output_dir=OUTPUT_DIR,
            conf_threshold=YOLO_CONF_THRESHOLD
        )

        print_yolo_detections(detections)

    else:
        print("\nResNet confidence below threshold. YOLO was not run.")
        detections = []
        yolo_time = 0.0

    total_time = time.time() - total_start

    print("\nTiming Summary")
    print(f"ResNet time: {resnet_time:.4f} sec")
    print(f"YOLO time: {yolo_time:.4f} sec")
    print(f"Total pipeline time: {total_time:.4f} sec")
    print(f"YOLO was run: {classifier_conf >= CLASSIFIER_CONF_THRESHOLD}")


if __name__ == "__main__":
    main()