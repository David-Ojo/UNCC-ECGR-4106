import os
import time
import cv2
from resource_tracker import ResourceTracker, save_pipeline_results

from hazard_models import (
    load_yolo_model,
    run_yolo_inference,
    print_yolo_detections
)

PIPELINE_NAME = "YOLO_only"

YOLO_GFLOPS = 8.2
#YOLO_GFLOPS = 28.7

RESULTS_CSV = "pipeline_outputs/resource_comparison_Yolo.csv"
 
 
 

VIDEO_SOURCE = "test_video.mp4"


#YOLO_CHECKPOINT = r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bests.pt"
YOLO_CHECKPOINT = r"runs/detect/hazard_yolo_results/yolov8n_hazard_detection/weights/bestn.pt"

OUTPUT_DIR = "pipeline_outputs/yolo_only_video"

FRAME_INTERVAL_SECONDS = 2.0
YOLO_CONF_THRESHOLD = 0.05


def save_frame(frame, output_dir, frame_count):
    os.makedirs(output_dir, exist_ok=True)

    frame_path = os.path.join(output_dir, f"sampled_frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)

    return frame_path


def main():
    print("Pipeline: YOLO-only video hazard localization")
    print("Video source:", VIDEO_SOURCE)

    yolo_model = load_yolo_model(YOLO_CHECKPOINT)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_step = int(fps * FRAME_INTERVAL_SECONDS)

    print("FPS:", fps)
    print("Frame step:", frame_step)

    frame_count = 0
    sampled_count = 0

    total_yolo_time = 0.0

    start_total = time.time()
    tracker = ResourceTracker()
    tracker.start()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_step == 0:
            sampled_count += 1

            frame_path = save_frame(
                frame,
                output_dir=os.path.join(OUTPUT_DIR, "sampled_frames"),
                frame_count=frame_count
            )

            print("\n" + "=" * 60)
            print(f"Frame {frame_count}")
            print("Running YOLO...")

            detections, yolo_time = run_yolo_inference(
                yolo_model,
                frame_path,
                output_dir=os.path.join(OUTPUT_DIR, "yolo_outputs"),
                conf_threshold=YOLO_CONF_THRESHOLD
            )

            total_yolo_time += yolo_time

            print_yolo_detections(detections)
            print(f"YOLO inference time: {yolo_time:.4f} sec")

        frame_count += 1

    cap.release()

    total_time = time.time() - start_total
    tracker.stop()
    resource_summary = tracker.get_summary()
    yolo_runs = sampled_count
    yolo_calls_avoided = 0
    yolo_avoidance_percent = 0.0

    total_estimated_gflops = sampled_count * YOLO_GFLOPS

    if sampled_count > 0:
        avg_estimated_gflops_per_sample = total_estimated_gflops / sampled_count
    else:
        avg_estimated_gflops_per_sample = 0.0

    result_row = {
        "pipeline": PIPELINE_NAME,
        "total_frames_read": frame_count,
        "sampled_frames": sampled_count,
        "classifier_runs": 0,
        "yolo_runs": yolo_runs,
        "yolo_calls_avoided": yolo_calls_avoided,
        "yolo_avoidance_percent": yolo_avoidance_percent,
        "total_classifier_time_sec": 0.0,
        "total_yolo_time_sec": total_yolo_time,
        "total_wall_time_sec": resource_summary["wall_time_sec"],
        "peak_gpu_memory_mb": resource_summary["peak_gpu_mem_mb"],
        "cpu_ram_mb": resource_summary["cpu_ram_mb"],
        "estimated_total_gflops": total_estimated_gflops,
        "estimated_gflops_per_sampled_frame": avg_estimated_gflops_per_sample
    }

    save_pipeline_results(RESULTS_CSV, result_row)
    print("\n" + "=" * 60)
    print("Video Pipeline Summary")
    print("Total frames read:", frame_count)
    print("Sampled frames processed:", sampled_count)
    print("Total YOLO time:", f"{total_yolo_time:.4f} sec")
    print("Total time:", f"{total_time:.4f} sec")

    if sampled_count > 0:
        print("Average YOLO time per sampled frame:", f"{total_yolo_time / sampled_count:.4f} sec")

    print("YOLO calls avoided:", yolo_calls_avoided)
    print("YOLO avoidance percent:", f"{yolo_avoidance_percent:.2f}%")
    print("Peak GPU memory:", f"{resource_summary['peak_gpu_mem_mb']:.2f} MB")
    print("CPU RAM:", f"{resource_summary['cpu_ram_mb']:.2f} MB")
    print("Estimated total GFLOPs:", f"{total_estimated_gflops:.2f}")
    print("Estimated GFLOPs per sampled frame:", f"{avg_estimated_gflops_per_sample:.2f}")
    print("Saved resource comparison to:", RESULTS_CSV)
if __name__ == "__main__":
    main()