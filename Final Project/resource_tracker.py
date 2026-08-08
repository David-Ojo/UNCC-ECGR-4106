import os
import time
import csv
import psutil
import torch


class ResourceTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.time()

    def get_summary(self):
        wall_time = self.end_time - self.start_time

        cpu_ram_mb = self.process.memory_info().rss / (1024 ** 2)

        if torch.cuda.is_available():
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_gpu_mem_mb = 0.0

        return {
            "wall_time_sec": wall_time,
            "cpu_ram_mb": cpu_ram_mb,
            "peak_gpu_mem_mb": peak_gpu_mem_mb
        }


def save_pipeline_results(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)