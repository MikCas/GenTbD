"""Performance profiling script to identify bottlenecks in detection pipeline."""

import time
import torch
import numpy as np
import cv2
from src.detecting.detectors.object_detector import ObjectDetector

def profile_class_filtering():
    """Profile the class filtering implementation."""
    print("\n=== Class Filtering Performance ===")

    # Simulate detection output
    num_detections = 100
    labels = torch.randint(1, 80, (num_detections,))
    classes_to_filter = [1, 2, 3]

    # Current implementation (slow)
    start = time.time()
    for _ in range(1000):
        class_mask = torch.zeros(len(labels), dtype=torch.bool)
        for class_id in classes_to_filter:
            class_mask |= (labels == class_id)
    slow_time = time.time() - start

    # Optimized implementation
    start = time.time()
    for _ in range(1000):
        class_mask = torch.isin(labels, torch.tensor(classes_to_filter))
    fast_time = time.time() - start

    print(f"Current (loop-based): {slow_time*1000:.2f}ms for 1000 iterations")
    print(f"Optimized (torch.isin): {fast_time*1000:.2f}ms for 1000 iterations")
    print(f"Speedup: {slow_time/fast_time:.1f}x faster")

def profile_frame_copy():
    """Profile frame copying overhead."""
    print("\n=== Frame Copy Performance ===")

    # Simulate 1080p frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Test copy performance
    times = []
    for _ in range(100):
        start = time.time()
        copy = frame.copy()
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    print(f"Average frame copy time: {avg_time:.2f}ms")
    print(f"Memory copied: {frame.nbytes / 1024 / 1024:.2f}MB")
    print(f"Impact at 30 FPS: {avg_time * 30:.2f}ms/second")

def profile_detection_pipeline():
    """Profile the full detection pipeline."""
    print("\n=== Full Detection Pipeline ===")

    # Create detector
    detector = ObjectDetector(
        model='mobilenet',
        device='cpu',
        conf_threshold=0.5,
        classes=[1]  # Only people
    )

    # Create test frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warm up
    for _ in range(3):
        _ = detector.detect(frame)

    # Profile
    times = {
        'preprocess': [],
        'inference': [],
        'postprocess': [],
        'total': []
    }

    for _ in range(20):
        start_total = time.time()

        # Preprocess
        start = time.time()
        input_tensor = detector.preprocess(frame)
        times['preprocess'].append((time.time() - start) * 1000)

        # Inference
        start = time.time()
        with torch.no_grad():
            output = detector.inference(input_tensor)
        times['inference'].append((time.time() - start) * 1000)

        # Postprocess
        start = time.time()
        detections = detector.postprocess(output, frame.shape[:2])
        times['postprocess'].append((time.time() - start) * 1000)

        times['total'].append((time.time() - start_total) * 1000)

    # Print results
    for stage, stage_times in times.items():
        avg = np.mean(stage_times)
        std = np.std(stage_times)
        pct = (avg / np.mean(times['total']) * 100) if stage != 'total' else 100
        print(f"{stage:12s}: {avg:6.2f}ms ± {std:5.2f}ms ({pct:5.1f}%)")

    fps = 1000.0 / np.mean(times['total'])
    print(f"\nEstimated FPS: {fps:.1f}")

def profile_cpu_gpu_transfer():
    """Profile CPU to GPU transfer overhead."""
    print("\n=== CPU-GPU Transfer Performance ===")

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS not available on this system")
        return

    device = 'mps'

    # Create test tensors of different sizes
    sizes = [
        (640, 480, 3),   # VGA
        (1280, 720, 3),  # 720p
        (1920, 1080, 3)  # 1080p
    ]

    for size in sizes:
        # Create tensor on CPU
        tensor_cpu = torch.randn(3, size[0], size[1])

        # Measure transfer time
        times = []
        for _ in range(100):
            start = time.time()
            tensor_gpu = tensor_cpu.to(device)
            torch.mps.synchronize()  # Wait for transfer
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        print(f"{size[1]}x{size[0]}: {avg_time:.2f}ms per transfer")

if __name__ == '__main__':
    print("=" * 60)
    print("GenTbD Performance Profiling")
    print("=" * 60)

    profile_class_filtering()
    profile_frame_copy()
    profile_detection_pipeline()

    # Only run if MPS available
    try:
        profile_cpu_gpu_transfer()
    except Exception as e:
        print(f"\nCPU-GPU profiling skipped: {e}")

    print("\n" + "=" * 60)
    print("Profiling Complete")
    print("=" * 60)
