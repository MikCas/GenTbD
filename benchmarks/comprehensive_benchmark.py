#!/usr/bin/env python3
"""
Comprehensive Detector Performance Benchmark

Compares all detector types across devices to identify optimal configurations.
Tests object detection, keypoint detection, and ReID feature extraction.

Usage:
    python benchmarks/comprehensive_benchmark.py --video data/TownCent.mp4
    python benchmarks/comprehensive_benchmark.py --video data/TownCent.mp4 --devices cpu mps
    python benchmarks/comprehensive_benchmark.py --quick  # Fast test with fewer iterations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple
from tabulate import tabulate
import json

from src.detecting.detectors import ObjectDetector, KeypointDetector, ReIDDetector


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Benchmark configurations
RESOLUTIONS = [
    (320, 180, "320p"),
    (640, 360, "640p"),
    (960, 540, "960p"),
    (1280, 720, "720p"),
]

OBJECT_MODELS = ['mobilenet', 'resnet50', 'retinanet']
KEYPOINT_MODELS = ['resnet50']
REID_MODELS = ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5']


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def get_available_devices() -> List[str]:
    """Detect available devices on this system."""
    devices = ['cpu']

    if torch.backends.mps.is_available():
        devices.append('mps')

    if torch.cuda.is_available():
        devices.append('cuda')

    return devices


def warmup_detector(detector, frame: np.ndarray, num_warmup: int = 3):
    """Warmup detector to compile shaders and optimize."""
    for _ in range(num_warmup):
        _ = detector.detect(frame)


def benchmark_detector(
    detector,
    frame: np.ndarray,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> Dict[str, float]:
    """Benchmark a detector on a frame.

    Args:
        detector: Detector instance to benchmark
        frame: Frame to run detection on
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dict with timing statistics
    """
    # Warmup
    warmup_detector(detector, frame, warmup_iterations)

    # Benchmark
    times = []
    detection_counts = []

    for _ in range(num_iterations):
        start = time.time()
        detections = detector.detect(frame)
        elapsed = time.time() - start

        times.append(elapsed * 1000)  # Convert to ms
        detection_counts.append(len(detections))

    # Calculate statistics (skip first after warmup to be safe)
    times_clean = times[1:] if len(times) > 1 else times

    return {
        'mean_ms': np.mean(times_clean),
        'std_ms': np.std(times_clean),
        'min_ms': np.min(times_clean),
        'max_ms': np.max(times_clean),
        'fps': 1000 / np.mean(times_clean),
        'detections': np.mean(detection_counts),
    }


def benchmark_object_detector(
    model: str,
    device: str,
    frame: np.ndarray,
    num_iterations: int = 10,
    min_size: int = None,
    max_size: int = None
) -> Dict[str, float]:
    """Benchmark object detector.

    Args:
        model: Model name (mobilenet, resnet50, retinanet)
        device: Device to run on (cpu, mps, cuda)
        frame: Frame to process
        num_iterations: Number of iterations
        min_size: Minimum image size (prevents internal resize)
        max_size: Maximum image size (prevents internal resize)

    Returns:
        Timing statistics
    """
    try:
        detector = ObjectDetector(
            model=model,
            device=device,
            conf_threshold=0.5,
            min_size=min_size,
            max_size=max_size
        )
        return benchmark_detector(detector, frame, num_iterations)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def benchmark_keypoint_detector(
    device: str,
    frame: np.ndarray,
    num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark keypoint detector.

    Args:
        device: Device to run on
        frame: Frame to process
        num_iterations: Number of iterations

    Returns:
        Timing statistics
    """
    try:
        detector = KeypointDetector(device=device, conf_threshold=0.5)
        return benchmark_detector(detector, frame, num_iterations)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def benchmark_reid_detector(
    model: str,
    device: str,
    frame: np.ndarray,
    num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark ReID detector (on crops).

    Note: ReID works on cropped regions, so we create synthetic crops.

    Args:
        model: ReID model name
        device: Device to run on
        frame: Frame to extract crops from
        num_iterations: Number of iterations

    Returns:
        Timing statistics
    """
    try:
        # Create synthetic crops (person-sized regions)
        h, w = frame.shape[:2]
        crop_h, crop_w = min(h, 256), min(w, 128)
        crops = [frame[:crop_h, :crop_w] for _ in range(4)]

        extractor = ReIDDetector(model=model, device=device)

        # Warmup
        for _ in range(3):
            for crop in crops:
                _ = extractor.extract(crop)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            for crop in crops:
                _ = extractor.extract(crop)
            elapsed = time.time() - start
            times.append(elapsed * 1000)

        times_clean = times[1:]

        return {
            'mean_ms': np.mean(times_clean) / len(crops),  # Per crop
            'std_ms': np.std(times_clean) / len(crops),
            'min_ms': np.min(times_clean) / len(crops),
            'max_ms': np.max(times_clean) / len(crops),
            'fps': 1000 / (np.mean(times_clean) / len(crops)),
            'detections': len(crops),
        }
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

def run_comprehensive_benchmark(
    video_path: str,
    devices: List[str] = None,
    resolutions: List[Tuple[int, int, str]] = None,
    num_iterations: int = 10,
    output_file: str = None
):
    """Run comprehensive benchmark across all configurations.

    Args:
        video_path: Path to test video
        devices: List of devices to test (None = all available)
        resolutions: List of (width, height, name) tuples
        num_iterations: Number of iterations per test
        output_file: JSON file to save results (None = don't save)
    """
    # Setup
    if devices is None:
        devices = get_available_devices()

    if resolutions is None:
        resolutions = RESOLUTIONS

    print("="*80)
    print("COMPREHENSIVE DETECTOR BENCHMARK")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Devices: {', '.join(devices)}")
    print(f"Iterations: {num_iterations}")
    print(f"Resolutions: {', '.join([r[2] for r in resolutions])}")
    print("="*80)

    # Load video
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()

    if not ret:
        print("✗ Failed to read video!")
        return

    orig_h, orig_w = original_frame.shape[:2]
    print(f"Original resolution: {orig_w}x{orig_h}\n")

    # Storage for all results
    all_results = {
        'video': video_path,
        'original_resolution': (orig_w, orig_h),
        'devices': devices,
        'iterations': num_iterations,
        'results': {}
    }

    # ========================================================================
    # OBJECT DETECTION BENCHMARKS
    # ========================================================================

    print("\n" + "="*80)
    print("OBJECT DETECTION BENCHMARKS")
    print("="*80)

    for res_w, res_h, res_name in resolutions:
        print(f"\n--- Resolution: {res_name} ({res_w}x{res_h}) ---")

        # Resize frame
        frame = cv2.resize(original_frame, (res_w, res_h))

        for model in OBJECT_MODELS:
            print(f"\nModel: {model}")

            for device in devices:
                print(f"  Device: {device}...", end=" ", flush=True)

                # Disable internal resize since we've already resized
                result = benchmark_object_detector(
                    model=model,
                    device=device,
                    frame=frame,
                    num_iterations=num_iterations,
                    min_size=max(res_w, res_h),
                    max_size=max(res_w, res_h)
                )

                if result:
                    print(f"✓ {result['mean_ms']:.1f}ms ({result['fps']:.1f} FPS, "
                          f"{result['detections']:.0f} detections)")

                    # Store result
                    key = f"object_{model}_{device}_{res_name}"
                    all_results['results'][key] = result
                else:
                    print("✗ Failed")

    # ========================================================================
    # KEYPOINT DETECTION BENCHMARKS
    # ========================================================================

    print("\n" + "="*80)
    print("KEYPOINT DETECTION BENCHMARKS")
    print("="*80)

    for res_w, res_h, res_name in resolutions:
        print(f"\n--- Resolution: {res_name} ({res_w}x{res_h}) ---")

        frame = cv2.resize(original_frame, (res_w, res_h))

        for device in devices:
            print(f"  Device: {device}...", end=" ", flush=True)

            result = benchmark_keypoint_detector(
                device=device,
                frame=frame,
                num_iterations=num_iterations
            )

            if result:
                print(f"✓ {result['mean_ms']:.1f}ms ({result['fps']:.1f} FPS, "
                      f"{result['detections']:.0f} people)")

                key = f"keypoint_resnet50_{device}_{res_name}"
                all_results['results'][key] = result
            else:
                print("✗ Failed")

    # ========================================================================
    # REID BENCHMARKS (resolution-independent, works on crops)
    # ========================================================================

    print("\n" + "="*80)
    print("REID FEATURE EXTRACTION BENCHMARKS")
    print("="*80)
    print("(Measured on 128x256 person crops)")

    frame = cv2.resize(original_frame, (640, 360))  # Use 640p for crops

    for model in REID_MODELS:
        print(f"\nModel: {model}")

        for device in devices:
            print(f"  Device: {device}...", end=" ", flush=True)

            result = benchmark_reid_detector(
                model=model,
                device=device,
                frame=frame,
                num_iterations=num_iterations
            )

            if result:
                print(f"✓ {result['mean_ms']:.1f}ms ({result['fps']:.1f} FPS per crop)")

                key = f"reid_{model}_{device}"
                all_results['results'][key] = result
            else:
                print("✗ Failed")

    # ========================================================================
    # SUMMARY TABLES
    # ========================================================================

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # Object Detection Summary (640p)
    print("\n--- Object Detection @ 640p ---")
    obj_table = []
    for model in OBJECT_MODELS:
        row = [model]
        for device in devices:
            key = f"object_{model}_{device}_640p"
            if key in all_results['results']:
                fps = all_results['results'][key]['fps']
                row.append(f"{fps:.1f}")
            else:
                row.append("N/A")
        obj_table.append(row)

    print(tabulate(obj_table, headers=['Model'] + devices, tablefmt='grid'))

    # Keypoint Detection Summary
    print("\n--- Keypoint Detection @ 640p ---")
    kpt_table = [['resnet50'] + [
        f"{all_results['results'].get(f'keypoint_resnet50_{d}_640p', {}).get('fps', 0):.1f}"
        for d in devices
    ]]
    print(tabulate(kpt_table, headers=['Model'] + devices, tablefmt='grid'))

    # ReID Summary
    print("\n--- ReID Feature Extraction ---")
    reid_table = []
    for model in REID_MODELS:
        row = [model]
        for device in devices:
            key = f"reid_{model}_{device}"
            if key in all_results['results']:
                fps = all_results['results'][key]['fps']
                row.append(f"{fps:.1f}")
            else:
                row.append("N/A")
        reid_table.append(row)

    print(tabulate(reid_table, headers=['Model'] + devices, tablefmt='grid'))

    # Resolution Scaling (MobileNet example)
    if 'mps' in devices or 'cuda' in devices:
        device = 'mps' if 'mps' in devices else 'cuda'
    else:
        device = 'cpu'

    print(f"\n--- Resolution Scaling (mobilenet on {device}) ---")
    res_table = []
    for _, _, res_name in resolutions:
        key = f"object_mobilenet_{device}_{res_name}"
        if key in all_results['results']:
            result = all_results['results'][key]
            res_table.append([
                res_name,
                f"{result['mean_ms']:.1f}",
                f"{result['fps']:.1f}"
            ])

    print(tabulate(res_table, headers=['Resolution', 'Time (ms)', 'FPS'], tablefmt='grid'))

    # Save results to JSON
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find best configurations
    best_speed = None
    best_accuracy = None

    for key, result in all_results['results'].items():
        if key.startswith('object_') and '640p' in key:
            if best_speed is None or result['fps'] > all_results['results'][best_speed]['fps']:
                best_speed = key

            if 'resnet50' in key and (best_accuracy is None or
                                      result['fps'] > all_results['results'].get(best_accuracy, {}).get('fps', 0)):
                best_accuracy = key

    if best_speed:
        result = all_results['results'][best_speed]
        print(f"\n✓ Fastest Config: {best_speed}")
        print(f"  {result['fps']:.1f} FPS ({result['mean_ms']:.1f}ms)")

    if best_accuracy and best_accuracy != best_speed:
        result = all_results['results'][best_accuracy]
        print(f"\n✓ Best Accuracy: {best_accuracy}")
        print(f"  {result['fps']:.1f} FPS ({result['mean_ms']:.1f}ms)")

    print("\n" + "="*80)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive detector benchmark')
    parser.add_argument('--video', default='data/TownCent.mp4',
                       help='Path to test video')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'mps', 'cuda'],
                       help='Devices to test (default: all available)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per test (default: 10)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer iterations')
    parser.add_argument('--output', default='benchmark_results.json',
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.iterations = 5
        resolutions = [(640, 360, "640p")]
    else:
        resolutions = RESOLUTIONS

    # Run benchmark
    run_comprehensive_benchmark(
        video_path=args.video,
        devices=args.devices,
        resolutions=resolutions,
        num_iterations=args.iterations,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
