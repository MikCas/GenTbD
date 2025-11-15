"""Benchmark postprocessing operations.

This module tests the performance of different postprocessing stages:
- CPU transfer (GPU/MPS → CPU)
- Confidence filtering
- Class filtering
- Detection object creation
"""

import time
import numpy as np
import torch
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detecting.detection import Detection
from src.core.properties import BoundingBox


class PostprocessingBenchmark:
    """Test postprocessing bottlenecks."""

    @staticmethod
    def create_mock_output(num_detections: int, device: str = 'cpu') -> Dict:
        """Create mock model output for testing.

        Args:
            num_detections: Number of detections to create
            device: Device to create tensors on

        Returns:
            Mock output dictionary like FasterRCNN
        """
        return {
            'boxes': torch.rand(num_detections, 4, device=device) * 640,  # Random boxes
            'labels': torch.randint(1, 81, (num_detections,), device=device),  # COCO classes
            'scores': torch.rand(num_detections, device=device)  # Random scores
        }

    @staticmethod
    def benchmark_cpu_transfer(output: Dict, iterations: int = 500) -> Dict:
        """Measure time to transfer tensors from GPU/MPS to CPU.

        Args:
            output: Model output with tensors on device
            iterations: Number of test iterations

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            boxes_cpu = output['boxes'].cpu()
            labels_cpu = output['labels'].cpu()
            scores_cpu = output['scores'].cpu()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        times = np.array(times)

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times)
        }

    @staticmethod
    def benchmark_filtering(boxes: torch.Tensor, labels: torch.Tensor,
                           scores: torch.Tensor, conf_threshold: float = 0.5,
                           class_filter: list = None, iterations: int = 500) -> Dict:
        """Measure time for confidence and class filtering.

        Args:
            boxes: Bounding boxes tensor
            labels: Labels tensor
            scores: Scores tensor
            conf_threshold: Confidence threshold
            class_filter: List of class IDs to filter (None = no filtering)
            iterations: Number of test iterations

        Returns:
            Dictionary with timing statistics
        """
        times_conf = []
        times_class = []
        times_total = []

        for _ in range(iterations):
            # Confidence filtering
            start = time.perf_counter()
            mask = scores >= conf_threshold
            boxes_filtered = boxes[mask]
            labels_filtered = labels[mask]
            scores_filtered = scores[mask]
            elapsed_conf = time.perf_counter() - start
            times_conf.append(elapsed_conf * 1000)

            # Class filtering (if specified)
            if class_filter is not None:
                start = time.perf_counter()
                class_mask = torch.zeros(len(labels_filtered), dtype=torch.bool)
                for class_id in class_filter:
                    class_mask |= (labels_filtered == class_id)
                boxes_final = boxes_filtered[class_mask]
                labels_final = labels_filtered[class_mask]
                scores_final = scores_filtered[class_mask]
                elapsed_class = time.perf_counter() - start
                times_class.append(elapsed_class * 1000)
                times_total.append((elapsed_conf + elapsed_class) * 1000)
            else:
                times_total.append(elapsed_conf * 1000)

        return {
            'conf_filtering_ms': np.mean(times_conf),
            'class_filtering_ms': np.mean(times_class) if times_class else 0.0,
            'total_filtering_ms': np.mean(times_total),
            'std_ms': np.std(times_total)
        }

    @staticmethod
    def benchmark_detection_creation(boxes: torch.Tensor, labels: torch.Tensor,
                                     scores: torch.Tensor, iterations: int = 200) -> Dict:
        """Measure time to create Detection objects.

        Args:
            boxes: Bounding boxes tensor (CPU)
            labels: Labels tensor (CPU)
            scores: Scores tensor (CPU)
            iterations: Number of test iterations

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(iterations):
            start = time.perf_counter()

            detections = []
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                detection = Detection({
                    'bbox': BoundingBox(x1, y1, x2, y2),
                    'class_id': int(label),
                    'confidence': float(score)
                })
                detections.append(detection)

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        times = np.array(times)

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'per_detection_us': np.mean(times) * 1000 / len(boxes),  # Microseconds per detection
        }


def main():
    """Run comprehensive postprocessing benchmarks."""

    print("\n" + "="*70)
    print("POSTPROCESSING BENCHMARK")
    print("="*70 + "\n")

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print(f"Available devices: {devices}\n")

    # Test different numbers of detections
    detection_counts = [10, 50, 100, 200]

    # Test 1: CPU Transfer Time
    if len(devices) > 1:
        print("="*70)
        print("TEST 1: CPU Transfer Time (GPU/MPS → CPU)")
        print("="*70 + "\n")

        gpu_device = 'mps' if 'mps' in devices else 'cuda'

        print(f"{'Num Detections':<20} {'Mean (ms)':<15} {'Std (ms)':<15} {'Per Detection (μs)':<20}")
        print("-" * 70)

        for num_det in detection_counts:
            output = PostprocessingBenchmark.create_mock_output(num_det, device=gpu_device)

            stats = PostprocessingBenchmark.benchmark_cpu_transfer(
                output=output,
                iterations=500
            )

            per_det_us = stats['mean_ms'] * 1000 / num_det

            print(f"{num_det:<20} "
                  f"{stats['mean_ms']:<15.4f} "
                  f"{stats['std_ms']:<15.4f} "
                  f"{per_det_us:<20.2f}")

        print("\nℹ️  Note: High transfer times indicate GPU↔CPU communication overhead\n")

    # Test 2: Filtering Performance
    print("="*70)
    print("TEST 2: Filtering Performance")
    print("="*70 + "\n")

    print(f"{'Num Detections':<20} {'Conf Filter (ms)':<20} {'Class Filter (ms)':<20} {'Total (ms)':<15}")
    print("-" * 70)

    for num_det in detection_counts:
        # Create mock data on CPU
        output = PostprocessingBenchmark.create_mock_output(num_det, device='cpu')

        # Test with class filtering
        stats = PostprocessingBenchmark.benchmark_filtering(
            boxes=output['boxes'],
            labels=output['labels'],
            scores=output['scores'],
            conf_threshold=0.5,
            class_filter=[0, 1, 2, 5, 7],  # person, bicycle, car, bus, truck
            iterations=500
        )

        print(f"{num_det:<20} "
              f"{stats['conf_filtering_ms']:<20.4f} "
              f"{stats['class_filtering_ms']:<20.4f} "
              f"{stats['total_filtering_ms']:<15.4f}")

    # Test 3: Detection Object Creation
    print("\n" + "="*70)
    print("TEST 3: Detection Object Creation")
    print("="*70 + "\n")

    print(f"{'Num Detections':<20} {'Total Time (ms)':<20} {'Per Detection (μs)':<20} {'FPS Limit':<15}")
    print("-" * 70)

    for num_det in detection_counts:
        # Create filtered mock data (simulate typical ~10-50 detections after filtering)
        if num_det > 50:
            actual_num = 50
        else:
            actual_num = num_det

        output = PostprocessingBenchmark.create_mock_output(actual_num, device='cpu')

        stats = PostprocessingBenchmark.benchmark_detection_creation(
            boxes=output['boxes'],
            labels=output['labels'],
            scores=output['scores'],
            iterations=200
        )

        # Calculate FPS limit imposed by this operation
        fps_limit = 1000 / stats['mean_ms'] if stats['mean_ms'] > 0 else float('inf')

        print(f"{actual_num:<20} "
              f"{stats['mean_ms']:<20.4f} "
              f"{stats['per_detection_us']:<20.2f} "
              f"{fps_limit:<15.2f}")

    # Test 4: End-to-End Postprocessing
    print("\n" + "="*70)
    print("TEST 4: Complete Postprocessing Pipeline")
    print("="*70 + "\n")

    for device in devices:
        if device == 'cpu':
            continue  # Skip CPU-only test

        print(f"Device: {device}")
        print("-" * 70)
        print(f"{'Stage':<25} {'Time (ms)':<15} {'Percentage':<15}")
        print("-" * 70)

        # Create mock output on device
        num_det = 100
        output = PostprocessingBenchmark.create_mock_output(num_det, device=device)

        # Stage 1: CPU Transfer
        transfer_times = []
        for _ in range(200):
            start = time.perf_counter()
            boxes_cpu = output['boxes'].cpu()
            labels_cpu = output['labels'].cpu()
            scores_cpu = output['scores'].cpu()
            elapsed = time.perf_counter() - start
            transfer_times.append(elapsed * 1000)
        transfer_time = np.mean(transfer_times)

        # Stage 2: Filtering
        filter_stats = PostprocessingBenchmark.benchmark_filtering(
            boxes=boxes_cpu,
            labels=labels_cpu,
            scores=scores_cpu,
            conf_threshold=0.5,
            class_filter=[0, 1, 2],
            iterations=200
        )
        filter_time = filter_stats['total_filtering_ms']

        # Stage 3: Detection creation
        # Simulate ~20 detections after filtering
        filtered_output = PostprocessingBenchmark.create_mock_output(20, device='cpu')
        creation_stats = PostprocessingBenchmark.benchmark_detection_creation(
            boxes=filtered_output['boxes'],
            labels=filtered_output['labels'],
            scores=filtered_output['scores'],
            iterations=200
        )
        creation_time = creation_stats['mean_ms']

        # Total
        total_time = transfer_time + filter_time + creation_time

        # Print breakdown
        print(f"{'CPU Transfer':<25} {transfer_time:<15.4f} {transfer_time/total_time*100:<15.1f}")
        print(f"{'Filtering':<25} {filter_time:<15.4f} {filter_time/total_time*100:<15.1f}")
        print(f"{'Object Creation':<25} {creation_time:<15.4f} {creation_time/total_time*100:<15.1f}")
        print("-" * 70)
        print(f"{'TOTAL':<25} {total_time:<15.4f} {100.0:<15.1f}")

        # FPS impact
        fps_limit = 1000 / total_time
        print(f"\nPostprocessing FPS limit: {fps_limit:.2f}")
        print("="*70 + "\n")

    # Test 5: Optimization suggestions
    print("="*70)
    print("OPTIMIZATION ANALYSIS")
    print("="*70 + "\n")

    # Compare different filtering approaches
    num_det = 100
    output = PostprocessingBenchmark.create_mock_output(num_det, device='cpu')

    print("Comparing filtering implementations:\n")

    # Current approach (loop over class IDs)
    times_current = []
    for _ in range(500):
        class_filter = [0, 1, 2, 5, 7]
        start = time.perf_counter()
        class_mask = torch.zeros(len(output['labels']), dtype=torch.bool)
        for class_id in class_filter:
            class_mask |= (output['labels'] == class_id)
        result = output['labels'][class_mask]
        elapsed = time.perf_counter() - start
        times_current.append(elapsed * 1000)

    # Alternative: Use isin (if available)
    times_isin = []
    for _ in range(500):
        class_filter = torch.tensor([0, 1, 2, 5, 7])
        start = time.perf_counter()
        # Create mask using broadcasting
        class_mask = (output['labels'].unsqueeze(1) == class_filter).any(dim=1)
        result = output['labels'][class_mask]
        elapsed = time.perf_counter() - start
        times_isin.append(elapsed * 1000)

    print(f"Current approach (loop): {np.mean(times_current):.4f} ms")
    print(f"Vectorized approach:     {np.mean(times_isin):.4f} ms")
    speedup = np.mean(times_current) / np.mean(times_isin)
    print(f"Speedup: {speedup:.2f}x\n")

    if speedup > 1.2:
        print("✓ Recommendation: Use vectorized filtering for better performance")
    else:
        print("ℹ️  Current approach is acceptable")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
