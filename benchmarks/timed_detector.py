"""Timed detector variant for measuring each pipeline stage.

This module provides a detector wrapper that measures the time spent in each
stage of the detection pipeline: preprocessing, inference, and postprocessing.
"""

import time
import numpy as np
import torch
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detecting.detectors.object_detector import ObjectDetector
from src.detecting.detection import Detection


class TimedDetector:
    """Detector wrapper that provides detailed timing for each pipeline stage."""

    def __init__(self, model='mobilenet', device='cpu', conf_threshold=0.5, classes=None):
        """Initialize timed detector.

        Args:
            model: Model name - 'resnet50', 'mobilenet', or 'retinanet'
            device: 'cpu', 'mps', or 'cuda'
            conf_threshold: Minimum confidence for detections
            classes: List of class IDs to filter (None = all classes)
        """
        self.detector = ObjectDetector(
            model=model,
            device=device,
            conf_threshold=conf_threshold,
            classes=classes
        )
        self.device = device

    def _synchronize(self):
        """Synchronize device operations before timing measurement."""
        if self.device == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def detect_timed(self, image: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        """Run detection with detailed timing for each stage.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            Tuple of (detections, timings) where timings is a dict with:
                - 'preprocess': Preprocessing time in seconds
                - 'inference': Inference time in seconds
                - 'postprocess': Postprocessing time in seconds
                - 'total': Total time in seconds
        """
        timings = {}

        # Stage 1: Preprocessing
        start = time.perf_counter()
        input_tensor = self.detector.preprocess(image)
        timings['preprocess'] = time.perf_counter() - start

        # Stage 2: Inference (with proper synchronization!)
        self._synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = self.detector.inference(input_tensor)

        # CRITICAL: Synchronize before measuring elapsed time
        self._synchronize()
        timings['inference'] = time.perf_counter() - start

        # Stage 3: Postprocessing
        start = time.perf_counter()
        detections = self.detector.postprocess(output, image.shape[:2])
        timings['postprocess'] = time.perf_counter() - start

        # Total time
        timings['total'] = sum(timings.values())

        return detections, timings


def benchmark_detector(model='mobilenet', device='cpu', resolution=(640, 480),
                       num_frames=100, warmup=5):
    """Benchmark a detector configuration.

    Args:
        model: Model name - 'resnet50', 'mobilenet', or 'retinanet'
        device: 'cpu', 'mps', or 'cuda'
        resolution: Tuple of (width, height)
        num_frames: Number of frames to test
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics for each stage
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model} on {device} at {resolution[0]}x{resolution[1]}")
    print(f"{'='*70}\n")

    # Create detector
    detector = TimedDetector(model=model, device=device, conf_threshold=0.5)

    # Create dummy frames
    h, w = resolution[1], resolution[0]
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
              for _ in range(num_frames)]

    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for i in range(warmup):
        detector.detect_timed(frames[0])

    # Measure
    print(f"Measuring {num_frames} frames...\n")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    for i, frame in enumerate(frames):
        detections, timings = detector.detect_timed(frame)

        preprocess_times.append(timings['preprocess'])
        inference_times.append(timings['inference'])
        postprocess_times.append(timings['postprocess'])
        total_times.append(timings['total'])

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")

    # Calculate statistics
    def calc_stats(times, name):
        times_ms = np.array(times) * 1000  # Convert to ms
        return {
            'stage': name,
            'mean_ms': np.mean(times_ms),
            'std_ms': np.std(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'p50_ms': np.percentile(times_ms, 50),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'percentage': np.mean(times) / np.mean(total_times) * 100
        }

    results = {
        'preprocess': calc_stats(preprocess_times, 'Preprocessing'),
        'inference': calc_stats(inference_times, 'Inference'),
        'postprocess': calc_stats(postprocess_times, 'Postprocessing'),
        'total': calc_stats(total_times, 'Total')
    }

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")

    print(f"Configuration: {model} on {device} at {resolution[0]}x{resolution[1]}\n")

    # Summary table
    print(f"{'Stage':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'% Total':<10}")
    print("-" * 70)

    for stage in ['preprocess', 'inference', 'postprocess', 'total']:
        stats = results[stage]
        print(f"{stats['stage']:<20} "
              f"{stats['mean_ms']:<12.2f} "
              f"{stats['std_ms']:<12.2f} "
              f"{stats['p95_ms']:<12.2f} "
              f"{stats['percentage']:<10.1f}")

    print("\n" + "="*70)
    print(f"Average FPS: {1000 / results['total']['mean_ms']:.2f}")
    print(f"P50 FPS: {1000 / results['total']['p50_ms']:.2f}")
    print(f"P95 FPS: {1000 / results['total']['p95_ms']:.2f}")
    print("="*70 + "\n")

    # Bottleneck analysis
    print("BOTTLENECK ANALYSIS")
    print("-" * 70)

    # Find slowest stage
    stages = ['preprocess', 'inference', 'postprocess']
    slowest_stage = max(stages, key=lambda s: results[s]['percentage'])

    print(f"Primary bottleneck: {results[slowest_stage]['stage']}")
    print(f"  - Takes {results[slowest_stage]['percentage']:.1f}% of total time")
    print(f"  - Average: {results[slowest_stage]['mean_ms']:.2f} ms")
    print(f"  - P95: {results[slowest_stage]['p95_ms']:.2f} ms")

    # Check for high variance
    for stage in stages:
        cv = results[stage]['std_ms'] / results[stage]['mean_ms']  # Coefficient of variation
        if cv > 0.2:
            print(f"\n⚠️  {results[stage]['stage']} shows high variance (CV: {cv:.2%})")
            print(f"   This suggests inconsistent performance")

    print("="*70 + "\n")

    return results


def main():
    """Run comprehensive timing benchmarks."""

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print("Available devices:", devices)

    # Test configurations
    configs = [
        ('mobilenet', 'cpu', (640, 480)),
    ]

    # Add MPS if available
    if 'mps' in devices:
        configs.extend([
            ('mobilenet', 'mps', (640, 480)),
            ('resnet50', 'mps', (640, 480)),
        ])

    # Run benchmarks
    all_results = []
    for model, device, resolution in configs:
        try:
            results = benchmark_detector(
                model=model,
                device=device,
                resolution=resolution,
                num_frames=50,  # Reduced for faster testing
                warmup=5
            )
            all_results.append({
                'config': f"{model}-{device}-{resolution}",
                'results': results
            })
        except Exception as e:
            print(f"\n❌ Error benchmarking {model} on {device}: {e}\n")
            import traceback
            traceback.print_exc()

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("CONFIGURATION COMPARISON")
        print("="*70 + "\n")

        print(f"{'Configuration':<30} {'Total (ms)':<15} {'FPS':<10} {'Bottleneck':<20}")
        print("-" * 70)

        for result in all_results:
            config = result['config']
            total_ms = result['results']['total']['mean_ms']
            fps = 1000 / total_ms

            # Find bottleneck
            stages = ['preprocess', 'inference', 'postprocess']
            bottleneck = max(stages, key=lambda s: result['results'][s]['percentage'])

            print(f"{config:<30} {total_ms:<15.2f} {fps:<10.2f} {bottleneck:<20}")

        print("="*70 + "\n")


if __name__ == '__main__':
    main()
