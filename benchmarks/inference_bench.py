"""Benchmark inference under different conditions.

This module tests model inference performance across different scenarios:
- Single-frame vs batched inference
- Different resolutions
- Different models
- CPU vs GPU/MPS
"""

import time
import numpy as np
import torch
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detecting.detectors.object_detector import ObjectDetector


class InferenceBenchmark:
    """Test inference under different conditions."""

    @staticmethod
    def _synchronize(device: str):
        """Synchronize device operations."""
        if device == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def benchmark_single_frame(model, device: str, frame_tensor: torch.Tensor,
                               iterations: int = 100, warmup: int = 5) -> Dict:
        """Benchmark single-frame inference.

        Args:
            model: PyTorch model
            device: Device name
            frame_tensor: Preprocessed frame tensor (C, H, W)
            iterations: Number of test iterations
            warmup: Number of warmup iterations

        Returns:
            Dictionary with timing statistics
        """
        # Add batch dimension and move to device
        input_batch = frame_tensor.unsqueeze(0).to(device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_batch)
            InferenceBenchmark._synchronize(device)

        # Measure
        times = []
        for _ in range(iterations):
            InferenceBenchmark._synchronize(device)
            start = time.perf_counter()

            with torch.no_grad():
                output = model(input_batch)

            InferenceBenchmark._synchronize(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }

    @staticmethod
    def benchmark_batched(model, device: str, frame_tensors: List[torch.Tensor],
                         batch_sizes: List[int], warmup: int = 5) -> Dict:
        """Benchmark batched inference.

        Args:
            model: PyTorch model
            device: Device name
            frame_tensors: List of preprocessed frame tensors
            batch_sizes: List of batch sizes to test
            warmup: Number of warmup iterations

        Returns:
            Dictionary mapping batch_size to timing statistics
        """
        results = {}

        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")

            # Create batch
            batch_tensors = frame_tensors[:batch_size]
            input_batch = torch.stack(batch_tensors).to(device)

            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(input_batch)
                InferenceBenchmark._synchronize(device)

            # Measure
            times = []
            for _ in range(20):  # Fewer iterations for batched
                InferenceBenchmark._synchronize(device)
                start = time.perf_counter()

                with torch.no_grad():
                    outputs = model(input_batch)

                InferenceBenchmark._synchronize(device)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

            # Calculate per-frame metrics
            times = np.array(times)
            per_frame_times = times / batch_size

            results[batch_size] = {
                'batch_time_ms': np.mean(times),
                'per_frame_ms': np.mean(per_frame_times),
                'fps': 1000 / np.mean(per_frame_times),
                'throughput': batch_size * (1000 / np.mean(times))
            }

        return results

    @staticmethod
    def benchmark_resolutions(model_name: str, device: str,
                             resolutions: List[tuple]) -> Dict:
        """Benchmark inference at different resolutions.

        Args:
            model_name: Model name
            device: Device name
            resolutions: List of (width, height) tuples

        Returns:
            Dictionary mapping resolution to timing statistics
        """
        results = {}

        # Load model once
        detector = ObjectDetector(model=model_name, device=device)

        for resolution in resolutions:
            w, h = resolution
            print(f"  Testing {w}x{h}...")

            # Create dummy tensor
            dummy_tensor = torch.randn(3, h, w)

            # Benchmark
            stats = InferenceBenchmark.benchmark_single_frame(
                model=detector.detector.model,
                device=device,
                frame_tensor=dummy_tensor,
                iterations=50,
                warmup=5
            )

            # Calculate pixels processed
            pixels = w * h
            megapixels = pixels / 1e6

            results[resolution] = {
                **stats,
                'megapixels': megapixels,
                'mpixels_per_sec': megapixels * stats['fps']
            }

        return results


def main():
    """Run comprehensive inference benchmarks."""

    print("\n" + "="*70)
    print("INFERENCE BENCHMARK")
    print("="*70 + "\n")

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print(f"Available devices: {devices}\n")

    # Test 1: Single-frame inference comparison
    print("="*70)
    print("TEST 1: Single-Frame Inference")
    print("="*70 + "\n")

    models = ['mobilenet', 'resnet50']
    resolution = (640, 480)

    # Create test frame
    test_frame = torch.randn(3, resolution[1], resolution[0])

    print(f"Resolution: {resolution[0]}x{resolution[1]}\n")

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 70)
        print(f"{'Model':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
        print("-" * 70)

        for model_name in models:
            try:
                detector = ObjectDetector(model=model_name, device=device)

                stats = InferenceBenchmark.benchmark_single_frame(
                    model=detector.detector.model,
                    device=device,
                    frame_tensor=test_frame,
                    iterations=50,
                    warmup=5
                )

                print(f"{model_name:<15} "
                      f"{stats['mean_ms']:<12.2f} "
                      f"{stats['std_ms']:<12.2f} "
                      f"{stats['p95_ms']:<12.2f} "
                      f"{stats['fps']:<10.2f}")

            except Exception as e:
                print(f"{model_name:<15} ERROR: {e}")

    # Test 2: Batched inference (if MPS or CUDA available)
    if len(devices) > 1:
        print("\n\n" + "="*70)
        print("TEST 2: Batched Inference (MobileNet)")
        print("="*70 + "\n")

        # Use GPU device
        gpu_device = 'mps' if 'mps' in devices else 'cuda'
        print(f"Device: {gpu_device}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}\n")

        # Create test frames
        num_frames = 16
        test_frames = [torch.randn(3, resolution[1], resolution[0]) for _ in range(num_frames)]

        detector = ObjectDetector(model='mobilenet', device=gpu_device)

        batch_sizes = [1, 2, 4, 8]
        results = InferenceBenchmark.benchmark_batched(
            model=detector.detector.model,
            device=gpu_device,
            frame_tensors=test_frames,
            batch_sizes=batch_sizes,
            warmup=5
        )

        print(f"\n{'Batch Size':<12} {'Batch Time':<15} {'Per Frame':<15} {'FPS':<12} {'Throughput':<12}")
        print("-" * 70)

        for batch_size, stats in results.items():
            print(f"{batch_size:<12} "
                  f"{stats['batch_time_ms']:<15.2f} "
                  f"{stats['per_frame_ms']:<15.2f} "
                  f"{stats['fps']:<12.2f} "
                  f"{stats['throughput']:<12.2f}")

        # Analyze batching efficiency
        if 1 in results and len(results) > 1:
            single_fps = results[1]['fps']
            best_batch = max(results.items(), key=lambda x: x[1]['throughput'])

            print("-" * 70)
            print(f"Single-frame FPS: {single_fps:.2f}")
            print(f"Best throughput: Batch size {best_batch[0]} → {best_batch[1]['throughput']:.2f} FPS")
            speedup = best_batch[1]['throughput'] / single_fps
            print(f"Batching speedup: {speedup:.2f}x")

    # Test 3: Resolution scaling
    print("\n\n" + "="*70)
    print("TEST 3: Resolution Scaling (MobileNet)")
    print("="*70 + "\n")

    resolutions = [
        (320, 240),
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 70)
        print(f"{'Resolution':<15} {'Megapixels':<12} {'Mean (ms)':<12} {'FPS':<10} {'MP/s':<10}")
        print("-" * 70)

        results = InferenceBenchmark.benchmark_resolutions(
            model_name='mobilenet',
            device=device,
            resolutions=resolutions
        )

        for resolution, stats in results.items():
            res_str = f"{resolution[0]}x{resolution[1]}"
            print(f"{res_str:<15} "
                  f"{stats['megapixels']:<12.2f} "
                  f"{stats['mean_ms']:<12.2f} "
                  f"{stats['fps']:<10.2f} "
                  f"{stats['mpixels_per_sec']:<10.2f}")

        # Calculate scaling factor
        if len(results) >= 2:
            res_list = sorted(results.keys(), key=lambda r: r[0] * r[1])
            small_res = res_list[0]
            large_res = res_list[-1]

            pixel_ratio = (large_res[0] * large_res[1]) / (small_res[0] * small_res[1])
            time_ratio = results[large_res]['mean_ms'] / results[small_res]['mean_ms']

            print("-" * 70)
            print(f"Pixel ratio ({small_res} → {large_res}): {pixel_ratio:.2f}x")
            print(f"Time ratio: {time_ratio:.2f}x")

            if time_ratio < pixel_ratio:
                print(f"✓ Sub-linear scaling (efficient)")
            else:
                print(f"⚠️  Super-linear scaling (resolution-sensitive)")

    # Test 4: Model comparison summary
    print("\n\n" + "="*70)
    print("TEST 4: Model Comparison Summary (640x480)")
    print("="*70 + "\n")

    test_frame = torch.randn(3, 480, 640)

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 70)
        print(f"{'Model':<15} {'FPS':<10} {'ms/frame':<12} {'Speedup':<10}")
        print("-" * 70)

        baseline_fps = None
        for model_name in ['resnet50', 'mobilenet']:
            try:
                detector = ObjectDetector(model=model_name, device=device)

                stats = InferenceBenchmark.benchmark_single_frame(
                    model=detector.detector.model,
                    device=device,
                    frame_tensor=test_frame,
                    iterations=50,
                    warmup=5
                )

                if baseline_fps is None:
                    baseline_fps = stats['fps']
                    speedup_str = "baseline"
                else:
                    speedup = stats['fps'] / baseline_fps
                    speedup_str = f"{speedup:.2f}x"

                print(f"{model_name:<15} "
                      f"{stats['fps']:<10.2f} "
                      f"{stats['mean_ms']:<12.2f} "
                      f"{speedup_str:<10}")

            except Exception as e:
                print(f"{model_name:<15} ERROR: {e}")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
