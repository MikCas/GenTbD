"""Benchmark different preprocessing approaches.

This module tests various preprocessing techniques to identify the fastest
approach for converting images to tensors.
"""

import time
import numpy as np
import torch
import cv2
from typing import Dict, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class PreprocessingBenchmark:
    """Test different preprocessing approaches."""

    @staticmethod
    def current_approach(image: np.ndarray) -> torch.Tensor:
        """Current approach: cv2.cvtColor + division normalization."""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(image_rgb).float() / 255.0

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def numpy_slice_bgr_to_rgb(image: np.ndarray) -> torch.Tensor:
        """Alternative: Use numpy slicing instead of cv2.cvtColor."""
        # BGR to RGB using numpy slicing
        image_rgb = image[:, :, ::-1]

        # Normalize to [0, 1]
        tensor = torch.from_numpy(image_rgb).float() / 255.0

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def numpy_slice_with_copy(image: np.ndarray) -> torch.Tensor:
        """Alternative: Numpy slicing with explicit copy (might be faster)."""
        # BGR to RGB using numpy slicing with copy
        image_rgb = image[:, :, ::-1].copy()

        # Normalize to [0, 1]
        tensor = torch.from_numpy(image_rgb).float() / 255.0

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def multiply_normalization(image: np.ndarray) -> torch.Tensor:
        """Alternative: Multiplication instead of division for normalization."""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] using multiplication
        tensor = torch.from_numpy(image_rgb).float() * (1.0 / 255.0)

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def direct_tensor_creation(image: np.ndarray) -> torch.Tensor:
        """Alternative: Create tensor first, then process."""
        # Convert to tensor directly from BGR
        tensor = torch.from_numpy(image)

        # Flip channels (BGR to RGB) in tensor space
        tensor = torch.flip(tensor, dims=[2])

        # Normalize and convert to float
        tensor = tensor.float() / 255.0

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def contiguous_tensor(image: np.ndarray) -> torch.Tensor:
        """Alternative: Ensure contiguous memory layout."""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(image_rgb).float() / 255.0

        # HWC to CHW with contiguous()
        tensor = tensor.permute(2, 0, 1).contiguous()

        return tensor


def benchmark_method(method: Callable, image: np.ndarray, device: str,
                     iterations: int = 1000, warmup: int = 10) -> Dict:
    """Benchmark a preprocessing method.

    Args:
        method: Preprocessing function to test
        image: Input image
        device: Target device ('cpu', 'mps', or 'cuda')
        iterations: Number of test iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        tensor = method(image)
        if device != 'cpu':
            tensor = tensor.to(device)

    # Measure preprocessing only (no device transfer)
    times_preprocess = []
    for _ in range(iterations):
        start = time.perf_counter()
        tensor = method(image)
        elapsed = time.perf_counter() - start
        times_preprocess.append(elapsed)

    # Measure preprocessing + device transfer
    times_with_transfer = []
    for _ in range(iterations):
        start = time.perf_counter()
        tensor = method(image).to(device)
        if device == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_with_transfer.append(elapsed)

    times_preprocess = np.array(times_preprocess) * 1000  # Convert to ms
    times_with_transfer = np.array(times_with_transfer) * 1000

    transfer_time = times_with_transfer - times_preprocess

    return {
        'preprocess_mean_ms': np.mean(times_preprocess),
        'preprocess_std_ms': np.std(times_preprocess),
        'transfer_mean_ms': np.mean(transfer_time),
        'total_mean_ms': np.mean(times_with_transfer),
        'total_std_ms': np.std(times_with_transfer),
    }


def main():
    """Run preprocessing benchmarks."""

    print("\n" + "="*70)
    print("PREPROCESSING BENCHMARK")
    print("="*70 + "\n")

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print(f"Available devices: {devices}\n")

    # Create test images at different resolutions
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]

    # Preprocessing methods to test
    methods = {
        'Current (cv2.cvtColor)': PreprocessingBenchmark.current_approach,
        'Numpy slice': PreprocessingBenchmark.numpy_slice_bgr_to_rgb,
        'Numpy slice + copy': PreprocessingBenchmark.numpy_slice_with_copy,
        'Multiply normalize': PreprocessingBenchmark.multiply_normalization,
        'Direct tensor': PreprocessingBenchmark.direct_tensor_creation,
        'Contiguous tensor': PreprocessingBenchmark.contiguous_tensor,
    }

    # Test each resolution
    for resolution in resolutions:
        w, h = resolution
        print(f"\n{'='*70}")
        print(f"Resolution: {w}x{h}")
        print(f"{'='*70}\n")

        # Create test image
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Test each device
        for device in devices:
            print(f"\nDevice: {device}")
            print("-" * 70)
            print(f"{'Method':<25} {'Preproc (ms)':<15} {'Transfer (ms)':<15} {'Total (ms)':<15}")
            print("-" * 70)

            results = {}
            for name, method in methods.items():
                try:
                    stats = benchmark_method(
                        method=method,
                        image=image,
                        device=device,
                        iterations=200,
                        warmup=10
                    )
                    results[name] = stats

                    print(f"{name:<25} "
                          f"{stats['preprocess_mean_ms']:<15.3f} "
                          f"{stats['transfer_mean_ms']:<15.3f} "
                          f"{stats['total_mean_ms']:<15.3f}")

                except Exception as e:
                    print(f"{name:<25} ERROR: {e}")

            # Find fastest method
            if results:
                fastest = min(results.items(), key=lambda x: x[1]['total_mean_ms'])
                print("-" * 70)
                print(f"✓ Fastest: {fastest[0]} ({fastest[1]['total_mean_ms']:.3f} ms)")

                # Calculate speedup vs current
                if 'Current (cv2.cvtColor)' in results:
                    current_time = results['Current (cv2.cvtColor)']['total_mean_ms']
                    speedup = current_time / fastest[1]['total_mean_ms']
                    improvement_pct = (current_time - fastest[1]['total_mean_ms']) / current_time * 100

                    if speedup > 1.1:
                        print(f"⚡ Speedup vs current: {speedup:.2f}x ({improvement_pct:.1f}% faster)")
                    else:
                        print("ℹ️  Current approach is already optimal")

    # Additional analysis: Component breakdown
    print("\n\n" + "="*70)
    print("COMPONENT BREAKDOWN (640x480 image)")
    print("="*70 + "\n")

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test individual operations
    operations = {
        'cv2.cvtColor (BGR→RGB)': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'Numpy slice (BGR→RGB)': lambda img: img[:, :, ::-1],
        'Numpy slice + copy': lambda img: img[:, :, ::-1].copy(),
        'torch.from_numpy': lambda img: torch.from_numpy(img),
        'Tensor division /255': lambda t: t.float() / 255.0,
        'Tensor multiply *inv': lambda t: t.float() * (1.0/255.0),
        'Tensor permute': lambda t: t.permute(2, 0, 1),
        'Tensor contiguous': lambda t: t.contiguous(),
    }

    print(f"{'Operation':<30} {'Mean (ms)':<15} {'Std (ms)':<15}")
    print("-" * 70)

    for name, op in operations.items():
        times = []

        # Determine what to pass to the operation
        if 'Tensor' in name:
            # Create a tensor for tensor operations
            test_input = torch.from_numpy(image)
            if 'division' in name or 'multiply' in name:
                test_input = test_input
            elif 'permute' in name or 'contiguous' in name:
                test_input = torch.from_numpy(image).float().permute(2, 0, 1)
        else:
            # Use numpy array for image operations
            test_input = image

        # Warmup
        for _ in range(10):
            _ = op(test_input)

        # Measure
        for _ in range(500):
            start = time.perf_counter()
            _ = op(test_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)

        print(f"{name:<30} {mean_time:<15.3f} {std_time:<15.3f}")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
