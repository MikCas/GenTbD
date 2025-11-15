"""End-to-end benchmark comparing all configurations.

This module provides comprehensive comparison of different model, device,
and resolution combinations.
"""

import time
import numpy as np
import torch
import pandas as pd
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detecting.detectors.object_detector import ObjectDetector


class E2EBenchmark:
    """Compare different model + device + resolution combinations."""

    @staticmethod
    def benchmark_configuration(model_name: str, device: str, resolution: tuple,
                               num_frames: int = 100, warmup: int = 5) -> Dict:
        """Benchmark a complete configuration.

        Args:
            model_name: Model name - 'resnet50', 'mobilenet', or 'retinanet'
            device: 'cpu', 'mps', or 'cuda'
            resolution: Tuple of (width, height)
            num_frames: Number of frames to test
            warmup: Number of warmup iterations

        Returns:
            Dictionary with comprehensive timing statistics
        """
        w, h = resolution

        # Create detector
        detector = ObjectDetector(model=model_name, device=device, conf_threshold=0.5)

        # Create dummy frames
        frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                  for _ in range(num_frames)]

        # Warmup
        for i in range(warmup):
            detector.detect(frames[0])

        # Synchronize helper
        def sync():
            if device == 'mps' and torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Measure
        times = []
        detection_counts = []

        for frame in frames:
            sync()
            start = time.perf_counter()
            detections = detector.detect(frame)
            sync()
            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)  # Convert to ms
            detection_counts.append(len(detections))

        times = np.array(times)
        pixels = w * h
        megapixels = pixels / 1e6

        return {
            'config': f"{model_name}-{device}-{w}x{h}",
            'model': model_name,
            'device': device,
            'resolution': f"{w}x{h}",
            'width': w,
            'height': h,
            'megapixels': megapixels,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'mean_fps': 1000 / np.mean(times),
            'p50_fps': 1000 / np.percentile(times, 50),
            'p95_fps': 1000 / np.percentile(times, 95),
            'avg_detections': np.mean(detection_counts),
            'mpixels_per_sec': megapixels * (1000 / np.mean(times))
        }

    @staticmethod
    def run_full_benchmark(configs: List[tuple] = None) -> pd.DataFrame:
        """Test all configurations and return results.

        Args:
            configs: List of (model, device, resolution) tuples.
                    If None, uses default comprehensive configs.

        Returns:
            Pandas DataFrame with all results
        """
        if configs is None:
            # Default configurations
            devices = ['cpu']
            if torch.backends.mps.is_available():
                devices.append('mps')
            if torch.cuda.is_available():
                devices.append('cuda')

            models = ['mobilenet', 'resnet50']
            resolutions = [
                (640, 480),
                (1280, 720),
            ]

            # Create all combinations
            configs = []
            for device in devices:
                for model in models:
                    for res in resolutions:
                        configs.append((model, device, res))

        results = []

        for i, (model, device, res) in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Benchmarking {model} on {device} at {res[0]}x{res[1]}...")

            try:
                result = E2EBenchmark.benchmark_configuration(
                    model_name=model,
                    device=device,
                    resolution=res,
                    num_frames=50,  # Reduced for faster testing
                    warmup=5
                )
                results.append(result)

                print(f"  → {result['mean_fps']:.2f} FPS (mean), "
                      f"{result['p95_fps']:.2f} FPS (P95)")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        return pd.DataFrame(results)


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table."""

    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100 + "\n")

    print(f"{'Configuration':<35} {'Mean FPS':<12} {'P95 FPS':<12} {'Mean (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(f"{row['config']:<35} "
              f"{row['mean_fps']:<12.2f} "
              f"{row['p95_fps']:<12.2f} "
              f"{row['mean_ms']:<12.2f} "
              f"{row['p95_ms']:<12.2f}")

    print("="*100 + "\n")


def print_device_comparison(df: pd.DataFrame):
    """Print device comparison for same model/resolution."""

    devices = df['device'].unique()

    if len(devices) > 1:
        print("\n" + "="*100)
        print("DEVICE COMPARISON (MobileNet @ 640x480)")
        print("="*100 + "\n")

        # Filter for mobilenet at 640x480
        subset = df[(df['model'] == 'mobilenet') & (df['resolution'] == '640x480')]

        if len(subset) > 1:
            print(f"{'Device':<15} {'Mean FPS':<15} {'P95 FPS':<15} {'Speedup vs CPU':<20}")
            print("-" * 100)

            cpu_fps = None
            for _, row in subset.iterrows():
                if row['device'] == 'cpu':
                    cpu_fps = row['mean_fps']

            for _, row in subset.iterrows():
                if cpu_fps and row['device'] != 'cpu':
                    speedup = row['mean_fps'] / cpu_fps
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "baseline"

                print(f"{row['device']:<15} "
                      f"{row['mean_fps']:<15.2f} "
                      f"{row['p95_fps']:<15.2f} "
                      f"{speedup_str:<20}")

            print("="*100 + "\n")


def print_model_comparison(df: pd.DataFrame):
    """Print model comparison on same device."""

    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100 + "\n")

    devices = df['device'].unique()

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 100)
        print(f"{'Model':<15} {'Resolution':<15} {'Mean FPS':<15} {'P95 FPS':<15} {'Speedup':<15}")
        print("-" * 100)

        device_df = df[df['device'] == device].sort_values('mean_fps', ascending=False)

        baseline_fps = None
        for _, row in device_df.iterrows():
            if baseline_fps is None:
                baseline_fps = row['mean_fps']
                speedup_str = "baseline"
            else:
                speedup = row['mean_fps'] / baseline_fps
                speedup_str = f"{speedup:.2f}x"

            print(f"{row['model']:<15} "
                  f"{row['resolution']:<15} "
                  f"{row['mean_fps']:<15.2f} "
                  f"{row['p95_fps']:<15.2f} "
                  f"{speedup_str:<15}")

    print("="*100 + "\n")


def print_resolution_scaling(df: pd.DataFrame):
    """Print resolution scaling analysis."""

    print("\n" + "="*100)
    print("RESOLUTION SCALING ANALYSIS")
    print("="*100 + "\n")

    # Group by model and device
    for model in df['model'].unique():
        for device in df['device'].unique():
            subset = df[(df['model'] == model) & (df['device'] == device)].sort_values('megapixels')

            if len(subset) > 1:
                print(f"\n{model.upper()} on {device.upper()}")
                print("-" * 100)
                print(f"{'Resolution':<15} {'Megapixels':<15} {'FPS':<15} {'MP/s':<15} {'ms/frame':<15}")
                print("-" * 100)

                for _, row in subset.iterrows():
                    print(f"{row['resolution']:<15} "
                          f"{row['megapixels']:<15.2f} "
                          f"{row['mean_fps']:<15.2f} "
                          f"{row['mpixels_per_sec']:<15.2f} "
                          f"{row['mean_ms']:<15.2f}")

                # Calculate scaling efficiency
                if len(subset) >= 2:
                    rows = list(subset.iterrows())
                    small = rows[0][1]
                    large = rows[-1][1]

                    pixel_ratio = large['megapixels'] / small['megapixels']
                    time_ratio = large['mean_ms'] / small['mean_ms']
                    efficiency = pixel_ratio / time_ratio

                    print("-" * 100)
                    print(f"Scaling efficiency: {efficiency:.2f} (1.0 = linear)")

                    if efficiency > 0.9:
                        print("✓ Excellent: Nearly linear scaling")
                    elif efficiency > 0.7:
                        print("✓ Good: Sub-linear scaling (efficient)")
                    else:
                        print("⚠️  Poor: Super-linear scaling (resolution bottleneck)")

    print("\n" + "="*100 + "\n")


def print_recommendations(df: pd.DataFrame):
    """Print optimization recommendations based on results."""

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100 + "\n")

    # Find best configuration overall
    best = df.loc[df['mean_fps'].idxmax()]

    print(f"🏆 Best overall configuration:")
    print(f"   {best['config']}")
    print(f"   {best['mean_fps']:.2f} FPS (mean), {best['p95_fps']:.2f} FPS (P95)\n")

    # Find best for real-time (30+ FPS)
    realtime = df[df['mean_fps'] >= 30]

    if len(realtime) > 0:
        print(f"✓ Real-time capable configurations (≥30 FPS):")
        for _, row in realtime.sort_values('mean_fps', ascending=False).iterrows():
            print(f"   - {row['config']}: {row['mean_fps']:.2f} FPS")
    else:
        print(f"⚠️  No configurations achieve real-time performance (≥30 FPS)")

        # Find fastest
        fastest = df.loc[df['mean_fps'].idxmax()]
        print(f"   Fastest: {fastest['config']} at {fastest['mean_fps']:.2f} FPS")
        print(f"   Consider: Lower resolution, lighter model, or frame skipping")

    print("\n")

    # Device recommendations
    if 'mps' in df['device'].values and 'cpu' in df['device'].values:
        mps_subset = df[(df['device'] == 'mps') & (df['model'] == 'mobilenet') & (df['resolution'] == '640x480')]
        cpu_subset = df[(df['device'] == 'cpu') & (df['model'] == 'mobilenet') & (df['resolution'] == '640x480')]

        if len(mps_subset) > 0 and len(cpu_subset) > 0:
            mps_fps = mps_subset.iloc[0]['mean_fps']
            cpu_fps = cpu_subset.iloc[0]['mean_fps']
            speedup = mps_fps / cpu_fps

            print(f"🎯 MPS Performance:")
            if speedup > 2.0:
                print(f"   ✓ MPS is {speedup:.2f}x faster than CPU - use MPS!")
            elif speedup > 1.2:
                print(f"   ✓ MPS is {speedup:.2f}x faster than CPU - moderate improvement")
            elif speedup > 0.8:
                print(f"   ⚠️  MPS is {speedup:.2f}x vs CPU - minimal benefit")
            else:
                print(f"   ❌ MPS is slower than CPU ({speedup:.2f}x) - use CPU instead!")
                print(f"      This suggests MPS overhead or missing optimizations")

    print("\n" + "="*100 + "\n")


def main():
    """Run comprehensive end-to-end benchmarks."""

    print("\n" + "="*100)
    print("END-TO-END BENCHMARK SUITE")
    print("="*100 + "\n")

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print(f"Available devices: {devices}")
    print(f"Testing configurations...\n")

    # Run benchmarks
    df = E2EBenchmark.run_full_benchmark()

    # Save results
    output_file = 'benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")

    # Print analyses
    print_summary_table(df)
    print_device_comparison(df)
    print_model_comparison(df)
    print_resolution_scaling(df)
    print_recommendations(df)


if __name__ == '__main__':
    main()
