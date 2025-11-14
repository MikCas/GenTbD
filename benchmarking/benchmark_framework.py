"""
Comprehensive Benchmarking Framework for GenTbD Performance Analysis

This module provides a professional benchmarking framework that demonstrates
best practices for performance analysis when optimizing code.

Key Concepts Demonstrated:
1. Warmup iterations to stabilize performance
2. Multiple samples for statistical significance
3. Outlier detection and handling
4. Statistical analysis (mean, median, std, percentiles)
5. Baseline vs optimized comparison
6. Result persistence for historical tracking
7. Different test scenarios (resolution, model, device)

Usage:
    # Run baseline benchmarks
    python -m benchmarking.benchmark_framework --mode baseline

    # Run optimized benchmarks
    python -m benchmarking.benchmark_framework --mode optimized

    # Compare results
    python -m benchmarking.benchmark_framework --mode compare
"""

import time
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark scenario."""
    name: str
    resolution: Tuple[int, int]  # (width, height)
    model: str
    device: str
    conf_threshold: float
    classes: Optional[List[int]]
    num_warmup: int = 5
    num_samples: int = 30
    description: str = ""


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    samples: List[float]  # All timing samples in milliseconds
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    timestamp: str
    git_commit: str = ""


class BenchmarkRunner:
    """
    Professional benchmarking framework for detection pipeline.

    This class demonstrates proper performance benchmarking techniques:
    - Warmup iterations to stabilize CPU/GPU states
    - Multiple samples for statistical significance
    - Outlier detection using IQR method
    - Comprehensive statistics (mean, median, percentiles)
    - Result persistence for historical comparison
    """

    def __init__(self, output_dir: str = "benchmarking/results"):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: Dict[str, BenchmarkResult] = {}

    def create_synthetic_frame(self, width: int, height: int) -> np.ndarray:
        """Create a synthetic frame for testing.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            Synthetic BGR frame
        """
        # Create a realistic-looking frame with patterns
        # This is more realistic than random noise
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some structure (gradients, shapes) that detectors can find
        # This ensures the detector actually does work
        cv2.rectangle(frame, (100, 100), (300, 400), (255, 0, 0), -1)
        cv2.circle(frame, (500, 300), 80, (0, 255, 0), -1)
        cv2.rectangle(frame, (700, 200), (900, 500), (0, 0, 255), -1)

        # Add some noise for realism
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame

    def warmup(self, detector, frame: np.ndarray, num_iterations: int = 5):
        """Perform warmup iterations to stabilize performance.

        Why warmup is critical:
        - GPU kernels are compiled on first use (JIT compilation)
        - CPU caches need to be populated
        - Python GC can trigger during first runs
        - CUDA/MPS streams need initialization

        Args:
            detector: Detector instance
            frame: Test frame
            num_iterations: Number of warmup iterations
        """
        print(f"  Warming up ({num_iterations} iterations)...", end=" ", flush=True)
        for _ in range(num_iterations):
            _ = detector.detect(frame)

        # Synchronize GPU if using CUDA/MPS
        if detector.device in ['cuda', 'mps']:
            if detector.device == 'cuda':
                torch.cuda.synchronize()
            elif detector.device == 'mps':
                torch.mps.synchronize()

        print("Done")

    def run_benchmark(
        self,
        detector,
        config: BenchmarkConfig,
        verbose: bool = True
    ) -> BenchmarkResult:
        """Run a single benchmark scenario.

        This demonstrates the complete benchmarking methodology:
        1. Create synthetic test data
        2. Warmup iterations
        3. Collect multiple samples
        4. Calculate statistics
        5. Detect and report outliers

        Args:
            detector: Detector instance
            config: Benchmark configuration
            verbose: Print progress

        Returns:
            BenchmarkResult with statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmark: {config.name}")
            print(f"  Resolution: {config.resolution[0]}x{config.resolution[1]}")
            print(f"  Model: {config.model}")
            print(f"  Device: {config.device}")
            print(f"{'='*60}")

        # Create test frame
        frame = self.create_synthetic_frame(*config.resolution)

        # Warmup
        self.warmup(detector, frame, config.num_warmup)

        # Collect samples
        if verbose:
            print(f"  Collecting {config.num_samples} samples...", end=" ", flush=True)

        samples = []
        for i in range(config.num_samples):
            # Measure time
            start = time.perf_counter()
            _ = detector.detect(frame)
            elapsed = time.perf_counter() - start

            # Synchronize GPU if needed
            if detector.device in ['cuda', 'mps']:
                if detector.device == 'cuda':
                    torch.cuda.synchronize()
                elif detector.device == 'mps':
                    torch.mps.synchronize()

            samples.append(elapsed * 1000)  # Convert to ms

        if verbose:
            print("Done")

        # Calculate statistics
        samples_array = np.array(samples)

        # Detect outliers using IQR method
        q1, q3 = np.percentile(samples_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = samples_array[(samples_array < lower_bound) | (samples_array > upper_bound)]

        if verbose and len(outliers) > 0:
            print(f"  ⚠ Detected {len(outliers)} outliers (may indicate GC or system noise)")

        # Calculate statistics (including outliers - we want realistic data)
        mean_ms = float(np.mean(samples_array))
        median_ms = float(np.median(samples_array))
        std_ms = float(np.std(samples_array))
        min_ms = float(np.min(samples_array))
        max_ms = float(np.max(samples_array))
        p95_ms = float(np.percentile(samples_array, 95))
        p99_ms = float(np.percentile(samples_array, 99))
        fps = 1000.0 / mean_ms

        if verbose:
            print(f"\n  Results:")
            print(f"    Mean:   {mean_ms:7.2f} ms  ({fps:5.1f} FPS)")
            print(f"    Median: {median_ms:7.2f} ms")
            print(f"    Std:    {std_ms:7.2f} ms")
            print(f"    Min:    {min_ms:7.2f} ms")
            print(f"    Max:    {max_ms:7.2f} ms")
            print(f"    P95:    {p95_ms:7.2f} ms")
            print(f"    P99:    {p99_ms:7.2f} ms")

        # Get git commit hash for tracking
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD']
            ).decode('ascii').strip()
        except:
            git_commit = "unknown"

        # Create result object
        result = BenchmarkResult(
            config_name=config.name,
            samples=samples,
            mean_ms=mean_ms,
            median_ms=median_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            fps=fps,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit
        )

        self.results[config.name] = result
        return result

    def save_results(self, filename: str):
        """Save benchmark results to JSON file.

        Args:
            filename: Output filename (without extension)
        """
        output_path = self.output_dir / f"{filename}.json"

        # Convert results to dict
        results_dict = {
            name: asdict(result)
            for name, result in self.results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    @staticmethod
    def load_results(filepath: str) -> Dict[str, BenchmarkResult]:
        """Load benchmark results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary of benchmark results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert back to BenchmarkResult objects
        results = {
            name: BenchmarkResult(**result_data)
            for name, result_data in data.items()
        }

        return results

    @staticmethod
    def compare_results(
        baseline: Dict[str, BenchmarkResult],
        optimized: Dict[str, BenchmarkResult]
    ) -> str:
        """Compare baseline vs optimized results and generate report.

        This demonstrates how to:
        - Calculate speedup ratios
        - Determine statistical significance
        - Present results clearly

        Args:
            baseline: Baseline benchmark results
            optimized: Optimized benchmark results

        Returns:
            Formatted comparison report
        """
        report = []
        report.append("\n" + "="*80)
        report.append("PERFORMANCE COMPARISON: Baseline vs Optimized")
        report.append("="*80)

        # Compare each scenario
        for name in baseline.keys():
            if name not in optimized:
                continue

            base = baseline[name]
            opt = optimized[name]

            # Calculate improvements
            speedup = base.mean_ms / opt.mean_ms
            fps_improvement = opt.fps / base.fps
            time_saved_ms = base.mean_ms - opt.mean_ms
            time_saved_pct = (time_saved_ms / base.mean_ms) * 100

            report.append(f"\n{name}")
            report.append("-" * 80)
            report.append(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
            report.append("-" * 80)
            report.append(f"{'Mean (ms)':<20} {base.mean_ms:>7.2f}        {opt.mean_ms:>7.2f}        {speedup:>6.2f}x faster")
            report.append(f"{'Median (ms)':<20} {base.median_ms:>7.2f}        {opt.median_ms:>7.2f}")
            report.append(f"{'Std Dev (ms)':<20} {base.std_ms:>7.2f}        {opt.std_ms:>7.2f}")
            report.append(f"{'P95 (ms)':<20} {base.p95_ms:>7.2f}        {opt.p95_ms:>7.2f}")
            report.append(f"{'FPS':<20} {base.fps:>7.1f}        {opt.fps:>7.1f}        {fps_improvement:>6.2f}x faster")
            report.append(f"{'Time Saved':<20} {time_saved_ms:>7.2f} ms ({time_saved_pct:>5.1f}%)")

            # Statistical significance check (simple t-test approximation)
            # If means differ by more than 2 standard errors, likely significant
            pooled_std = np.sqrt((base.std_ms**2 + opt.std_ms**2) / 2)
            std_error = pooled_std / np.sqrt(30)  # assuming 30 samples
            t_statistic = abs(base.mean_ms - opt.mean_ms) / std_error

            if t_statistic > 2:
                report.append(f"✅ Improvement is statistically significant (t={t_statistic:.1f})")
            else:
                report.append(f"⚠️  Improvement may not be statistically significant (t={t_statistic:.1f})")

        report.append("\n" + "="*80)

        # Overall summary
        total_speedup = statistics.mean([
            baseline[name].mean_ms / optimized[name].mean_ms
            for name in baseline.keys() if name in optimized
        ])

        report.append(f"\n📊 OVERALL SUMMARY")
        report.append(f"   Average Speedup: {total_speedup:.2f}x faster")
        report.append(f"   Scenarios Tested: {len(baseline)}")
        report.append("="*80 + "\n")

        return "\n".join(report)


def create_standard_configs() -> List[BenchmarkConfig]:
    """Create standard benchmark configurations.

    This demonstrates testing across different scenarios to ensure
    optimizations work across different use cases.

    Returns:
        List of benchmark configurations
    """
    configs = [
        BenchmarkConfig(
            name="mobilenet_720p_cpu",
            resolution=(1280, 720),
            model="mobilenet",
            device="cpu",
            conf_threshold=0.5,
            classes=[1],  # People only
            description="Fast model on CPU at 720p"
        ),
        BenchmarkConfig(
            name="mobilenet_1080p_cpu",
            resolution=(1920, 1080),
            model="mobilenet",
            device="cpu",
            conf_threshold=0.5,
            classes=[1],
            description="Fast model on CPU at 1080p"
        ),
        BenchmarkConfig(
            name="resnet50_720p_cpu",
            resolution=(1280, 720),
            model="resnet50",
            device="cpu",
            conf_threshold=0.5,
            classes=[1],
            description="Accurate model on CPU at 720p"
        ),
    ]

    # Add MPS configs if available
    if torch.backends.mps.is_available():
        configs.extend([
            BenchmarkConfig(
                name="mobilenet_720p_mps",
                resolution=(1280, 720),
                model="mobilenet",
                device="mps",
                conf_threshold=0.5,
                classes=[1],
                description="Fast model on MPS at 720p"
            ),
            BenchmarkConfig(
                name="mobilenet_1080p_mps",
                resolution=(1920, 1080),
                model="mobilenet",
                device="mps",
                conf_threshold=0.5,
                classes=[1],
                description="Fast model on MPS at 1080p"
            ),
        ])

    return configs


if __name__ == '__main__':
    print("Benchmark Framework - Ready for use")
    print("Import this module to use the benchmarking tools")
