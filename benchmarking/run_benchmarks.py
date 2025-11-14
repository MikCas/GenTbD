"""
Benchmark runner script for GenTbD performance analysis.

This script demonstrates how to properly benchmark a codebase before and after
optimizations, following best practices for performance analysis.

Usage:
    # Run baseline benchmarks (before optimization)
    python benchmarking/run_benchmarks.py --mode baseline

    # Run optimized benchmarks (after optimization)
    python benchmarking/run_benchmarks.py --mode optimized

    # Compare results
    python benchmarking/run_benchmarks.py --mode compare
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_framework import (
    BenchmarkRunner,
    create_standard_configs
)
from src.detecting.detectors.object_detector import ObjectDetector


def run_baseline_benchmarks():
    """
    Run baseline benchmarks on the CURRENT (unoptimized) code.

    This should be run BEFORE making any optimizations to establish
    a performance baseline for comparison.
    """
    print("\n" + "="*80)
    print("RUNNING BASELINE BENCHMARKS")
    print("="*80)
    print("\nThis will test the CURRENT (unoptimized) code performance.")
    print("Results will be saved to: benchmarking/results/baseline.json")
    print("\n⚠️  Make sure you haven't applied optimizations yet!")
    print("="*80)

    input("\nPress Enter to continue...")

    runner = BenchmarkRunner()
    configs = create_standard_configs()

    for config in configs:
        # Create detector with config settings
        detector = ObjectDetector(
            model=config.model,
            device=config.device,
            conf_threshold=config.conf_threshold,
            classes=config.classes
        )

        # Run benchmark
        runner.run_benchmark(detector, config, verbose=True)

    # Save results
    runner.save_results("baseline")

    print("\n✅ Baseline benchmarks complete!")
    print("   Next step: Apply optimizations, then run: --mode optimized")


def run_optimized_benchmarks():
    """
    Run benchmarks on the OPTIMIZED code.

    This should be run AFTER applying optimizations to measure improvements.
    """
    print("\n" + "="*80)
    print("RUNNING OPTIMIZED BENCHMARKS")
    print("="*80)
    print("\nThis will test the OPTIMIZED code performance.")
    print("Results will be saved to: benchmarking/results/optimized.json")
    print("\n✅ Make sure you've applied optimizations!")
    print("="*80)

    input("\nPress Enter to continue...")

    runner = BenchmarkRunner()
    configs = create_standard_configs()

    for config in configs:
        # Create detector with config settings
        detector = ObjectDetector(
            model=config.model,
            device=config.device,
            conf_threshold=config.conf_threshold,
            classes=config.classes
        )

        # Run benchmark
        runner.run_benchmark(detector, config, verbose=True)

    # Save results
    runner.save_results("optimized")

    print("\n✅ Optimized benchmarks complete!")
    print("   Next step: Compare results with: --mode compare")


def compare_benchmarks():
    """
    Compare baseline vs optimized benchmarks.

    This loads both result files and generates a detailed comparison report
    showing the performance improvements achieved.
    """
    print("\n" + "="*80)
    print("COMPARING BASELINE VS OPTIMIZED")
    print("="*80)

    # Load results
    baseline_path = Path("benchmarking/results/baseline.json")
    optimized_path = Path("benchmarking/results/optimized.json")

    if not baseline_path.exists():
        print(f"\n❌ Baseline results not found: {baseline_path}")
        print("   Run: python benchmarking/run_benchmarks.py --mode baseline")
        return

    if not optimized_path.exists():
        print(f"\n❌ Optimized results not found: {optimized_path}")
        print("   Run: python benchmarking/run_benchmarks.py --mode optimized")
        return

    print(f"\n📁 Loading baseline results: {baseline_path}")
    baseline = BenchmarkRunner.load_results(str(baseline_path))

    print(f"📁 Loading optimized results: {optimized_path}")
    optimized = BenchmarkRunner.load_results(str(optimized_path))

    # Generate comparison report
    report = BenchmarkRunner.compare_results(baseline, optimized)
    print(report)

    # Save report to file
    report_path = Path("benchmarking/results/comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✅ Comparison report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run GenTbD performance benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline benchmarks (before optimization)
  python benchmarking/run_benchmarks.py --mode baseline

  # Run optimized benchmarks (after optimization)
  python benchmarking/run_benchmarks.py --mode optimized

  # Compare baseline vs optimized
  python benchmarking/run_benchmarks.py --mode compare
        """
    )

    parser.add_argument(
        '--mode',
        choices=['baseline', 'optimized', 'compare'],
        required=True,
        help='Benchmark mode: baseline (before), optimized (after), or compare'
    )

    args = parser.parse_args()

    if args.mode == 'baseline':
        run_baseline_benchmarks()
    elif args.mode == 'optimized':
        run_optimized_benchmarks()
    elif args.mode == 'compare':
        compare_benchmarks()


if __name__ == '__main__':
    main()
