"""PyTorch profiler wrapper for detailed performance analysis.

This module uses PyTorch's built-in profiler to identify bottlenecks
at the operation level.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detecting.detectors.object_detector import ObjectDetector


class ProfilerAnalysis:
    """PyTorch profiler wrapper for detection pipeline."""

    @staticmethod
    def profile_detector(model_name: str = 'mobilenet', device: str = 'cpu',
                        resolution: tuple = (640, 480), num_iterations: int = 10):
        """Profile detector using PyTorch profiler.

        Args:
            model_name: Model name
            device: Device name
            resolution: Tuple of (width, height)
            num_iterations: Number of iterations to profile

        Returns:
            Profiler object with results
        """
        w, h = resolution

        # Create detector
        detector = ObjectDetector(model=model_name, device=device)

        # Create dummy frames
        frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                  for _ in range(num_iterations)]

        # Warmup
        for _ in range(3):
            detector.detect(frames[0])

        # Determine activities based on device
        activities = [ProfilerActivity.CPU]
        if device == 'cuda' and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        # Note: MPS profiling may not be fully supported in all PyTorch versions

        # Profile
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for frame in frames:
                with record_function("detect"):
                    detections = detector.detect(frame)

        return prof

    @staticmethod
    def profile_pipeline_stages(model_name: str = 'mobilenet', device: str = 'cpu',
                               resolution: tuple = (640, 480), num_iterations: int = 10):
        """Profile individual pipeline stages with annotations.

        Args:
            model_name: Model name
            device: Device name
            resolution: Tuple of (width, height)
            num_iterations: Number of iterations to profile

        Returns:
            Profiler object with results
        """
        w, h = resolution

        # Create detector
        detector = ObjectDetector(model=model_name, device=device)

        # Create dummy frames
        frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                  for _ in range(num_iterations)]

        # Warmup
        for _ in range(3):
            detector.detect(frames[0])

        # Determine activities
        activities = [ProfilerActivity.CPU]
        if device == 'cuda' and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Profile with stage annotations
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for frame in frames:
                with record_function("preprocess"):
                    input_tensor = detector.preprocess(frame)

                with record_function("inference"):
                    with torch.no_grad():
                        output = detector.inference(input_tensor)

                with record_function("postprocess"):
                    detections = detector.postprocess(output, frame.shape[:2])

        return prof


def print_top_operations(prof, sort_by: str = 'cpu_time_total', row_limit: int = 20):
    """Print top operations from profiler.

    Args:
        prof: PyTorch profiler object
        sort_by: Sort key ('cpu_time_total', 'cuda_time_total', 'cpu_memory_usage')
        row_limit: Number of rows to display
    """
    print("\n" + "="*100)
    print(f"TOP {row_limit} OPERATIONS (sorted by {sort_by})")
    print("="*100 + "\n")

    print(prof.key_averages().table(
        sort_by=sort_by,
        row_limit=row_limit
    ))


def print_stage_breakdown(prof):
    """Print breakdown by pipeline stage.

    Args:
        prof: PyTorch profiler object
    """
    print("\n" + "="*100)
    print("PIPELINE STAGE BREAKDOWN")
    print("="*100 + "\n")

    # Get events grouped by name
    events = prof.key_averages()

    # Find our custom markers
    stages = {}
    for evt in events:
        if evt.key in ['preprocess', 'inference', 'postprocess', 'detect']:
            stages[evt.key] = evt

    if stages:
        print(f"{'Stage':<20} {'CPU Time (ms)':<20} {'# Calls':<15} {'Avg (ms)':<15}")
        print("-" * 100)

        total_time = 0
        for stage_name in ['preprocess', 'inference', 'postprocess']:
            if stage_name in stages:
                evt = stages[stage_name]
                cpu_time_ms = evt.cpu_time_total / 1000  # Convert to ms
                avg_time_ms = cpu_time_ms / evt.count if evt.count > 0 else 0
                total_time += cpu_time_ms

                print(f"{stage_name:<20} "
                      f"{cpu_time_ms:<20.2f} "
                      f"{evt.count:<15} "
                      f"{avg_time_ms:<15.2f}")

        if total_time > 0:
            print("-" * 100)
            print(f"{'TOTAL':<20} {total_time:<20.2f}\n")

            # Percentage breakdown
            print("Percentage breakdown:")
            for stage_name in ['preprocess', 'inference', 'postprocess']:
                if stage_name in stages:
                    evt = stages[stage_name]
                    cpu_time_ms = evt.cpu_time_total / 1000
                    percentage = (cpu_time_ms / total_time) * 100
                    print(f"  {stage_name}: {percentage:.1f}%")


def identify_bottlenecks(prof, threshold_pct: float = 10.0):
    """Identify operations taking more than threshold percentage of time.

    Args:
        prof: PyTorch profiler object
        threshold_pct: Minimum percentage to report
    """
    print("\n" + "="*100)
    print(f"BOTTLENECKS (operations taking >{threshold_pct}% of CPU time)")
    print("="*100 + "\n")

    events = prof.key_averages()

    # Calculate total time
    total_time = sum(evt.cpu_time_total for evt in events)

    bottlenecks = []
    for evt in events:
        percentage = (evt.cpu_time_total / total_time) * 100
        if percentage >= threshold_pct:
            bottlenecks.append((evt.key, percentage, evt.cpu_time_total / 1000))

    if bottlenecks:
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Operation':<50} {'% of Total':<15} {'Time (ms)':<15}")
        print("-" * 100)

        for op_name, pct, time_ms in bottlenecks:
            print(f"{op_name[:50]:<50} {pct:<15.1f} {time_ms:<15.2f}")

        print("\n")
    else:
        print(f"No individual operations exceed {threshold_pct}% threshold\n")


def analyze_memory_usage(prof):
    """Analyze memory usage patterns.

    Args:
        prof: PyTorch profiler object
    """
    print("\n" + "="*100)
    print("MEMORY USAGE ANALYSIS")
    print("="*100 + "\n")

    events = prof.key_averages()

    # Find memory-intensive operations
    memory_ops = []
    for evt in events:
        if evt.cpu_memory_usage > 0:
            memory_ops.append((evt.key, evt.cpu_memory_usage / (1024**2), evt.count))

    if memory_ops:
        memory_ops.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Operation':<50} {'Memory (MB)':<15} {'# Calls':<15}")
        print("-" * 100)

        for op_name, memory_mb, count in memory_ops[:20]:
            print(f"{op_name[:50]:<50} {memory_mb:<15.2f} {count:<15}")

        print("\n")

        # Total memory
        total_memory_mb = sum(mem for _, mem, _ in memory_ops)
        print(f"Total CPU memory allocated: {total_memory_mb:.2f} MB\n")
    else:
        print("No memory usage data available\n")


def generate_recommendations(prof):
    """Generate optimization recommendations based on profiler results.

    Args:
        prof: PyTorch profiler object
    """
    print("\n" + "="*100)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*100 + "\n")

    events = prof.key_averages()

    # Calculate total time
    total_time = sum(evt.cpu_time_total for evt in events)

    # Look for common issues
    issues = []

    # 1. Check for data transfer (aten::to, aten::copy_)
    transfer_time = sum(evt.cpu_time_total for evt in events
                       if 'copy' in evt.key.lower() or evt.key == 'aten::to')
    transfer_pct = (transfer_time / total_time) * 100

    if transfer_pct > 5:
        issues.append({
            'severity': 'HIGH' if transfer_pct > 15 else 'MEDIUM',
            'issue': f'Data transfer operations take {transfer_pct:.1f}% of time',
            'recommendation': 'Minimize CPU↔GPU transfers, keep tensors on device'
        })

    # 2. Check for contiguous operations
    contiguous_time = sum(evt.cpu_time_total for evt in events if 'contiguous' in evt.key.lower())
    contiguous_pct = (contiguous_time / total_time) * 100

    if contiguous_pct > 2:
        issues.append({
            'severity': 'MEDIUM',
            'issue': f'Contiguous operations take {contiguous_pct:.1f}% of time',
            'recommendation': 'Use .contiguous() after transpose/permute to avoid overhead'
        })

    # 3. Check for linear/matmul dominance
    compute_ops = ['aten::linear', 'aten::matmul', 'aten::addmm', 'aten::mm']
    compute_time = sum(evt.cpu_time_total for evt in events if evt.key in compute_ops)
    compute_pct = (compute_time / total_time) * 100

    if compute_pct > 50:
        issues.append({
            'severity': 'INFO',
            'issue': f'Computation (matmul/linear) takes {compute_pct:.1f}% of time',
            'recommendation': 'Model inference is the bottleneck - consider lighter model or quantization'
        })

    # 4. Check for convolution time
    conv_time = sum(evt.cpu_time_total for evt in events if 'conv' in evt.key.lower())
    conv_pct = (conv_time / total_time) * 100

    if conv_pct > 40:
        issues.append({
            'severity': 'INFO',
            'issue': f'Convolution operations take {conv_pct:.1f}% of time',
            'recommendation': 'Convolutions dominate - consider lower resolution or depthwise separable convs'
        })

    # Print recommendations
    if issues:
        for i, issue in enumerate(sorted(issues, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'INFO': 2}[x['severity']])):
            severity_symbol = {'HIGH': '🔴', 'MEDIUM': '🟡', 'INFO': '🔵'}[issue['severity']]
            print(f"{severity_symbol} [{issue['severity']}] {issue['issue']}")
            print(f"   → {issue['recommendation']}\n")
    else:
        print("✓ No obvious performance issues detected\n")

    print("="*100 + "\n")


def main():
    """Run profiler analysis."""

    print("\n" + "="*100)
    print("PYTORCH PROFILER ANALYSIS")
    print("="*100 + "\n")

    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')

    print(f"Available devices: {devices}\n")

    # Test configurations
    configs = [
        ('mobilenet', 'cpu', (640, 480)),
    ]

    # Add GPU config if available
    if len(devices) > 1:
        gpu_device = 'mps' if 'mps' in devices else 'cuda'
        configs.append(('mobilenet', gpu_device, (640, 480)))

    for model, device, resolution in configs:
        print("\n" + "="*100)
        print(f"Profiling: {model} on {device} at {resolution[0]}x{resolution[1]}")
        print("="*100)

        try:
            # Profile with stage annotations
            prof = ProfilerAnalysis.profile_pipeline_stages(
                model_name=model,
                device=device,
                resolution=resolution,
                num_iterations=10
            )

            # Print analyses
            print_stage_breakdown(prof)
            print_top_operations(prof, sort_by='cpu_time_total', row_limit=15)
            identify_bottlenecks(prof, threshold_pct=5.0)
            analyze_memory_usage(prof)
            generate_recommendations(prof)

            # Export trace for visualization
            trace_file = f"trace_{model}_{device}_{resolution[0]}x{resolution[1]}.json"
            prof.export_chrome_trace(trace_file)
            print(f"✓ Chrome trace exported to {trace_file}")
            print(f"  Open in chrome://tracing or https://ui.perfetto.dev/\n")

        except Exception as e:
            print(f"\n❌ Error profiling {model} on {device}: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "="*100)
    print("PROFILING COMPLETE")
    print("="*100 + "\n")

    print("Next steps:")
    print("  1. Open trace files in chrome://tracing to visualize timeline")
    print("  2. Look for gaps between operations (indicates overhead)")
    print("  3. Identify longest-running operations")
    print("  4. Check for unexpected data transfers\n")


if __name__ == '__main__':
    main()
