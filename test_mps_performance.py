"""Diagnostic script to identify MPS performance issues.

This script will help identify why MPS is slow:
1. Tests shader compilation overhead
2. Checks synchronization issues
3. Compares CPU vs MPS performance
4. Provides detailed timing breakdown

Usage:
    python test_mps_performance.py --video data/TownCent.mp4
"""

import torch
import cv2
import time
import argparse
from src.detecting.detectors import ObjectDetector


def test_mps_availability():
    """Check if MPS is available and working."""
    print("=" * 60)
    print("MPS AVAILABILITY TEST")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        if not torch.backends.mps.is_built():
            print("   PyTorch was not built with MPS support")
        return False

    print("✅ MPS is available")
    print(f"   PyTorch version: {torch.__version__}")

    # Test basic MPS operation
    try:
        x = torch.randn(10, 10, device='mps')
        y = x @ x.T
        torch.mps.synchronize()  # Wait for completion
        print("✅ Basic MPS operations work")
        return True
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False


def test_shader_compilation(detector, frame):
    """Test first-run shader compilation overhead."""
    print("\n" + "=" * 60)
    print("SHADER COMPILATION TEST (First 5 frames)")
    print("=" * 60)
    print("⚠️  First frame will be VERY slow (shader compilation)")
    print("    This is normal! Subsequent frames should be faster.\n")

    for i in range(5):
        start = time.time()
        _ = detector.detect(frame)
        torch.mps.synchronize()  # Wait for completion
        elapsed = time.time() - start

        print(f"Frame {i+1}: {elapsed*1000:6.1f} ms", end="")
        if i == 0:
            print("  ← Shader compilation (slow!)")
        elif i < 3:
            print("  ← Still compiling some shaders")
        else:
            print("  ← Should be normal speed now")


def test_timing_accuracy(detector, frame):
    """Test whether timing measurements are accurate."""
    print("\n" + "=" * 60)
    print("TIMING ACCURACY TEST")
    print("=" * 60)

    # Without synchronization
    start = time.time()
    _ = detector.detect(frame)
    elapsed_no_sync = time.time() - start

    # With synchronization
    start = time.time()
    _ = detector.detect(frame)
    torch.mps.synchronize()  # Actually wait for completion
    elapsed_with_sync = time.time() - start

    print(f"Without sync: {elapsed_no_sync*1000:.1f} ms (incorrect - queue time only)")
    print(f"With sync:    {elapsed_with_sync*1000:.1f} ms (correct - actual execution)")
    print(f"Difference:   {(elapsed_with_sync - elapsed_no_sync)*1000:.1f} ms")

    if elapsed_with_sync > elapsed_no_sync * 1.5:
        print("⚠️  Large difference detected! Your timing code is inaccurate.")
        print("   Add torch.mps.synchronize() after detector.detect()")


def detailed_timing_breakdown(detector, frame):
    """Detailed breakdown of where time is spent."""
    print("\n" + "=" * 60)
    print("DETAILED TIMING BREAKDOWN")
    print("=" * 60)

    # Warmup (shader compilation)
    for _ in range(3):
        _ = detector.detect(frame)
    torch.mps.synchronize()

    # Time each stage
    times = []
    for _ in range(10):
        t0 = time.time()
        input_tensor = detector.preprocess(frame)
        t1 = time.time()

        with torch.no_grad():
            output = detector.inference(input_tensor)
        torch.mps.synchronize()
        t2 = time.time()

        detections = detector.postprocess(output, frame.shape[:2])
        t3 = time.time()

        times.append({
            'preprocess': (t1 - t0) * 1000,
            'inference': (t2 - t1) * 1000,
            'postprocess': (t3 - t2) * 1000,
            'total': (t3 - t0) * 1000
        })

    # Average
    avg_times = {k: sum(t[k] for t in times) / len(times) for k in times[0]}

    print(f"Preprocess:  {avg_times['preprocess']:6.1f} ms")
    print(f"Inference:   {avg_times['inference']:6.1f} ms  ← GPU work")
    print(f"Postprocess: {avg_times['postprocess']:6.1f} ms")
    print(f"Total:       {avg_times['total']:6.1f} ms  ({1000/avg_times['total']:.1f} FPS)")

    # Check for bottlenecks
    if avg_times['inference'] < avg_times['total'] * 0.5:
        print("\n⚠️  Inference is not the bottleneck!")
        print("   Most time is spent in pre/post-processing.")


def compare_cpu_vs_mps(frame):
    """Compare CPU vs MPS performance."""
    print("\n" + "=" * 60)
    print("CPU vs MPS COMPARISON")
    print("=" * 60)

    # Test CPU
    print("\nTesting CPU...")
    detector_cpu = ObjectDetector.from_fasterrcnn_resnet50(device='cpu', conf_threshold=0.5)

    # Warmup
    for _ in range(3):
        _ = detector_cpu.detect(frame)

    # Measure
    cpu_times = []
    for _ in range(10):
        start = time.time()
        _ = detector_cpu.detect(frame)
        cpu_times.append(time.time() - start)

    avg_cpu = sum(cpu_times) * 1000 / len(cpu_times)
    print(f"CPU average: {avg_cpu:.1f} ms ({1000/avg_cpu:.1f} FPS)")

    # Test MPS
    print("\nTesting MPS...")
    detector_mps = ObjectDetector.from_fasterrcnn_resnet50(device='mps', conf_threshold=0.5)

    # Warmup (shader compilation)
    print("  Warming up (compiling shaders)...", end="", flush=True)
    for _ in range(5):
        _ = detector_mps.detect(frame)
    torch.mps.synchronize()
    print(" done")

    # Measure
    mps_times = []
    for _ in range(10):
        start = time.time()
        _ = detector_mps.detect(frame)
        torch.mps.synchronize()
        mps_times.append(time.time() - start)

    avg_mps = sum(mps_times) * 1000 / len(mps_times)
    print(f"MPS average: {avg_mps:.1f} ms ({1000/avg_mps:.1f} FPS)")

    # Comparison
    print("\n" + "-" * 60)
    speedup = avg_cpu / avg_mps
    if speedup > 1:
        print(f"🚀 MPS is {speedup:.2f}x FASTER than CPU")
    else:
        print(f"🐌 MPS is {1/speedup:.2f}x SLOWER than CPU")
        print("\nPossible reasons:")
        print("  1. Single-frame overhead dominates")
        print("  2. Model not optimized for MPS")
        print("  3. PyTorch MPS backend issues")
        print("  4. Unified memory overhead")

    return avg_cpu, avg_mps


def main():
    parser = argparse.ArgumentParser(description='Diagnose MPS performance issues')
    parser.add_argument('--video', default='data/TownCent.mp4', help='Test video file')
    args = parser.parse_args()

    print("\n🔍 MPS PERFORMANCE DIAGNOSTIC TOOL\n")

    # Test 1: MPS availability
    if not test_mps_availability():
        print("\n❌ MPS not available. Cannot proceed with MPS tests.")
        return

    # Load test frame
    print(f"\nLoading test frame from: {args.video}")
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Could not read video: {args.video}")
        return

    print(f"✅ Loaded frame: {frame.shape[1]}x{frame.shape[0]}")

    # Create MPS detector
    print("\nLoading detector on MPS...")
    detector = ObjectDetector.from_fasterrcnn_resnet50(device='mps', conf_threshold=0.5)
    print("✅ Detector loaded")

    # Run tests
    test_shader_compilation(detector, frame)
    test_timing_accuracy(detector, frame)
    detailed_timing_breakdown(detector, frame)
    compare_cpu_vs_mps(frame)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\n📋 Summary and Recommendations:\n")
    print("1. If first frame was very slow (>20s): This is normal shader compilation")
    print("2. If timing shows large difference: Add torch.mps.synchronize() to your code")
    print("3. If MPS is slower than CPU: Use CPU for single-frame real-time processing")
    print("4. If MPS is faster: Use MPS, but add warmup before processing video")
    print("\n")


if __name__ == '__main__':
    main()
