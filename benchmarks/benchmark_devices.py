#!/usr/bin/env python3
"""
Benchmark detector performance across different devices (CPU, MPS, CUDA).

Usage:
    python benchmarks/benchmark_devices.py
    python benchmarks/benchmark_devices.py --video data/TownCent.mp4 --frames 50
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import cv2
import numpy as np
from src.detecting.detectors import ObjectDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_device(device, video_path, num_frames=50, warmup=5):
    """Benchmark detector on specific device.

    Args:
        device: 'cpu', 'mps', or 'cuda'
        video_path: Path to test video
        num_frames: Number of frames to process
        warmup: Warmup frames to skip

    Returns:
        dict with timing statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {device.upper()}")
    logger.info(f"{'='*60}")

    # Create detector
    try:
        detector = ObjectDetector(model='mobilenet', device=device, conf_threshold=0.5)
    except Exception as e:
        logger.error(f"Failed to create detector on {device}: {e}")
        return None

    # Verify device
    model_device = next(detector.model.parameters()).device
    logger.info(f"Model device: {model_device}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {width}x{height}")

    # Timing storage
    total_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    frame_count = 0

    while frame_count < num_frames + warmup:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
            continue

        # Warmup
        if frame_count < warmup:
            detector.detect(frame)
            frame_count += 1
            continue

        # Timed run
        t0 = time.time()

        # Preprocess
        t_prep = time.time()
        input_tensor = detector.preprocess(frame)
        preprocess_time = time.time() - t_prep

        # Inference
        t_inf = time.time()
        with torch.no_grad():
            output = detector.inference(input_tensor)

        # Synchronize for accurate timing
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.time() - t_inf

        # Postprocess
        t_post = time.time()
        detections = detector.postprocess(output, frame.shape[:2])
        postprocess_time = time.time() - t_post

        total_time = time.time() - t0

        # Store times
        total_times.append(total_time * 1000)  # Convert to ms
        preprocess_times.append(preprocess_time * 1000)
        inference_times.append(inference_time * 1000)
        postprocess_times.append(postprocess_time * 1000)

        frame_count += 1

        if (frame_count - warmup) % 10 == 0:
            logger.info(f"  Processed {frame_count - warmup}/{num_frames}")

    cap.release()

    # Calculate statistics
    times_np = np.array(total_times)
    prep_np = np.array(preprocess_times)
    inf_np = np.array(inference_times)
    post_np = np.array(postprocess_times)

    return {
        'device': device,
        'frames': num_frames,
        'total_mean': times_np.mean(),
        'total_std': times_np.std(),
        'fps': 1000 / times_np.mean(),
        'preprocess_mean': prep_np.mean(),
        'inference_mean': inf_np.mean(),
        'postprocess_mean': post_np.mean(),
    }


def print_results(results):
    """Print benchmark results."""
    if not results:
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {results['device'].upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Frames: {results['frames']}")
    logger.info(f"\nTiming Breakdown:")
    logger.info(f"  Preprocess:  {results['preprocess_mean']:6.2f} ms")
    logger.info(f"  Inference:   {results['inference_mean']:6.2f} ms")
    logger.info(f"  Postprocess: {results['postprocess_mean']:6.2f} ms")
    logger.info(f"  Total:       {results['total_mean']:6.2f} ± {results['total_std']:.2f} ms")
    logger.info(f"\n  FPS: {results['fps']:.2f}")


def compare_devices(video_path, num_frames=50):
    """Compare all available devices."""

    logger.info("\n" + "="*60)
    logger.info("Device Comparison Benchmark")
    logger.info("="*60)

    # Detect available devices
    devices = ['cpu']

    if torch.backends.mps.is_available():
        devices.append('mps')
        logger.info("✓ MPS available")
    else:
        logger.info("✗ MPS not available")

    if torch.cuda.is_available():
        devices.append('cuda')
        logger.info(f"✓ CUDA available ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("✗ CUDA not available")

    # Benchmark each device
    results = {}
    for device in devices:
        result = benchmark_device(device, video_path, num_frames)
        if result:
            results[device] = result
            print_results(result)

    # Comparison
    if len(results) > 1:
        logger.info("\n" + "="*60)
        logger.info("COMPARISON")
        logger.info("="*60)

        baseline = results['cpu']
        for device, result in results.items():
            if device == 'cpu':
                continue
            speedup = baseline['fps'] / result['fps'] if result['fps'] < baseline['fps'] else result['fps'] / baseline['fps']
            faster = result['fps'] > baseline['fps']

            logger.info(f"\n{device.upper()} vs CPU:")
            logger.info(f"  CPU:     {baseline['fps']:6.2f} FPS")
            logger.info(f"  {device.upper()}:     {result['fps']:6.2f} FPS")

            if faster:
                logger.info(f"  Speedup: {speedup:.2f}x faster ✓")
            else:
                logger.info(f"  Slowdown: {speedup:.2f}x slower ✗")

            # Diagnosis
            if not faster:
                logger.warning(f"\n⚠️  {device.upper()} is SLOWER than CPU!")
                logger.warning("Likely causes:")
                logger.warning("  - CPU-GPU transfer overhead")
                logger.warning("  - Small tensor operations")
                logger.warning("  - Check postprocess time (should be <5ms)")

                if result['postprocess_mean'] > 10:
                    logger.warning(f"\n  → Postprocess is {result['postprocess_mean']:.1f}ms (too high!)")
                    logger.warning("  → Fix: Keep tensors on GPU longer")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark device performance")
    parser.add_argument('--video', default='data/TownCent.mp4', help='Video file path')
    parser.add_argument('--frames', type=int, default=50, help='Frames to process')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'all'], default='all',
                       help='Device to benchmark (default: all)')

    args = parser.parse_args()

    # Check video exists
    if not Path(args.video).exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    try:
        if args.device == 'all':
            results = compare_devices(args.video, args.frames)
        else:
            result = benchmark_device(args.device, args.video, args.frames)
            if result:
                print_results(result)

        logger.info(f"\n{'='*60}")
        logger.info("Benchmark complete!")
        logger.info(f"{'='*60}\n")

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
