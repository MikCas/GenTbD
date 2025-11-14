"""Benchmark CPU vs MPS performance on M3 chip."""

import time
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detecting.detectors import ObjectDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_detector(device, video_path, num_frames=100, warmup=5):
    """Benchmark detector on specified device.

    Args:
        device: 'cpu' or 'mps'
        video_path: Path to test video
        num_frames: Number of frames to process
        warmup: Number of warmup frames (not counted)

    Returns:
        dict with timing statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking on device: {device.upper()}")
    logger.info(f"{'='*60}")

    # Create detector
    detector = ObjectDetector(
        model='mobilenet',
        device=device,
        conf_threshold=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video resolution: {width}x{height}")

    # Storage for timing
    inference_times = []
    preprocess_times = []
    postprocess_times = []

    frame_count = 0

    while frame_count < num_frames + warmup:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Timing for each stage
        t_total_start = time.time()

        # Preprocess
        t_prep_start = time.time()
        input_tensor = detector.preprocess(frame)
        t_prep = time.time() - t_prep_start

        # Inference
        t_inf_start = time.time()
        with torch.no_grad():
            output = detector.inference(input_tensor)

        # Force synchronization (critical for GPU timing)
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        t_inf = time.time() - t_inf_start

        # Postprocess
        t_post_start = time.time()
        detections = detector.postprocess(output, frame.shape[:2])
        t_post = time.time() - t_post_start

        t_total = time.time() - t_total_start

        # Skip warmup frames
        if frame_count >= warmup:
            inference_times.append(t_inf)
            preprocess_times.append(t_prep)
            postprocess_times.append(t_post)

        frame_count += 1

        # Progress
        if frame_count % 10 == 0:
            logger.info(f"Processed {frame_count}/{num_frames + warmup} frames")

    cap.release()

    # Calculate statistics
    inference_times = np.array(inference_times)
    preprocess_times = np.array(preprocess_times)
    postprocess_times = np.array(postprocess_times)
    total_times = preprocess_times + inference_times + postprocess_times

    stats = {
        'device': device,
        'num_frames': num_frames,
        'preprocess': {
            'mean_ms': preprocess_times.mean() * 1000,
            'std_ms': preprocess_times.std() * 1000,
        },
        'inference': {
            'mean_ms': inference_times.mean() * 1000,
            'std_ms': inference_times.std() * 1000,
        },
        'postprocess': {
            'mean_ms': postprocess_times.mean() * 1000,
            'std_ms': postprocess_times.std() * 1000,
        },
        'total': {
            'mean_ms': total_times.mean() * 1000,
            'std_ms': total_times.std() * 1000,
            'fps': 1000 / (total_times.mean() * 1000),
        }
    }

    return stats

def print_stats(stats):
    """Print benchmark statistics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Results for {stats['device'].upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Frames processed: {stats['num_frames']}")
    logger.info(f"\nBreakdown:")
    logger.info(f"  Preprocessing:  {stats['preprocess']['mean_ms']:6.2f} ± {stats['preprocess']['std_ms']:5.2f} ms")
    logger.info(f"  Inference:      {stats['inference']['mean_ms']:6.2f} ± {stats['inference']['std_ms']:5.2f} ms")
    logger.info(f"  Postprocessing: {stats['postprocess']['mean_ms']:6.2f} ± {stats['postprocess']['std_ms']:5.2f} ms")
    logger.info(f"  {'─'*50}")
    logger.info(f"  Total:          {stats['total']['mean_ms']:6.2f} ± {stats['total']['std_ms']:5.2f} ms")
    logger.info(f"\n  Average FPS:    {stats['total']['fps']:.2f}")

def compare_devices(video_path='data/TownCent.mp4', num_frames=100):
    """Compare CPU vs MPS performance."""

    logger.info("\n" + "="*60)
    logger.info("GenTbD Performance Benchmark: CPU vs MPS (M3)")
    logger.info("="*60)

    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    logger.info(f"\nMPS Available: {mps_available}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    if not mps_available:
        logger.warning("\n⚠️  MPS not available! Only benchmarking CPU.")
        logger.warning("Make sure you're running on Apple Silicon (M1/M2/M3)")
        devices = ['cpu']
    else:
        devices = ['cpu', 'mps']

    results = {}

    # Benchmark each device
    for device in devices:
        try:
            stats = benchmark_detector(device, video_path, num_frames=num_frames)
            results[device] = stats
            print_stats(stats)
        except Exception as e:
            logger.error(f"Error benchmarking {device}: {e}")
            import traceback
            traceback.print_exc()

    # Comparison
    if 'mps' in results and 'cpu' in results:
        logger.info("\n" + "="*60)
        logger.info("COMPARISON")
        logger.info("="*60)

        cpu_fps = results['cpu']['total']['fps']
        mps_fps = results['mps']['total']['fps']
        speedup = mps_fps / cpu_fps

        logger.info(f"\nThroughput:")
        logger.info(f"  CPU:     {cpu_fps:6.2f} FPS  ({results['cpu']['total']['mean_ms']:.2f} ms/frame)")
        logger.info(f"  MPS:     {mps_fps:6.2f} FPS  ({results['mps']['total']['mean_ms']:.2f} ms/frame)")
        logger.info(f"  Speedup: {speedup:6.2f}x")

        # Detailed breakdown
        logger.info(f"\nPreprocessing speedup:  {results['cpu']['preprocess']['mean_ms'] / results['mps']['preprocess']['mean_ms']:.2f}x")
        logger.info(f"Inference speedup:      {results['cpu']['inference']['mean_ms'] / results['mps']['inference']['mean_ms']:.2f}x")
        logger.info(f"Postprocessing speedup: {results['cpu']['postprocess']['mean_ms'] / results['mps']['postprocess']['mean_ms']:.2f}x")

        # Diagnosis
        logger.info(f"\n{'='*60}")
        logger.info("DIAGNOSIS")
        logger.info("="*60)

        if speedup < 1.0:
            logger.warning(f"\n⚠️  MPS is {1/speedup:.2f}x SLOWER than CPU!")
            logger.warning("\nLikely causes:")
            logger.warning("  1. Frequent CPU-GPU synchronization (.cpu() calls)")
            logger.warning("  2. Small tensor transfers dominate computation time")
            logger.warning("  3. No batching (single-frame inference)")
            logger.warning("\n📖 See OPTIMIZATION_ANALYSIS.md for detailed fixes")

        elif speedup < 2.0:
            logger.warning(f"\n⚠️  MPS is only {speedup:.2f}x faster - expected 3-5x")
            logger.warning("\nPotential issues:")
            logger.warning("  1. Postprocessing on CPU (notice high postprocess time on MPS)")
            logger.warning("  2. Data transfer overhead")
            logger.warning("\n📖 See OPTIMIZATION_ANALYSIS.md Task 2 for optimizations")

        elif speedup < 3.0:
            logger.info(f"\n✅ MPS is {speedup:.2f}x faster - Decent performance")
            logger.info("   Could be improved with optimizations (see OPTIMIZATION_ANALYSIS.md)")

        else:
            logger.info(f"\n✅ MPS is {speedup:.2f}x faster - Excellent performance!")
            logger.info("   M3 GPU is being utilized effectively")

        # Specific bottleneck identification
        mps_prep_pct = (results['mps']['preprocess']['mean_ms'] / results['mps']['total']['mean_ms']) * 100
        mps_inf_pct = (results['mps']['inference']['mean_ms'] / results['mps']['total']['mean_ms']) * 100
        mps_post_pct = (results['mps']['postprocess']['mean_ms'] / results['mps']['total']['mean_ms']) * 100

        logger.info(f"\nMPS Time Breakdown:")
        logger.info(f"  Preprocessing:  {mps_prep_pct:5.1f}%")
        logger.info(f"  Inference:      {mps_inf_pct:5.1f}%")
        logger.info(f"  Postprocessing: {mps_post_pct:5.1f}%")

        if mps_post_pct > 30:
            logger.warning(f"\n⚠️  Postprocessing is {mps_post_pct:.1f}% of total time!")
            logger.warning("   → Implement Optimization #1: Keep tensors on GPU")

        if mps_prep_pct > 20:
            logger.warning(f"\n⚠️  Preprocessing is {mps_prep_pct:.1f}% of total time!")
            logger.warning("   → Implement Optimization #2: GPU preprocessing")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark CPU vs MPS performance on M3 chip',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/benchmark_mps.py
  python experiments/benchmark_mps.py --frames 200
  python experiments/benchmark_mps.py --video path/to/video.mp4

Output:
  - Detailed timing breakdown for each stage
  - FPS comparison between CPU and MPS
  - Diagnostic information about bottlenecks
  - Optimization recommendations

See OPTIMIZATION_ANALYSIS.md for detailed optimization strategies.
        """
    )
    parser.add_argument('--video', default='data/TownCent.mp4',
                       help='Path to test video (default: data/TownCent.mp4)')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to process (default: 100)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup frames to skip (default: 5)')

    args = parser.parse_args()

    # Check video file exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        logger.info(f"Please provide a valid video file with --video")
        sys.exit(1)

    # Run benchmark
    try:
        results = compare_devices(args.video, args.frames)
        logger.info(f"\n{'='*60}")
        logger.info("Benchmark complete!")
        logger.info(f"{'='*60}\n")

    except KeyboardInterrupt:
        logger.info("\n\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
