#!/usr/bin/env python3
"""Diagnostic script to check if optimizations are actually working."""

import sys
sys.path.insert(0, '.')

import cv2
import time
from src.detecting.detectors import ObjectDetector
from src.config import Config
import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--video', default='data/TownCent.mp4')
parser.add_argument('--device', default='cpu')
parser.add_argument('--max-dimension', type=int, default=640)
args = parser.parse_args()

print("="*60)
print("DIAGNOSTIC CHECK")
print("="*60)

# Create detector WITH the fix (disable internal resize)
print(f"\n1. Creating detector on {args.device}...")
if args.max_dimension:
    # Apply the fix: set min_size to prevent internal resize to 800px
    detector = ObjectDetector(
        model='mobilenet',
        device=args.device,
        conf_threshold=0.5,
        min_size=args.max_dimension,
        max_size=args.max_dimension
    )
    print(f"   ✓ Detector created (internal resize disabled: min_size={args.max_dimension})")
else:
    detector = ObjectDetector(model='mobilenet', device=args.device, conf_threshold=0.5)
    print(f"   ✓ Detector created (using model defaults)")

# Open video
print(f"\n2. Opening video: {args.video}")
cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
if not ret:
    print("   ✗ Failed to read video!")
    sys.exit(1)

original_h, original_w = frame.shape[:2]
print(f"   ✓ Original resolution: {original_w}x{original_h}")

# Check resizing
print(f"\n3. Testing resize to max_dimension={args.max_dimension}")
if args.max_dimension:
    h, w = frame.shape[:2]
    if max(h, w) > args.max_dimension:
        scale = args.max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        print(f"   ✓ Resized to: {new_w}x{new_h}")
        print(f"   ✓ Scale factor: {scale:.3f}")
        print(f"   ✓ Pixels reduced: {w*h} → {new_w*new_h} ({(new_w*new_h)/(w*h)*100:.1f}%)")
        test_frame = resized
    else:
        print(f"   ! Frame already smaller than {args.max_dimension}")
        test_frame = frame
else:
    print(f"   ! No resizing (max_dimension not set)")
    test_frame = frame

# Benchmark inference
print(f"\n4. Benchmarking inference on {test_frame.shape[1]}x{test_frame.shape[0]}")
print("   Running 10 inferences...")

times = []
for i in range(10):
    t0 = time.time()
    detections = detector.detect(test_frame)
    elapsed = time.time() - t0
    times.append(elapsed * 1000)  # Convert to ms

    if i == 0:
        print(f"   First inference: {times[0]:.1f}ms (may include warmup)")
        print(f"   Detections found: {len(detections)}")

avg_time = sum(times[1:]) / len(times[1:])  # Skip first
print(f"\n   Average (excluding first): {avg_time:.1f}ms")
print(f"   Expected FPS: {1000/avg_time:.1f}")

# Compare with baseline (no optimization)
print(f"\n5. Comparing BEFORE vs AFTER fix...")
if args.max_dimension:
    print("   Creating baseline detector (model defaults: min_size=800)...")
    baseline_detector = ObjectDetector(model='mobilenet', device=args.device, conf_threshold=0.5)

    # Test baseline on resized frame (model will resize internally to 800px)
    baseline_times = []
    for i in range(5):
        t0 = time.time()
        _ = baseline_detector.detect(test_frame)
        baseline_times.append((time.time() - t0) * 1000)

    baseline_avg = sum(baseline_times[1:]) / len(baseline_times[1:])
    speedup = baseline_avg / avg_time

    print(f"   BEFORE (640x360 → model resizes to ~800px): {baseline_avg:.1f}ms ({1000/baseline_avg:.1f} FPS)")
    print(f"   AFTER  (640x360 stays at 640x360):        {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"   Speedup: {speedup:.1f}x")
else:
    print("   Skipped (no max_dimension set)")

# Check if flags are being used
print(f"\n6. Checking if your command would work...")
print(f"   Command: python -m src.main --video {args.video} --device {args.device} --max-dimension {args.max_dimension} --skip-frames 2")

config = Config.from_default()
config.set('video.source', args.video)
config.set('detection.device', args.device)
config.set('video.max_dimension', args.max_dimension)
config.set('video.skip_frames', 2)

print(f"\n   Config values:")
print(f"   - video.source: {config.get('video.source')}")
print(f"   - video.max_dimension: {config.get('video.max_dimension')}")
print(f"   - video.skip_frames: {config.get('video.skip_frames')}")
print(f"   - detection.device: {config.get('detection.device')}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if avg_time > 200:
    print("\n⚠️  SLOW! Inference is taking >200ms")
    print("Possible issues:")
    print("  1. Resizing is not working (check resolution above)")
    print("  2. Using wrong model (should be mobilenet)")
    print("  3. Device issue (CPU is slow, try MPS)")
elif avg_time > 100:
    print("\n⚠️  Slower than expected")
    print(f"  Expected ~50ms for 640x360, got {avg_time:.1f}ms")
    print("  This is still usable but not optimal")
else:
    print(f"\n✓ Good! Inference is {avg_time:.1f}ms")
    print(f"  Expected FPS with skip-frames 2: ~{2*1000/avg_time:.1f}")

print("\n" + "="*60)
