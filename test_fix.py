#!/usr/bin/env python3
"""Quick test to verify the internal resize fix works."""

import sys
sys.path.insert(0, '.')

import cv2
import time
from src.detecting.detectors import ObjectDetector

print("="*60)
print("Testing Internal Resize Fix")
print("="*60)

# Open video
cap = cv2.VideoCapture('data/TownCent.mp4')
ret, frame = cap.read()
h, w = frame.shape[:2]
print(f"\nOriginal frame: {w}x{h}")

# Resize to 640x360
scale = 640 / max(h, w)
new_w, new_h = int(w * scale), int(h * scale)
resized = cv2.resize(frame, (new_w, new_h))
print(f"Resized frame: {new_w}x{new_h}")

# Test 1: WITHOUT fix (default behavior)
print(f"\nTest 1: Default detector (model resizes internally to 800px)")
detector_default = ObjectDetector(model='mobilenet', device='cpu')

times = []
for i in range(5):
    t0 = time.time()
    _ = detector_default.detect(resized)
    times.append((time.time() - t0) * 1000)

avg = sum(times[1:]) / len(times[1:])
print(f"  Average: {avg:.1f}ms ({1000/avg:.1f} FPS)")
print(f"  Expected: ~250ms (slow due to internal resize)")

# Test 2: WITH fix (disable internal resize)
print(f"\nTest 2: Fixed detector (min_size=640, no internal resize)")
detector_fixed = ObjectDetector(model='mobilenet', device='cpu', min_size=640, max_size=640)

times = []
for i in range(5):
    t0 = time.time()
    _ = detector_fixed.detect(resized)
    times.append((time.time() - t0) * 1000)

avg_fixed = sum(times[1:]) / len(times[1:])
print(f"  Average: {avg_fixed:.1f}ms ({1000/avg_fixed:.1f} FPS)")
print(f"  Expected: ~50ms (fast!)")

# Analysis
print(f"\n" + "="*60)
if avg_fixed < avg * 0.7:
    speedup = avg / avg_fixed
    print(f"✓ FIX WORKS! {speedup:.1f}x faster")
    print(f"  Before: {avg:.1f}ms")
    print(f"  After:  {avg_fixed:.1f}ms")
else:
    print(f"✗ Fix didn't work - still slow")
    print(f"  Both taking ~{avg:.1f}ms")

print("="*60)
