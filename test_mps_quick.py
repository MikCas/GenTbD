#!/usr/bin/env python3
"""Quick MPS diagnostic - test if MPS works at all."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import time
from src.detecting.detectors import ObjectDetector

print("="*60)
print("MPS Quick Diagnostic")
print("="*60)

# Check MPS
print(f"\nMPS Available: {torch.backends.mps.is_available()}")
print(f"PyTorch Version: {torch.__version__}")

# Create detector
print("\nCreating MPS detector...")
detector = ObjectDetector(model='mobilenet', device='mps', conf_threshold=0.5)

# Verify device
model_device = next(detector.model.parameters()).device
print(f"Model on device: {model_device}")

# Small test frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# First inference (will be VERY slow due to shader compilation)
print("\n⚠️  First inference (compiling shaders, may take 30-60 seconds)...")
print("Please wait, this is normal for MPS first run...\n")

t0 = time.time()
detections = detector.detect(frame)
t1 = time.time()

first_time = (t1 - t0) * 1000
print(f"First inference: {first_time:.0f}ms")

# Second inference (should be fast)
print("\nSecond inference (should be fast)...")
t0 = time.time()
detections = detector.detect(frame)
t1 = time.time()

second_time = (t1 - t0) * 1000
print(f"Second inference: {second_time:.0f}ms")

# Analysis
print("\n" + "="*60)
print("Results:")
print("="*60)
print(f"First run:  {first_time:.0f}ms (includes shader compilation)")
print(f"Second run: {second_time:.0f}ms (actual performance)")
print(f"\nSpeedup after warmup: {first_time/second_time:.1f}x")

if second_time > 500:
    print("\n⚠️  WARNING: MPS is very slow even after warmup!")
    print("This suggests a problem. Try:")
    print("  1. Update PyTorch: pip install --upgrade torch torchvision")
    print("  2. Use CPU instead: --device cpu")
elif second_time > 100:
    print("\n⚠️  MPS is slower than expected")
    print(f"Expected: ~30-60ms, Got: {second_time:.0f}ms")
else:
    print(f"\n✓ MPS is working! ~{1000/second_time:.1f} FPS")
