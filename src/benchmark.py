# Quick test script
import torch
from .detecting.detectors import ObjectDetector
import numpy as np

# Create detector
detector = ObjectDetector(model='mobilenet', device='mps')

# Check where model actually is
print("Model device:", next(detector.model.parameters()).device)
print("Expected: mps:0")

# Test single inference
dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
import time

t0 = time.time()
detections = detector.detect(dummy_frame)
t1 = time.time()

print(f"Time: {(t1-t0)*1000:.2f}ms")
print(f"Expected: ~20-50ms, Got: {(t1-t0)*1000:.2f}ms")