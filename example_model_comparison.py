"""
Example: Compare different YOLO models

This script demonstrates how easy it is to switch between models
using the ModelRegistry. Just change the model name!

Run with:
    source venv/bin/activate
    python example_model_comparison.py
"""

import sys
sys.path.insert(0, 'src')

from detecting.model_registry import ModelRegistry
import cv2
import time
import logging


def setup_logger():
    """Setup minimal logger."""
    logger = logging.getLogger('Comparison')
    logger.setLevel(logging.WARNING)  # Reduce spam
    return logger


def compare_models():
    """Compare different YOLO models on the same image."""
    print("\n" + "="*70)
    print("YOLO MODEL COMPARISON - How Easy Model Switching Is!")
    print("="*70)

    logger = setup_logger()

    # Load test image
    print("\nLoading test image...")
    cap = cv2.VideoCapture('data/TownCent.mp4')
    ret, image = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to load test image!")
        return

    print("✓ Image loaded")

    # Models to compare
    # Learning Point: Just change this list to try different models!
    models_to_test = [
        'yolov8n',  # Fastest
        'yolov8s',  # Balanced
        # Uncomment to try more:
        # 'yolov8m',  # More accurate
        # 'yolov7-tiny-onnx',  # ONNX version
    ]

    print(f"\nComparing {len(models_to_test)} models...")
    print("-"*70)

    results = {}

    for model_name in models_to_test:
        print(f"\n{model_name.upper()}:")
        print(f"  Loading model...")

        # Learning Point: THIS IS THE MAGIC!
        # Just change the model_name to switch models!
        detector = ModelRegistry.create(
            model_name=model_name,
            confidence_threshold=0.3,
            logger=logger
        )

        # Warm up (first run is always slower)
        print(f"  Warming up...")
        detector.detect(image)

        # Measure performance over multiple runs
        print(f"  Running 10 iterations...")
        times = []
        for i in range(10):
            start = time.time()
            detections = detector.detect(image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        results[model_name] = {
            'detections': len(detections),
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps': 1000 / avg_time
        }

        print(f"  ✓ Detections: {len(detections)}")
        print(f"  ✓ Average time: {avg_time:.2f} ms ({1000/avg_time:.1f} FPS)")
        print(f"  ✓ Range: {min_time:.2f} - {max_time:.2f} ms")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Model':<20} {'Detections':<15} {'Avg Time':<15} {'FPS':<10}")
    print("-"*70)

    for model, result in results.items():
        print(f"{model:<20} {result['detections']:<15} "
              f"{result['avg_time']:.2f} ms{'':<8} {result['fps']:.1f}")

    # Key takeaway
    print("\n" + "="*70)
    print("KEY TAKEAWAY")
    print("="*70)
    print("""
Now switching models is TRIVIAL:

Before (hardcoded):
    detector = YOLOv7ONNX(
        model_path='models/yolov7-tiny_640x640.onnx',
        confidence_threshold=0.3,
        ...
    )

After (registry):
    detector = ModelRegistry.create('yolov8n')

To try a different model, just change ONE word:
    detector = ModelRegistry.create('yolov8s')  # or yolov8m, yolov8l, etc.

Or edit config.yaml:
    detector:
      model: yolov8s  # Change this line and you're done!

This is the power of the Registry Pattern!
    """)

    print("="*70)

    # Show how to use with config
    print("\nBonus: Using with config.yaml")
    print("-"*70)
    print("""
1. Edit config.yaml:
   detector:
     model: yolov8s  # Change from yolov8n to yolov8s

2. Run:
   python -m src.main

3. That's it! No code changes needed!

Try it:
   - Edit config.yaml
   - Change 'yolov8n' to 'yolov8s'
   - Run: python -m src.main
   - See the difference!
    """)


if __name__ == '__main__':
    compare_models()
