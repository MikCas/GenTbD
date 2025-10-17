"""
Test script for Model and Tracker registries.

This script tests that:
1. ModelRegistry can create different detectors
2. TrackerRegistry can create trackers
3. Switching models via registry works correctly
4. Configuration-driven approach works

Run with:
    source venv/bin/activate
    python test_registry.py
"""

import sys
sys.path.insert(0, 'src')

from detecting.model_registry import ModelRegistry
from tracking.tracker_registry import TrackerRegistry
import cv2
import logging


def setup_logger():
    """Setup simple logger."""
    logger = logging.getLogger('Test_Registry')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def test_model_registry():
    """Test ModelRegistry functionality."""
    print("\n" + "="*60)
    print("TESTING MODEL REGISTRY")
    print("="*60)

    logger = setup_logger()

    # Test 1: List available models
    print("\n1. Available models:")
    models = ModelRegistry.list_models()
    for name, desc in models.items():
        print(f"   - {name}: {desc}")

    # Test 2: Get model info
    print("\n2. Getting info for 'yolov8n':")
    info = ModelRegistry.get_model_info('yolov8n')
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Test 3: Create detector
    print("\n3. Creating YOLOv8n detector...")
    detector = ModelRegistry.create('yolov8n', logger=logger)
    print(f"   ✓ Created: {type(detector).__name__}")

    # Test 4: Create different detector
    print("\n4. Creating YOLOv8s detector...")
    detector2 = ModelRegistry.create('yolov8s', confidence_threshold=0.5, logger=logger)
    print(f"   ✓ Created: {type(detector2).__name__}")

    # Test 5: Test detection
    print("\n5. Testing detection on sample image...")
    image_path = 'data/TownCent.mp4'
    cap = cv2.VideoCapture(image_path)
    ret, image = cap.read()
    cap.release()

    if ret:
        detections = detector.detect(image)
        print(f"   ✓ Found {len(detections)} objects")
        if detections:
            print(f"   Sample detection: confidence={detections[0].confidence_score:.3f}")
    else:
        print("   ✗ Failed to load test image")

    return True


def test_tracker_registry():
    """Test TrackerRegistry functionality."""
    print("\n" + "="*60)
    print("TESTING TRACKER REGISTRY")
    print("="*60)

    logger = setup_logger()

    # Test 1: List available trackers
    print("\n1. Available trackers:")
    trackers = TrackerRegistry.list_trackers()
    for name, desc in trackers.items():
        print(f"   - {name}: {desc}")

    # Test 2: Get tracker info
    print("\n2. Getting info for 'simple':")
    info = TrackerRegistry.get_tracker_info('simple')
    for key, value in info.items():
        if key != 'class':  # Skip class object
            print(f"   {key}: {value}")

    # Test 3: Create tracker
    print("\n3. Creating SimpleTracker...")
    tracker = TrackerRegistry.create('simple', logger=logger)
    print(f"   ✓ Created: {type(tracker).__name__}")

    # Test 4: Create tracker with custom params
    print("\n4. Creating tracker with custom parameters...")
    tracker2 = TrackerRegistry.create(
        'simple',
        max_age=50,
        min_hits=2,
        iou_threshold=0.4,
        logger=logger
    )
    print(f"   ✓ Created: {type(tracker2).__name__}")

    return True


def test_config_loading():
    """Test loading configuration from YAML."""
    print("\n" + "="*60)
    print("TESTING CONFIG LOADING")
    print("="*60)

    import yaml
    from pathlib import Path

    config_path = Path('config.yaml')
    if not config_path.exists():
        print("   ✗ config.yaml not found!")
        return False

    print("\n1. Loading config.yaml...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("   ✓ Config loaded successfully")

    print("\n2. Config contents:")
    print(f"   Detector model: {config['detector']['model']}")
    print(f"   Detector confidence: {config['detector']['confidence_threshold']}")
    print(f"   Tracker name: {config['tracker']['name']}")
    print(f"   Tracker max_age: {config['tracker']['max_age']}")
    print(f"   Video input: {config['video']['input']}")

    print("\n3. Creating detector from config...")
    logger = setup_logger()
    detector = ModelRegistry.create(
        model_name=config['detector']['model'],
        confidence_threshold=config['detector']['confidence_threshold'],
        iou_threshold=config['detector']['iou_threshold'],
        classes=config['detector']['classes'],
        device=config['detector']['device'],
        logger=logger
    )
    print(f"   ✓ Created detector: {type(detector).__name__}")

    return True


def test_model_switching():
    """Test switching between different models."""
    print("\n" + "="*60)
    print("TESTING MODEL SWITCHING")
    print("="*60)

    logger = setup_logger()

    # Load test image
    print("\n1. Loading test image...")
    image_path = 'data/TownCent.mp4'
    cap = cv2.VideoCapture(image_path)
    ret, image = cap.read()
    cap.release()

    if not ret:
        print("   ✗ Failed to load test image")
        return False

    print("   ✓ Image loaded")

    # Test different models
    models_to_test = ['yolov8n']  # Start with just one for speed
    results = {}

    for model_name in models_to_test:
        print(f"\n2. Testing {model_name}...")

        # Create detector
        detector = ModelRegistry.create(model_name, logger=logger)

        # Warm up
        detector.detect(image)

        # Measure performance
        import time
        start = time.time()
        detections = detector.detect(image)
        elapsed = (time.time() - start) * 1000

        results[model_name] = {
            'detections': len(detections),
            'time_ms': elapsed
        }

        print(f"   ✓ Detections: {len(detections)}")
        print(f"   ✓ Time: {elapsed:.2f} ms")

    print("\n3. Summary:")
    for model, result in results.items():
        print(f"   {model}: {result['detections']} objects in {result['time_ms']:.2f}ms")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REGISTRY SYSTEM TEST SUITE")
    print("="*60)

    tests = [
        ("Model Registry", test_model_registry),
        ("Tracker Registry", test_tracker_registry),
        ("Config Loading", test_config_loading),
        ("Model Switching", test_model_switching),
    ]

    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            print(f"\n   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "✗ ERROR"

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{result} - {name}")

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("✓ ModelRegistry makes it easy to create different detectors")
    print("✓ TrackerRegistry makes it easy to create different trackers")
    print("✓ Switch models by just changing the model name!")
    print("✓ Configuration-driven design separates config from code")
    print("\nNow you can:")
    print("1. Edit config.yaml to change models")
    print("2. Run 'python -m src.main' to use config-driven approach")
    print("3. Add new models with ModelRegistry.register_model()")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
