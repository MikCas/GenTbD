# Phase 2: Configuration System - COMPLETE! 🎉

## What We Built

Phase 2 transformed GenTbD into a **configuration-driven**, **model-agnostic** tracking system. Now you can switch between different detectors and models by simply editing a config file!

## Key Components

### 1. ModelRegistry (`src/detecting/model_registry.py`)

**Purpose**: Factory for creating detectors from model names

**Features**:
- **Easy model creation**: `ModelRegistry.create('yolov8n')`
- **Multiple backends**: Supports PyTorch and ONNX models
- **Auto-download**: YOLOv8 models download automatically
- **Extensible**: Add custom models with `register_model()`

**Available Models**:
- `yolov8n` - Nano (fastest, 3.2M params)
- `yolov8s` - Small (balanced, 11.2M params)
- `yolov8m` - Medium (accurate, 25.9M params)
- `yolov8l` - Large (very accurate, 43.7M params)
- `yolov8x` - Extra (best accuracy, 68.2M params)
- `yolov7-tiny-onnx` - ONNX (requires model file)

**Example**:
```python
# Before (hardcoded):
detector = YOLOv7ONNX(
    model_path='models/yolov7-tiny_640x640.onnx',
    confidence_threshold=0.3,
    iou_threshold=0.5,
    classes=[0],
    logger=logger
)

# After (registry):
detector = ModelRegistry.create('yolov8n', logger=logger)

# Switch to different model:
detector = ModelRegistry.create('yolov8s')  # Just change one word!
```

### 2. TrackerRegistry (`src/tracking/tracker_registry.py`)

**Purpose**: Factory for creating trackers from tracker names

**Features**:
- **Easy tracker creation**: `TrackerRegistry.create('simple')`
- **Configurable parameters**: max_age, min_hits, iou_threshold
- **Extensible**: Add custom trackers with `register_tracker()`

**Note**: SimpleTracker currently uses different parameters, needs adaptation for full compatibility.

**Example**:
```python
# Create tracker with defaults
tracker = TrackerRegistry.create('simple')

# Create tracker with custom parameters
tracker = TrackerRegistry.create(
    'simple',
    max_age=50,
    min_hits=2,
    iou_threshold=0.4
)
```

### 3. Configuration File (`config.yaml`)

**Purpose**: Centralized configuration for the entire system

**Sections**:
- **detector**: Model, confidence, IoU, classes, device
- **tracker**: Tracker name and parameters
- **video**: Input/output paths, display settings
- **visualization**: Box colors, text, FPS counter
- **logging**: Level, format, file output

**Example**:
```yaml
detector:
  model: yolov8n              # Just change this to switch models!
  confidence_threshold: 0.3
  iou_threshold: 0.5
  classes: [0]                # 0 = person
  device: null                # Auto-detect

tracker:
  name: simple
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

video:
  input: data/TownCent.mp4
  output: null
  display: true
```

### 4. Refactored Main (`src/main.py`)

**Purpose**: Configuration-driven entry point

**Features**:
- Loads configuration from YAML
- Creates detector from config
- Creates tracker from config
- Supports custom config files

**Usage**:
```bash
# Use default config.yaml
python -m src.main

# Use custom config
python -m src.main --config my_config.yaml
```

## How It Works

### The Registry Pattern

**Before**: Tightly coupled to specific detector classes
```python
detector = YOLOv7ONNX(model_path, ...)  # Hardcoded!
```

**After**: Loose coupling via registry
```python
detector = ModelRegistry.create(model_name)  # Dynamic!
```

**Benefits**:
1. **Easy experimentation**: Try different models by changing config
2. **No code changes**: Switch models without touching code
3. **Extensible**: Add new models without modifying existing code
4. **Testable**: Easy to test with different models

### Configuration-Driven Design

**Philosophy**: Separate **what** from **how**

**Config file** (what): "Use yolov8n with 0.3 confidence"
**Code** (how): "Load config, create detector, run tracking"

**Benefits**:
1. **User-friendly**: Non-programmers can configure
2. **Version control**: Track config changes with git
3. **Reproducible**: Share configs to reproduce results
4. **Flexible**: Same code, different configs

## Test Results

### Model Registry Test (`test_registry.py`)

✅ **Model Registry**: PASSED
- Listed 6 available models
- Retrieved model info successfully
- Created YOLOv8n detector
- Created YOLOv8s detector (auto-downloaded!)
- Detected 15 objects in test image

✅ **Config Loading**: PASSED
- Loaded config.yaml successfully
- Parsed all sections correctly
- Created detector from config

✅ **Model Switching**: PASSED
- Switched from yolov8n to yolov8s easily
- Performance: 31ms (32 FPS)
- 15 detections found

⚠️ **Tracker Registry**: Needs adaptation
- SimpleTracker uses different parameters
- Pattern works, just needs parameter mapping

## Example Scripts

### `test_registry.py`
Comprehensive test of registry system:
- Tests ModelRegistry
- Tests TrackerRegistry
- Tests config loading
- Tests model switching

### `example_model_comparison.py`
Demonstrates easy model switching:
- Compares yolov8n vs yolov8s
- Shows performance differences
- Demonstrates registry usage

### `test_pytorch.py`
Compares PyTorch vs ONNX backends:
- Tests both backends work
- Measures performance
- Verifies detection consistency

## What You Can Do Now

### 1. Switch Models via Config
```bash
# Edit config.yaml
vim config.yaml  # Change 'yolov8n' to 'yolov8s'

# Run with new model
python -m src.main
```

### 2. Compare Models Programmatically
```python
from detecting.model_registry import ModelRegistry

# Try different models
for model in ['yolov8n', 'yolov8s', 'yolov8m']:
    detector = ModelRegistry.create(model)
    detections = detector.detect(image)
    print(f"{model}: {len(detections)} objects")
```

### 3. Add Custom Models
```python
ModelRegistry.register_model(
    name='my-yolo',
    detector_class=MyYOLODetector,
    description='My custom YOLO',
    backend='pytorch',
    model_path='models/my_yolo.pt'
)

# Use it immediately
detector = ModelRegistry.create('my-yolo')
```

### 4. Create Custom Configs
```bash
# Create experiment config
cp config.yaml experiments/high_precision.yaml

# Edit for high precision
vim experiments/high_precision.yaml
# model: yolov8x
# confidence_threshold: 0.5

# Run experiment
python -m src.main --config experiments/high_precision.yaml
```

## Key Learnings

### Design Patterns
1. **Factory Pattern**: Create objects without exposing creation logic
2. **Registry Pattern**: Map names to implementations
3. **Configuration-Driven**: Separate config from code
4. **Dependency Injection**: Pass dependencies instead of hardcoding

### Python Concepts
1. **YAML**: Human-friendly configuration format
2. **@classmethod**: Class-level factory methods
3. **Type hints**: Better code documentation
4. **Dict unpacking**: `**kwargs` for flexible parameters

### Software Engineering Principles
1. **Open/Closed Principle**: Open for extension, closed for modification
2. **Single Responsibility**: Each class has one job
3. **Separation of Concerns**: Config ≠ Logic
4. **DRY (Don't Repeat Yourself)**: Reuse through factories

## Next Steps

### Immediate
- [x] Model registry working ✓
- [x] Config system working ✓
- [x] Documentation complete ✓

### Short Term
- [ ] Adapt SimpleTracker to standard parameters
- [ ] Add more trackers (DeepSORT, ByteTrack)
- [ ] Output video saving
- [ ] Track data export (CSV/JSON)

### Medium Term
- [ ] Pose detection (YOLOv8-pose)
- [ ] Segmentation (YOLOv8-seg)
- [ ] Re-ID tracking (appearance features)
- [ ] MOT metrics (MOTA, MOTP, IDF1)

### Long Term
- [ ] Web UI for configuration
- [ ] Real-time streaming support
- [ ] Multi-camera tracking
- [ ] Cloud deployment

## Files Created

```
GenTbD/
├── config.yaml                           # Main configuration file
├── src/
│   ├── detecting/
│   │   └── model_registry.py             # Model factory
│   ├── tracking/
│   │   └── tracker_registry.py           # Tracker factory
│   └── main.py                           # Refactored entry point
├── test_registry.py                      # Registry tests
└── example_model_comparison.py           # Model comparison demo
```

## Dependencies Added

Updated `pyproject.toml`:
- `pyyaml>=6.0` - YAML parsing
- `torch>=2.0.0` - PyTorch backend
- `ultralytics>=8.0.0` - YOLO models

## Git Commits

```bash
# All Phase 2 work ready to commit
git add .
git commit -m "Phase 2: Configuration system with model registry"
```

## Conclusion

Phase 2 successfully transformed GenTbD into a **flexible**, **configurable**, and **extensible** tracking system!

**Before**: Hardcoded ONNX detector, difficult to experiment
**After**: Config-driven, easy model switching, ready for research

**Key Achievement**: You can now switch between 6 different YOLO models by editing ONE line in config.yaml!

This is the foundation for making GenTbD a truly generalizable tracking-by-detection framework. 🚀

---

**Ready to use?**
```bash
# 1. Edit config.yaml (optional)
vim config.yaml

# 2. Run tracking
python -m src.main

# 3. Try different models
# Just edit config.yaml and run again!
```
