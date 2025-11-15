# CLAUDE.md - AI Assistant Guide for GenTbD

This document provides comprehensive guidance for AI assistants working on the GenTbD (Generalized Tracking-by-Detection) codebase.

## Project Overview

**GenTbD** is a PyTorch-based object detection system for video processing with a flexible, extensible architecture. The project focuses on:

- Multi-model object detection (FasterRCNN, RetinaNet, MobileNet)
- Keypoint detection for human pose estimation
- ReID (Re-identification) feature extraction
- Interactive video processing with real-time visualization
- Hardware acceleration (CPU, CUDA, MPS)
- Extensible detector architecture for future tracking systems

**Current Version**: 0.1.0
**Python Requirement**: >=3.10
**Primary Dependencies**: PyTorch, torchvision, OpenCV, NumPy

---

## Repository Structure

```
GenTbD/
├── src/                           # Source code
│   ├── main.py                    # Entry point with CLI argument parsing
│   ├── config.py                  # YAML + CLI configuration management
│   ├── video_processor.py         # Video processing with interactive controls
│   ├── detecting/                 # Detection system (core module)
│   │   ├── detector.py            # Abstract base class for all detectors
│   │   ├── detection.py           # Dict-like container for detection results
│   │   ├── factory.py             # DetectorFactory for centralized instantiation
│   │   ├── feature_extractor.py   # Base class for feature extraction (ReID)
│   │   └── detectors/             # Concrete detector implementations
│   │       ├── object_detector.py # Object detection (FasterRCNN, RetinaNet)
│   │       ├── optimized_object_detector.py # GPU-optimized object detection
│   │       ├── keypoint_detector.py # Human pose keypoint detection
│   │       └── reid_detector.py   # ReID feature extraction (OSNet, ResNet50)
│   └── core/                      # Core data structures
│       ├── frame.py               # Frame abstraction with temporal metadata
│       └── properties/            # Detection properties
│           ├── bounding_box.py    # BoundingBox with IoU and visualization
│           ├── keypoints.py       # Keypoints container for pose data
│           └── embedding.py       # Embedding vector for ReID features
├── tests/                         # Unit tests (pytest)
│   ├── test_object_detector.py
│   ├── test_keypoint_detector.py
│   ├── test_reid_detector.py
│   ├── test_detector_factory.py
│   ├── test_config.py
│   ├── test_video_processor.py
│   ├── test_frame.py
│   ├── test_bounding_box.py
│   ├── test_keypoints.py
│   └── test_embedding.py
├── config/                        # YAML configuration files
│   └── default.yaml               # Default configuration
├── data/                          # Input video files
├── ARCHIVE/                       # Archived documentation and old code
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies (pytest, flake8, black)
├── pyproject.toml                 # Build configuration
├── README.md                      # User-facing documentation
└── CHANGELOG.md                   # Version history
```

---

## Architecture Patterns

### 1. Detection Pipeline (Three-Stage Pattern)

All detectors follow a standardized three-stage pipeline:

```python
# Stage 1: preprocess() - Convert image to tensor
# Stage 2: inference() - Run model forward pass
# Stage 3: postprocess() - Convert outputs to Detection objects
```

**Implementation Pattern**:
```python
class Detector(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image (H, W, 3) to model input tensor"""
        pass

    @abstractmethod
    def inference(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass"""
        pass

    @abstractmethod
    def postprocess(self, output: Any, image_shape: tuple) -> List[Detection]:
        """Convert raw output to Detection objects"""
        pass

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run full pipeline"""
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            output = self.inference(input_tensor)
        return self.postprocess(output, image.shape[:2])
```

**Key Points**:
- `detect()` is the public interface - never override this
- Override the three abstract methods to implement custom detectors
- Helper methods `_default_preprocess_pytorch()` and `_default_inference_pytorch()` are available
- All models run in `.eval()` mode with `torch.no_grad()`

### 2. Factory Pattern for Detector Creation

The `DetectorFactory` centralizes all detector instantiation logic:

```python
# Configuration-based creation
config = Config.from_yaml('config/default.yaml')
detector = DetectorFactory.create(config, logger)

# Factory handles:
# - Type-specific instantiation (object, keypoint, reid)
# - Model validation and auto-selection
# - Parameter extraction from nested config
# - Error handling and logging
```

**When to use**: Always use the factory in `main.py` and high-level code. Only instantiate detectors directly in tests or when you need fine-grained control.

### 3. Configuration System

Configuration uses a two-tier system: YAML files + CLI arguments

```python
# Load from YAML
config = Config.from_yaml('config/custom.yaml')

# Or use defaults
config = Config.from_default()  # Loads config/default.yaml

# Merge CLI arguments (they take precedence)
config.merge_args(args)

# Access with dot notation
model = config.get('detection.model', 'mobilenet')
conf_threshold = config.get('detection.conf_threshold', 0.5)

# Set values
config.set('detection.device', 'cuda')
```

**Configuration Structure** (see [config/default.yaml](config/default.yaml)):
- `video.*` - Video source, processing, and output settings
- `detection.*` - Detector type, model, device, confidence threshold
- `detection.keypoint.*` - Keypoint-specific settings
- `detection.reid.*` - ReID-specific settings
- `tracking.*` - Tracking settings (for future implementation)
- `logging.*` - Logging configuration

### 4. Detection Container Pattern

The `Detection` class is a dict-like container for flexible detection properties:

```python
detection = Detection({
    'bbox': BoundingBox(10, 20, 100, 200),
    'class_id': 1,
    'confidence': 0.95,
    'keypoints': Keypoints(...),      # Optional
    'embedding': Embedding(...)        # Optional
})

# Strict access (raises KeyError if missing)
bbox = detection['bbox']

# Safe access with default
score = detection.get('confidence', 0.0)

# Check existence
if 'keypoints' in detection:
    keypoints = detection['keypoints']
```

**Extensibility**: Detectors can add custom properties without modifying the Detection class. Standard properties: `bbox`, `class_id`, `confidence`, `keypoints`, `embedding`.

### 5. Frame Abstraction Pattern

The `Frame` class encapsulates video frames with temporal and spatial metadata:

```python
frame = Frame(
    data=image,           # np.ndarray (H, W, C) in BGR
    frame_id=42,          # Sequential frame number
    timestamp=1.4,        # Seconds since video start
    source_id="camera_1"  # Video source identifier
)

# Properties
frame.width, frame.height, frame.shape
frame.original_shape  # Before any scaling
frame.scale_factor    # Cumulative scale (1.0 = no scaling)

# Scaling (creates new Frame)
half_size = frame.scaled(0.5)
quarter_size = half_size.scaled(0.5)  # Cumulative: scale_factor = 0.25

# Deep copy
copy = frame.copy()
```

**When to use**: Use Frame objects when temporal context is needed (tracking, ReID, multi-camera). For simple detection, raw numpy arrays are fine.

---

## Code Conventions

### Import Organization

```python
# Standard library
import os
from typing import List, Optional

# Third-party
import cv2
import numpy as np
import torch

# Local absolute imports (use src. prefix)
from src.detecting import Detector, Detection
from src.core import Frame
from src.core.properties import BoundingBox
```

### Naming Conventions

- **Classes**: PascalCase (`ObjectDetector`, `BoundingBox`, `Frame`)
- **Functions/Methods**: snake_case (`detect()`, `preprocess()`, `scaled()`)
- **Constants**: UPPER_SNAKE_CASE (`FPS_WINDOW`, `TEXT_COLOR`)
- **Private methods**: Leading underscore (`_setup_video()`, `_default_preprocess_pytorch()`)

### Docstrings

Use Google-style docstrings with type hints:

```python
def detect(self, image: np.ndarray) -> List[Detection]:
    """Run full detection pipeline.

    Args:
        image: Input image in BGR format (H, W, 3)

    Returns:
        List of Detection objects

    Raises:
        ValueError: If image shape is invalid

    Example:
        >>> detector = ObjectDetector(model='resnet50')
        >>> detections = detector.detect(image)
    """
```

### Type Hints

- Use type hints for all function signatures
- Import from `typing` module: `List`, `Dict`, `Optional`, `Any`, `Tuple`
- Use `np.ndarray` for NumPy arrays
- Use `torch.Tensor` for PyTorch tensors

### Error Handling

```python
# Validate inputs early
if image is None or not isinstance(image, np.ndarray):
    raise ValueError("Image must be a numpy array")

# Provide helpful error messages
if detector_type not in ['object', 'keypoint', 'reid']:
    raise ValueError(
        f"Unknown detector type: '{detector_type}'. "
        f"Valid options: 'object', 'keypoint', 'reid'"
    )

# Log errors with context
logger.error(f"Failed to load detector: {e}", exc_info=verbose)
```

---

## Development Workflow

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt      # Production
pip install -r requirements-dev.txt  # Development

# Run tests
pytest

# Check test coverage
pytest --cov=src --cov-report=html

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/
```

### Running the Application

```bash
# Basic usage with default video
python -m src.main --video data/TownCent.mp4

# Webcam with custom settings
python -m src.main --webcam --model mobilenet --conf 0.7 --device mps

# Using configuration file
python -m src.main --config config/custom.yaml

# CLI args override config file
python -m src.main --config config/default.yaml --model resnet50 --device cuda
```

### Interactive Controls

When running the video processor:
- `c` - Toggle continuous/step mode
- `SPACE` - Next frame (in step mode)
- `s` - Save current frame as image
- `q` - Quit

### Testing Strategy

**Test Structure**:
- One test file per module (`test_object_detector.py`, `test_frame.py`)
- Use pytest fixtures for common setup
- Mock external dependencies (models, file I/O)

**Example Test**:
```python
def test_object_detector_init():
    """Test ObjectDetector initialization."""
    detector = ObjectDetector(model='mobilenet', device='cpu')
    assert detector.device == 'cpu'
    assert detector.model is not None
```

**Running Tests**:
```bash
pytest                              # All tests
pytest tests/test_object_detector.py  # Single file
pytest -v                           # Verbose
pytest --cov=src                    # With coverage
```

---

## Common Tasks for AI Assistants

### Adding a New Detector

1. **Create detector class** in `src/detecting/detectors/`:
   ```python
   from ..detector import Detector
   from ..detection import Detection

   class MyDetector(Detector):
       def preprocess(self, image):
           # Convert to tensor

       def inference(self, input_tensor):
           # Run model

       def postprocess(self, output, image_shape):
           # Convert to Detection objects
   ```

2. **Update factory** in `src/detecting/factory.py`:
   ```python
   elif detector_type == 'my_detector':
       return DetectorFactory._create_my_detector(config, device, conf_threshold, logger)
   ```

3. **Add tests** in `tests/test_my_detector.py`

4. **Update configuration** in `config/default.yaml`

5. **Document** in README.md

### Modifying Detection Properties

1. **Add property class** in `src/core/properties/` (if needed):
   ```python
   class MyProperty:
       def __init__(self, data):
           self.data = data
   ```

2. **Update detector's postprocess()** to include new property:
   ```python
   detection = Detection({
       'bbox': BoundingBox(...),
       'my_property': MyProperty(...)
   })
   ```

3. **Add tests** for the new property

4. **Update documentation** to mention the new property

### Adding Configuration Options

1. **Update `config/default.yaml`** with new settings:
   ```yaml
   my_module:
     setting: value
   ```

2. **Update `Config.merge_args()`** if adding CLI argument:
   ```python
   if hasattr(args, 'my_setting') and args.my_setting:
       self.set('my_module.setting', args.my_setting)
   ```

3. **Update `main.py` argument parser** if adding CLI option:
   ```python
   parser.add_argument('--my-setting', type=str, help='Description')
   ```

4. **Access in code**:
   ```python
   setting = config.get('my_module.setting', default_value)
   ```

### Performance Optimization

**Video Pipeline Optimizations**:
1. Frame skipping (`--skip-frames N`) - Process every Nth frame for speedup
2. Frame resizing (`--max-dimension 640`) - Reduce input resolution
3. Model selection (mobilenet > resnet50) - Use lighter models

**GPU Acceleration**:
Use `OptimizedObjectDetector` for MPS/CUDA devices:
- GPU-accelerated preprocessing
- GPU-based filtering
- Reduced CPU↔GPU transfers

**Usage**:
```python
from src.detecting.detectors import OptimizedObjectDetector

detector = OptimizedObjectDetector(
    model='mobilenet',
    device='mps',  # or 'cuda'
    conf_threshold=0.5,
    gpu_preprocess=True,
    gpu_filter=True
)
```

**Known Performance Issues**:

*Apple Silicon (M1/M2/M3)*:
- MPS device has severe performance degradation with non-contiguous tensors
- PyTorch's `permute()` creates non-contiguous tensors, causing 1900x slowdown
- Workaround: Call `.contiguous()` after `permute()`, but may still have issues
- CPU on Apple Silicon lacks MKLDNN/oneDNN optimizations (Intel-specific)
- Recommendation: Test both CPU and MPS devices to find better performer for your hardware

*General*:
- First model inference is slower (JIT compilation, shader compilation)
- MPS may have compatibility issues with certain operations
- Consider lighter models (YOLO, SSD) if performance is critical

---

## Critical Information

### Device Handling

```python
# Detector automatically moves model to device
detector = ObjectDetector(model='resnet50', device='mps')

# Device options:
# - 'cpu': Universal, slow
# - 'cuda': NVIDIA GPU, fastest
# - 'mps': Apple Silicon, ~8-15 FPS
```

### Video Source Handling

The `VideoProcessor` supports multiple source types:

```python
# Video file
processor = VideoProcessor(source='data/video.mp4', detector=detector)

# Webcam (camera index)
processor = VideoProcessor(source=0, detector=detector)

# RTSP stream
processor = VideoProcessor(source='rtsp://camera/stream', detector=detector)
```

### COCO Class IDs

Common class IDs for filtering (see README.md for full list):
- `1` - person
- `2` - bicycle
- `3` - car
- `7` - train
- `16` - dog
- `17` - cat

### Model Compatibility

| Detector Type | Supported Models |
|--------------|------------------|
| `object` | `resnet50`, `mobilenet`, `retinanet` |
| `keypoint` | `resnet50` only |
| `reid` | `osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`, `resnet50` |

---

## Git Workflow

**Current Branch**: `base`
**Main Branch**: `main` (for PRs)

**Commit Message Style** (based on recent commits):
- Use present tense: "Add feature" not "Added feature"
- Be descriptive: "Add ReID detector for appearance-based feature extraction"
- Start with action verb: Add, Update, Fix, Refactor, Extract

**Example Workflow**:
```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "Add support for multi-camera tracking"

# Push to base branch
git push origin base

# Or merge to main and push
git checkout main
git merge base
git push origin main
```

---

## Future Development (Roadmap)

From README.md and CHANGELOG.md:

- [ ] **Tracking System**: Multi-object tracking with ID persistence (ByteTrack, SORT, DeepSORT)
- [ ] **Batch Processing**: Multi-frame batch inference for improved throughput
- [ ] **Model Zoo**: Additional pre-trained detector models
- [ ] **Multi-camera Support**: Cross-camera tracking with ReID
- [ ] **API Documentation**: Complete API reference and tutorials

---

## Troubleshooting

### Common Issues

1. **Import errors**: Always use `python -m src.main` not `python src/main.py`
2. **Device errors**: Check if `mps` or `cuda` is available: `torch.backends.mps.is_available()`
3. **Video not found**: Ensure video file exists in `data/` directory
4. **Slow performance**: Use `--model mobilenet --device mps --skip-frames 3 --max-dimension 640`
5. **Test failures**: Run `pytest -v` for detailed output

### Logging

Enable verbose logging for debugging:
```bash
python -m src.main --video data/video.mp4 --verbose
```

---

## Quick Reference

**Key Files**:
- Entry point: [src/main.py](src/main.py)
- Detection pipeline: [src/detecting/detector.py](src/detecting/detector.py)
- Factory: [src/detecting/factory.py](src/detecting/factory.py)
- Configuration: [src/config.py](src/config.py)
- Video processing: [src/video_processor.py](src/video_processor.py)

**Key Classes**:
- `Detector` - Abstract base class for all detectors
- `ObjectDetector` - Object detection implementation
- `KeypointDetector` - Keypoint detection implementation
- `ReIDDetector` - ReID feature extraction
- `Detection` - Dict-like container for detection results
- `Frame` - Frame abstraction with metadata
- `BoundingBox` - Bounding box with IoU and visualization

**Key Patterns**:
- Three-stage detection pipeline (preprocess → inference → postprocess)
- Factory pattern for detector creation
- Configuration with YAML + CLI merge
- Dict-like Detection container for extensibility
- Frame abstraction for temporal context

---

## Additional Resources

- [README.md](README.md) - User-facing documentation
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [Tracking_paper.pdf](Tracking_paper.pdf) - Research paper reference

---

**Last Updated**: 2025-11-15
**Version**: 0.1.0
