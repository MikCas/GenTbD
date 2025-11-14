# GenTbD: Generalized Tracking-by-Detection - AI Assistant Guide

## Project Overview

**GenTbD** is an object detection system for video processing built on PyTorch and torchvision. It provides a flexible, extensible framework for detection-based video analysis with support for multiple detection models, keypoint detection, and appearance-based re-identification (ReID).

**Current Version**: 0.1.0  
**Language**: Python 3.10+  
**Key Dependencies**: PyTorch, torchvision, OpenCV, torchreid, numpy, pyyaml

---

## Repository Structure

```
GenTbD/
├── src/                              # Main source code
│   ├── main.py                       # Entry point with CLI argument handling
│   ├── config.py                     # Configuration management (YAML + CLI args)
│   ├── video_processor.py            # Core video processing pipeline
│   ├── core/                         # Core domain abstractions
│   │   ├── frame.py                  # Frame abstraction with metadata
│   │   └── properties/               # Detection property classes
│   │       ├── bounding_box.py       # BoundingBox with IoU and scaling
│   │       ├── keypoints.py          # 17-point COCO keypoints
│   │       └── embedding.py          # ReID embedding vectors
│   └── detecting/                    # Detection system
│       ├── detector.py               # Abstract base class (Detector ABC)
│       ├── detection.py              # Detection result container
│       ├── feature_extractor.py      # Abstract base for feature extraction
│       ├── factory.py                # DetectorFactory for instantiation
│       └── detectors/                # Concrete detector implementations
│           ├── object_detector.py    # FasterRCNN/RetinaNet object detection
│           ├── keypoint_detector.py  # KeypointRCNN human pose estimation
│           └── reid_detector.py      # ReID feature extraction (OSNet, ResNet50)
├── tests/                            # Comprehensive test suite (11 test files)
│   ├── test_config.py
│   ├── test_frame.py
│   ├── test_detector_factory.py
│   ├── test_object_detector.py
│   ├── test_keypoint_detector.py
│   ├── test_reid_detector.py
│   ├── test_bounding_box.py
│   ├── test_keypoints.py
│   ├── test_embedding.py
│   ├── test_video_processor.py
│   └── test_video_processor.py
├── config/                           # Configuration files
│   └── default.yaml                  # Default system configuration
├── experiments/                      # Performance benchmarks and optimization docs
│   ├── test_performance.py
│   └── PERFORMANCE_OPTIMIZATION.md   # Comprehensive optimization guide (270x speedup)
├── requirements.txt                  # Core dependencies
├── requirements-dev.txt              # Development dependencies (pytest, flake8, black)
├── pyproject.toml                    # Package metadata and setuptools config
├── README.md                         # User-facing documentation
├── CHANGELOG.md                      # Version history
└── Tracking_paper.pdf                # Academic reference material
```

---

## Architecture & Design Patterns

### 1. Detection Pipeline Architecture

The system follows a **three-stage detection pipeline**:

```
Image (numpy array)
    ↓
[preprocess]  → Convert BGR → RGB, normalize, convert to tensor (CHW format)
    ↓
[inference]   → Run model forward pass with batch dimension
    ↓
[postprocess] → Filter by confidence, convert to Detection objects
    ↓
List[Detection]
```

**Key Classes**:
- `Detector` (abstract base class): Defines preprocess/inference/postprocess contract
- `ObjectDetector`: Implements FasterRCNN/RetinaNet for COCO object detection
- `KeypointDetector`: Implements KeypointRCNN for human pose estimation (17 keypoints)
- `ReIDDetector` (alias `ReIDExtractor`): Implements OSNet/ResNet50 for appearance features

### 2. Factory Pattern

**`DetectorFactory`** centralizes detector instantiation:

```python
# Type-based factory
detector = DetectorFactory.create(config, logger)
# Automatically creates ObjectDetector, KeypointDetector, or ReIDDetector
# based on config['detection.type']
```

**Benefits**:
- Decouples detector instantiation from usage
- Centralizes validation logic
- Easy to add new detector types
- CLI args override config file values

### 3. Detection Container (Dict-like Interface)

`Detection` class provides dict-like access to detection properties:

```python
det = Detection({'bbox': BoundingBox(...), 'class_id': 1, 'confidence': 0.95})
det['bbox']              # Strict access (raises KeyError if missing)
det.get('confidence', 0.0)  # Safe access with default
'bbox' in det            # Membership check
det.keys()               # List all properties
```

**Extensible**: Can store any properties (bbox, confidence, class_id, keypoints, embedding, custom)

### 4. Property Classes (Core Domain Objects)

#### BoundingBox
```python
bbox = BoundingBox(x1, y1, x2, y2)  # Immutable, stores as tensor
x1, y1, x2, y2 = bbox.xyxy           # Get coordinates
iou = bbox.iou(other_bbox)           # IoU calculation
bbox.draw(image, color=(0, 255, 0))  # Visualization
scaled = bbox.scale(0.5)             # Non-mutating scale
```

**Design**: Immutable (stores as PyTorch tensor internally), enables efficient IoU computation

#### Keypoints
```python
kpts = Keypoints(array_17x3, scores_17)  # COCO 17 points: nose, eyes, shoulders, etc.
kpts.keypoints                # Get [x, y, visibility] for each point
kpts.scores                   # Get confidence scores
kpts.num_visible              # Count visible points
kpts.get_keypoint('nose')     # Access by name
kpts.is_visible(5)            # Check visibility
kpts.draw(image)              # Visualization with skeleton
```

**Design**: Validates 17-point COCO format, provides semantic access by name

#### Embedding
```python
emb = Embedding(vector, model='osnet_x1_0')  # Auto L2-normalized
distance = emb.cosine_distance(other_emb)    # ReID matching metric
similarity = emb.similarity(other_emb)       # Cosine similarity [-1, 1]
assert emb.is_normalized()                   # Verify unit norm
```

**Design**: L2-normalized unit vectors, enables efficient cosine distance matching

### 5. Frame Abstraction

`Frame` provides temporal and spatial metadata:

```python
frame = Frame(
    data=np.ndarray,      # Raw video frame (H, W, C) BGR
    frame_id=42,          # Sequential frame number (0-indexed)
    timestamp=1.4,        # Time in seconds since start
    source_id='cam_1',    # Video source identifier (for multi-camera)
    scale_factor=1.0      # Cumulative scaling applied
)

# Lazy evaluation of dimensions
frame.height, frame.width, frame.channels  # Properties
frame.shape                                 # Tuple (H, W, C)

# Scaling creates new Frame with metadata preserved
scaled = frame.scaled(0.5)
scaled.scale_factor == 0.5
scaled.original_shape == frame.original_shape  # Original preserved
```

**Design**: Immutable for safety, cumulative scale tracking for downstream processing

### 6. Configuration Management

`Config` class merges YAML files with CLI arguments:

```python
config = Config.from_yaml('config/default.yaml')  # Load from file
config = Config.from_default()                     # Load bundled default
config.get('detection.model', 'mobilenet')         # Dot notation access
config.set('detection.device', 'cuda')             # Set value
config.merge_args(args)                            # CLI args override
```

**Hierarchy**: Default YAML → Custom YAML → CLI arguments (each level overrides previous)

---

## Key Modules & Classes

### Core Module (`src/core/`)

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Frame` | Video frame with metadata | `scaled()`, `copy()`, properties for height/width |
| `BoundingBox` | Immutable bounding box | `iou()`, `draw()`, `scale()` |
| `Keypoints` | 17-point COCO keypoints | `draw()`, `scale()`, `get_keypoint()`, `is_visible()` |
| `Embedding` | ReID feature vector | `cosine_distance()`, `similarity()`, `is_normalized()` |

### Detection Module (`src/detecting/`)

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Detector` (ABC) | Base detector interface | `detect()`, `preprocess()`, `inference()`, `postprocess()` |
| `ObjectDetector` | FasterRCNN/RetinaNet object detection | Inherits from Detector, supports resnet50/mobilenet/retinanet |
| `KeypointDetector` | KeypointRCNN human pose | Inherits from Detector, outputs 17 keypoints |
| `FeatureExtractor` (ABC) | Base feature extractor | `extract()`, `extract_batch()`, `enhance()` |
| `ReIDDetector`/`ReIDExtractor` | OSNet/ResNet50 ReID features | Uses torchreid internally, outputs 512-D embeddings |
| `Detection` | Result container | Dict-like interface (`__getitem__`, `get()`, `keys()`) |
| `DetectorFactory` | Detector instantiation | `create()` static method with type dispatch |

### Video Processing (`src/video_processor.py`)

**VideoProcessor** class orchestrates detection and visualization:

```
_setup_video()          → Initialize capture, set video properties
    ↓
_process_loop()         → Main detection and display loop
    ├─ _detect_objects()   → Run detection, handle frame scaling
    ├─ _scale_detections() → Scale bboxes/keypoints back
    ├─ _render_frame()     → Visualize detections and keypoints
    ├─ _draw_overlay()     → FPS, mode, detection count
    ├─ _show_frame()       → Display to window
    ├─ _handle_input()     → Keyboard controls (c/SPACE/s/q)
    └─ Optional: _setup_output_writer() → Save video
    ↓
_cleanup()              → Release resources, print summary
```

**Key Features**:
- Frame skipping (process every Nth frame)
- Frame resizing for speed
- Interactive controls
- FPS tracking (rolling average)
- Optional output saving

### Configuration (`src/config.py`)

Structure:
```yaml
video:
  source: "data/TownCent.mp4"  # File path, camera index, or RTSP URL
  max_dimension: 640            # Resize before detection
  skip_frames: 1                # Process every Nth frame
  save_output: false
  output_path: null

detection:
  type: "object"                # "object", "keypoint", or "reid"
  model: "mobilenet"            # resnet50, mobilenet, retinanet (object)
  device: "cpu"                 # cpu, mps, cuda
  conf_threshold: 0.5
  classes: [1, 2, 3]            # Filter by class IDs

  keypoint:                      # Keypoint-specific settings
    keypoint_threshold: 0.5
    draw_skeleton: true

  reid:                          # ReID-specific settings
    embedding_dim: 512

tracking:                        # Future tracking system
  enabled: false
  tracker_type: "bytetrack"

logging:
  verbose: false
```

### Main Entry Point (`src/main.py`)

**Argument Parsing**:
- `--config`: Path to YAML config file
- `--video`: Video file path (default: data/TownCent.mp4)
- `--webcam [index]`: Use webcam (default index 0)
- `--detector-type`: object|keypoint|reid
- `--model`: resnet50|mobilenet|retinanet (object); resnet50 (keypoint)
- `--conf`: Confidence threshold (0.0-1.0)
- `--device`: cpu|mps|cuda
- `--classes`: Class IDs to detect (COCO classes)
- `--max-dimension`: Resize frames to max dimension
- `--skip-frames`: Process every Nth frame
- `--save-output`: Save output video
- `--output`: Output video path
- `--verbose`: Verbose logging

**Execution Flow**:
1. Parse CLI arguments
2. Load configuration (YAML + CLI override)
3. Create detector via DetectorFactory
4. Create VideoProcessor with detector
5. Run processor (main event loop)

---

## Detector Types & Models

### ObjectDetector (Object Detection)

**Supported Models**:
- `resnet50`: FasterRCNN with ResNet50 backbone (44M params, ~0.5 FPS CPU)
- `mobilenet`: FasterRCNN with MobileNetV3 backbone (5.5M params, ~7 FPS CPU)
- `retinanet`: RetinaNet with ResNet50 (balanced speed/accuracy)

**Output**: Detection objects with:
- `bbox`: BoundingBox coordinates
- `class_id`: COCO class ID (0-79, e.g., 1=person, 3=car)
- `confidence`: Detection confidence (0-1)

**Usage**:
```python
detector = ObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5)
detections = detector.detect(frame_bgr)
for det in detections:
    bbox = det['bbox']
    class_id = det['class_id']
```

### KeypointDetector (Human Pose Estimation)

**Supported Models**:
- `resnet50`: KeypointRCNN with ResNet50 (only option)

**Output**: Detection objects with:
- `bbox`: BoundingBox for person
- `class_id`: Always 1 (person)
- `confidence`: Detection confidence
- `keypoints`: Keypoints object with 17 COCO points

**COCO Keypoints** (17 total):
0. nose, 1. left_eye, 2. right_eye, 3. left_ear, 4. right_ear
5. left_shoulder, 6. right_shoulder, 7. left_elbow, 8. right_elbow
9. left_wrist, 10. right_wrist, 11. left_hip, 12. right_hip
13. left_knee, 14. right_knee, 15. left_ankle, 16. right_ankle

**Usage**:
```python
detector = KeypointDetector(model='resnet50', device='cpu')
detections = detector.detect(frame_bgr)
for det in detections:
    if 'keypoints' in det:
        det['keypoints'].draw(frame)  # Visualize skeleton
```

### ReIDDetector / ReIDExtractor (Appearance Features)

**Supported Models**:
- `osnet_x1_0`: OSNet 1.0x (512-D, recommended)
- `osnet_x0_75`, `osnet_x0_5`, `osnet_x0_25`: Faster variants
- `resnet50`: ResNet50 for ReID

**Output**: Embedding objects with:
- `vector`: L2-normalized 512-D feature vector
- `dim`: Embedding dimension
- `model`: Model name

**Key Design**: Not a Detector (doesn't inherit from Detector ABC)
- Instead, inherits from FeatureExtractor
- Operates on cropped regions, not full frames
- Enhances existing ObjectDetector detections

**Usage**:
```python
# Two-stage pipeline
obj_detector = ObjectDetector(model='mobilenet', classes=[1])  # People only
reid_extractor = ReIDDetector(model='osnet_x1_0')

detections = obj_detector.detect(frame)
detections = reid_extractor.enhance(frame, detections)  # Add embeddings

for det in detections:
    if 'embedding' in det:
        distance = det['embedding'].cosine_distance(other_emb)
```

---

## Testing Strategy

**Framework**: pytest (7.4.0+) with pytest-cov for coverage

**Test Coverage** (11 test files, comprehensive):

| Module | Test File | Coverage |
|--------|-----------|----------|
| Config | test_config.py | 10 tests for YAML loading, merging, validation |
| Frame | test_frame.py | 13 tests for metadata, scaling, copying |
| Factory | test_detector_factory.py | 24 tests for all detector types and configs |
| ObjectDetector | test_object_detector.py | Tests instantiation and inference |
| KeypointDetector | test_keypoint_detector.py | Tests pose estimation output |
| ReIDDetector | test_reid_detector.py | Tests feature extraction |
| BoundingBox | test_bounding_box.py | Tests IoU, scaling, immutability |
| Keypoints | test_keypoints.py | Tests 17-point COCO format |
| Embedding | test_embedding.py | Tests L2 normalization, distance metrics |
| VideoProcessor | test_video_processor.py | Tests frame processing pipeline |

**Running Tests**:
```bash
pytest                          # Run all tests
pytest --cov=src               # With coverage report
pytest --cov=src --cov-report=html  # HTML coverage report
pytest -v                       # Verbose output
pytest tests/test_frame.py -v   # Single test file
```

**Key Testing Patterns**:
- Unit tests for core components
- Integration tests for pipelines
- Fixture-based test data
- Comprehensive error handling tests
- Edge case validation

---

## Dependencies & Environment

### Core Dependencies (requirements.txt)
```
numpy==1.24.3              # Numerical operations
opencv-python==4.8.0.76    # Video I/O, image processing
torch>=2.0.0               # Deep learning framework
torchvision>=0.15.0        # Pre-trained models (FasterRCNN, KeypointRCNN, RetinaNet)
pyyaml>=6.0                # YAML config parsing
torchreid>=1.4.0           # ReID models (OSNet, ResNet50)
```

### Development Dependencies (requirements-dev.txt)
```
pytest>=7.4.0              # Testing framework
pytest-cov>=4.1.0          # Code coverage
flake8>=6.0.0              # Linting
black>=23.0.0              # Code formatting
```

### Hardware Acceleration Support
- **CPU**: Baseline, ~0.5-2 FPS
- **MPS** (Apple Silicon): 5-10x faster, use `--device mps`
- **CUDA** (NVIDIA GPU): 20-50x faster, use `--device cuda`

### Python Version
Requires Python 3.10+

---

## Naming Conventions & Coding Style

### Module Organization
- **`src/`**: Source code (installed package)
- **`tests/`**: Test suite (pytest)
- **`config/`**: Configuration files (YAML)
- **`experiments/`**: Research/optimization code

### Class Naming
- **Concrete detectors**: `ObjectDetector`, `KeypointDetector`, `ReIDDetector`
- **Abstract bases**: `Detector`, `FeatureExtractor`
- **Data containers**: `Detection`, `Frame`, `BoundingBox`, `Keypoints`, `Embedding`
- **Factory**: `DetectorFactory`
- **Management**: `Config`, `VideoProcessor`

### Abbreviations
- `det`/`detections`: Detection objects
- `bbox`/`bboxes`: Bounding boxes
- `kpts`: Keypoints
- `emb`: Embeddings
- `conf_threshold`: Confidence threshold
- `fps`: Frames per second
- `mAP`: Mean Average Precision
- `IoU`: Intersection over Union

### Method Naming
- `from_*()`: Alternative constructors (factory methods)
- `_private_method()`: Internal methods (prefix underscore)
- `preprocess()`, `inference()`, `postprocess()`: Pipeline stages
- `enhance()`: Add properties to existing objects

---

## Performance Optimization

The codebase implements three orthogonal optimizations achieving **270x speedup**:

### 1. Model Selection (15x speedup)
- **ResNet50**: 44M parameters, accurate, slow
- **MobileNet**: 5.5M parameters, efficient, fast
- **Tradeoff**: ~5-10% accuracy loss for 15x speed gain

### 2. Frame Resizing (6x speedup)
- Process frames at lower resolution (e.g., 640px max)
- Scale bounding boxes back to original size
- **Tradeoff**: Smaller distant objects harder to detect

### 3. Frame Skipping (3x speedup)
- Process every Nth frame, reuse detections
- Exploits temporal coherence (objects don't move much between frames)
- **Tradeoff**: Boxes lag slightly behind fast-moving objects

**Combined**: 15 × 6 × 3 = **270x speedup** (0.5 FPS → 135+ FPS)

**Recommended Configurations**:
- **Maximum Accuracy**: `--model resnet50` (~0.5-2 FPS)
- **Balanced**: `--model mobilenet --max-dimension 640 --skip-frames 2` (~80-100 FPS)
- **Maximum Speed**: `--model mobilenet --max-dimension 640 --skip-frames 3` (~120-150 FPS)

See `experiments/PERFORMANCE_OPTIMIZATION.md` for detailed analysis.

---

## Architecture Decisions & Rationales

### 1. Three-Stage Detection Pipeline
**Decision**: Separate preprocess/inference/postprocess into distinct methods  
**Rationale**: Enables subclasses to override specific stages without reimplementing entire detection logic

### 2. Dict-like Detection Container
**Decision**: Use custom `Detection` class instead of dict or dataclass  
**Rationale**: 
- Provides both strict (`det['key']`) and safe (`det.get('key')`) access patterns
- Extensible to any property type without predefined schema
- Clearer semantics than plain dict

### 3. Immutable BoundingBox
**Decision**: Store coordinates as tensor, provide `scale()` method instead of mutable properties  
**Rationale**:
- Prevents accidental modifications
- Enables efficient PyTorch tensor operations (IoU)
- Makes scaling intent explicit

### 4. Frame Abstraction
**Decision**: Wrap numpy arrays in Frame class with metadata  
**Rationale**:
- Enables lazy evaluation of scaling
- Preserves temporal and spatial metadata for tracking
- Single source of truth for frame information

### 5. Factory Pattern for Detectors
**Decision**: Use DetectorFactory instead of constructor overloading  
**Rationale**:
- Centralizes instantiation logic
- Enables automatic validation and model selection
- Decouples detector creation from usage

### 6. Configuration Merging Hierarchy
**Decision**: Allow YAML config + CLI arg override  
**Rationale**:
- Supports both scripted workflows (YAML) and interactive experimentation (CLI)
- Clear precedence: CLI > file > defaults

### 7. Separate Detector and FeatureExtractor
**Decision**: Two distinct abstract base classes  
**Rationale**:
- Detector: Full frame → detections (one-to-many)
- FeatureExtractor: Crops → embeddings (one-to-one)
- Reflects different interfaces and usage patterns

---

## Common Workflows

### 1. Basic Object Detection on Video
```python
from src.detecting.detectors import ObjectDetector
from src.video_processor import VideoProcessor

detector = ObjectDetector(model='mobilenet', device='cpu')
processor = VideoProcessor(
    source='video.mp4',
    detector=detector,
    save_output=True
)
processor.run()
```

### 2. Human Pose Estimation
```python
from src.detecting.detectors import KeypointDetector
from src.video_processor import VideoProcessor

detector = KeypointDetector(device='mps')  # Apple Silicon
processor = VideoProcessor(source='video.mp4', detector=detector)
processor.run()
```

### 3. ObjectDetector + ReID (Two-Stage Pipeline)
```python
from src.detecting.detectors import ObjectDetector, ReIDDetector

obj_det = ObjectDetector(model='mobilenet', classes=[1])  # People
reid_ext = ReIDDetector(model='osnet_x1_0')

frame = cv2.imread('image.jpg')
detections = obj_det.detect(frame)
detections = reid_ext.enhance(frame, detections)  # Add embeddings

for det in detections:
    distance = det['embedding'].cosine_distance(query_embedding)
```

### 4. Custom Detector Implementation
```python
from src.detecting.detector import Detector
from src.detecting.detection import Detection
from src.core.properties import BoundingBox

class CustomDetector(Detector):
    def preprocess(self, image):
        # Convert to model input format
        return custom_preprocessing(image)
    
    def inference(self, input_tensor):
        # Run your model
        return self.model(input_tensor)
    
    def postprocess(self, output, image_shape):
        # Convert to Detection objects
        detections = []
        for box, score, class_id in zip(...):
            det = Detection({
                'bbox': BoundingBox(*box),
                'confidence': score,
                'class_id': class_id
            })
            detections.append(det)
        return detections
```

### 5. Configuration-Based Setup
```python
from src.config import Config
from src.detecting import DetectorFactory
from src.video_processor import VideoProcessor
import logging

config = Config.from_yaml('my_config.yaml')
detector = DetectorFactory.create(config, logging.getLogger())
processor = VideoProcessor(
    source=config.get('video.source'),
    detector=detector,
    save_output=config.get('video.save_output')
)
processor.run()
```

---

## Future Roadmap

From README.md:
- [ ] Tracking System: Multi-object tracking with ID persistence
- [ ] Keypoint Tracking: Track individual keypoints across frames
- [ ] Advanced ReID: Appearance-based re-identification across occlusions
- [ ] Batch Processing: Multi-frame parallel inference
- [ ] Model Zoo: Additional pre-trained models
- [ ] API Documentation: Complete reference and tutorials

---

## File Size & Complexity

- **Total Python Files**: 31
- **Source Code**: ~717 lines across core modules
- **Test Suite**: ~1000+ lines (11 test files)
- **Documentation**: ~800 lines (README, PERFORMANCE_OPTIMIZATION, CHANGELOG)

---

## Key Git History

Recent commits show evolution of the architecture:

1. **Latest** `e043a17`: Added detection factory tests
2. `b5e446f`: Refactor: Improve code quality and fix architectural issues
3. `1fa66a8`: Add ReID detector for appearance-based feature extraction
4. `4b43e02`: Extract DetectorFactory to centralize detector creation
5. `b69f420`: Add Frame abstraction for temporal context
6. `321631b`: Restructure: Move properties to core module
7. Earlier: Keypoint detector, YAML config, webcam support, class filtering

---

## Tips for Working with This Codebase

### 1. Understanding New Detector Types
- Always inherit from `Detector` ABC
- Implement: `preprocess()`, `inference()`, `postprocess()`, `detect()`
- Use parent's `_default_preprocess_pytorch()` and `_default_inference_pytorch()` if applicable

### 2. Adding New Properties to Detection
- Create new class in `src/core/properties/`
- Support `scale()` method if dimension-dependent
- Add visualization `draw()` method if visual
- Update `Detection` container as needed

### 3. Modifying VideoProcessor Rendering
- All visualization in `_render_frame()`, `_draw_overlay()`, `_draw_tracks()`
- Use `det['property'].draw(frame)` pattern
- Maintain separation: detection logic vs rendering

### 4. Configuration Best Practices
- Always provide sensible defaults in code
- Use `config.get(key, default)` for optional settings
- Document all config keys in default.yaml
- Test config merging behavior

### 5. Testing Checklist
- Unit tests for new components
- Integration tests for pipelines
- Test error cases (invalid input, edge cases)
- Use fixtures for test data
- Run with coverage: `pytest --cov=src`

---

## Glossary

| Term | Definition |
|------|-----------|
| **COCO** | Common Objects in Context dataset (80 object classes) |
| **IoU** | Intersection over Union (bounding box similarity) |
| **FPS** | Frames per second (processing speed) |
| **mAP** | Mean Average Precision (detection accuracy metric) |
| **ResNet50** | Deep CNN architecture (44M parameters, accurate) |
| **MobileNet** | Efficient CNN for mobile/edge (5.5M parameters, fast) |
| **FasterRCNN** | Two-stage object detection architecture |
| **RetinaNet** | Single-stage object detection with focal loss |
| **KeypointRCNN** | Object detection + keypoint localization |
| **OSNet** | Lightweight ReID model (person appearance matching) |
| **ReID** | Re-Identification (matching same person across images) |
| **L2 Normalization** | Scaling vector to unit length (||v|| = 1) |
| **Cosine Distance** | Similarity metric for normalized vectors |
| **Temporal Coherence** | Objects move smoothly between consecutive frames |
| **Frame Skipping** | Process every Nth frame, reuse detections |

---

## References

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Torchvision Models**: https://pytorch.org/vision/stable/models.html
- **COCO Dataset**: https://cocodataset.org/#explore
- **ByteTrack Paper**: https://github.com/ifzhang/ByteTrack
- **DeepSORT Paper**: https://github.com/nwojke/deep_sort
- **StrongSORT**: https://github.com/dyhBUPT/StrongSORT

---

**Last Updated**: November 14, 2025  
**Maintained By**: Mikhail Cassar  
**License**: MIT
