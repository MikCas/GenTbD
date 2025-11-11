# GenTbD: Generalized Tracking-by-Detection

**GenTbD** is an object detection system for video processing built on PyTorch and torchvision. It provides a flexible framework for detection-based video analysis with support for multiple detection models and extensible architecture.

> **Version**: 0.1.0

---

## Features

- **Detection Pipeline**: Standardized preprocess → inference → postprocess pattern
- **Multiple Models**: Support for FasterRCNN, RetinaNet, and other torchvision detectors
- **Interactive Video Processing**: Real-time video player with detection visualization and frame controls
- **Flexible Architecture**: Extensible Detection container and custom detector support
- **Hardware Acceleration**: CPU, CUDA (NVIDIA), and MPS (Apple Silicon) support
- **Class Filtering**: Detect specific object classes from the COCO dataset

---

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mikhailcassar/GenTbD.git
   cd GenTbD
   ```

2. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running Detection on Video

```bash
# Basic usage (CPU, default settings)
python -m src.main --video data/TownCent.mp4

# With custom settings
python -m src.main \
    --video data/your_video.mp4 \
    --conf 0.7 \
    --device mps \
    --classes 1 \
    --save-output
```

**Arguments:**
- `--video`: Path to input video (default: `data/TownCent.mp4`)
- `--conf`: Detection confidence threshold 0.0-1.0 (default: `0.5`)
- `--device`: Device for inference: `cpu`, `mps` (Apple Silicon), or `cuda` (default: `cpu`)
- `--classes`: Filter by class IDs (e.g., `--classes 1` for people only)
- `--save-output`: Save processed video to file
- `--output`: Output video path (auto-generated if not specified)
- `--verbose`: Enable detailed logging

**Interactive Controls:**
- `c` - Toggle continuous/step mode
- `SPACE` - Next frame (in step mode)
- `s` - Save current frame as image
- `q` - Quit

---

## Architecture

### Project Structure

```
GenTbD/
├── data/                          # Input videos
│   └── TownCent.mp4
├── src/                           # Source code
│   ├── main.py                    # Main entry point
│   ├── video_processor.py         # VideoProcessor class
│   └── detecting/                 # Detection system
│       ├── detector.py            # Base Detector ABC
│       ├── detection.py           # Detection result container
│       ├── detectors/             # Concrete implementations
│       │   └── object_detector.py # FasterRCNN/RetinaNet
│       └── properties/            # Detection properties
│           └── bounding_box.py    # BBox wrapper
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

### Detection System

The detection system follows an extensible architecture:

**1. Detector (Base Class)**
- Defines standard pipeline: `preprocess()` → `inference()` → `postprocess()` → `detect()`
- Abstract base class for implementing custom detectors

**2. ObjectDetector (Implementation)**
- Supports multiple torchvision models (FasterRCNN, RetinaNet)
- Factory methods: `from_fasterrcnn_resnet50()`, `from_retinanet()`, `from_pretrained()`
- Full CPU/GPU/MPS device support

**3. Detection (Result Container)**
- Dict-like container for detection properties
- Standard properties: `bbox`, `class_id`, `confidence`
- Extensible for additional properties (keypoints, embeddings, custom features)

**4. BoundingBox**
- PyTorch tensor-based bounding box representation
- Methods: `iou()` for IoU calculation, `draw()` for visualization, `xyxy` property for coordinates

---

## Usage Examples

### Basic Detection

```python
from src.detecting.detectors import ObjectDetector
import cv2

# Load detector
detector = ObjectDetector.from_fasterrcnn_resnet50(
    device='cpu',
    conf_threshold=0.5,
    classes=[1]  # Only detect people
)

# Detect on image
frame = cv2.imread('image.jpg')
detections = detector.detect(frame)

# Access results
for det in detections:
    bbox = det['bbox']           # BoundingBox object
    class_id = det['class_id']   # int
    confidence = det['confidence'] # float

    # Draw bounding box
    bbox.draw(frame, color=(0, 255, 0))
```

### Video Processing

```python
from src.video_processor import VideoProcessor
from src.detecting.detectors import ObjectDetector

# Setup detector
detector = ObjectDetector.from_fasterrcnn_resnet50(device='mps')

# Create video processor
processor = VideoProcessor(
    video_path='data/video.mp4',
    detector=detector,
    save_output=True
)

# Process video with interactive controls
processor.run()
```

### Custom Detector

Implement your own detector by subclassing the `Detector` base class:

```python
from src.detecting.detector import Detector
from src.detecting.detection import Detection
from src.detecting.properties.bounding_box import BoundingBox
import torch

class MyDetector(Detector):
    def preprocess(self, image):
        # Convert BGR image to model input format
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def inference(self, input_tensor):
        # Run your model
        return self.model(input_tensor)

    def postprocess(self, output, image_shape):
        # Convert model output to Detection objects
        detections = []
        for box, score, class_id in zip(output['boxes'], output['scores'], output['labels']):
            det = Detection({
                'bbox': BoundingBox(*box.tolist()),
                'confidence': float(score),
                'class_id': int(class_id)
            })
            detections.append(det)
        return detections
```

---

## COCO Class IDs

The detectors support the 80 classes from the COCO dataset. Common class IDs for filtering:

- `1` - person
- `2` - bicycle
- `3` - car
- `5` - airplane
- `7` - train
- `16` - dog
- `17` - cat

For the complete list of 80 classes, see the [COCO dataset documentation](https://cocodataset.org/#explore).

---

## Performance Notes

### Device Selection

- **CPU**: ~1-2 FPS on M-series Macs, good for development
- **MPS** (Apple Silicon): ~8-15 FPS, recommended for production
- **CUDA** (NVIDIA GPU): Fastest, recommended if available

### Model Selection

- **FasterRCNN ResNet50**: Most accurate, slower (~50-100ms/frame on GPU)
- **FasterRCNN MobileNet**: Faster, slightly less accurate
- **RetinaNet**: Good balance of speed and accuracy

### Optimization Tips

1. **Frame skipping**: Process every Nth frame for speed
2. **Lower resolution**: Resize frames before detection
3. **Confidence threshold**: Higher threshold = fewer false positives
4. **Class filtering**: Only detect specific classes

---

## Future Roadmap

Planned enhancements for future versions:

- [ ] **Tracking System**: Multi-object tracking with ID persistence across frames
- [ ] **Keypoint Detection**: Human pose estimation and keypoint tracking
- [ ] **ReID Features**: Appearance-based re-identification for tracking
- [ ] **Batch Processing**: Multi-frame batch inference for improved throughput
- [ ] **Model Zoo**: Additional pre-trained detector models
- [ ] **Unit Tests**: Comprehensive test coverage
- [ ] **API Documentation**: Complete API reference and tutorials

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (once available)
5. Submit a pull request

Please ensure your code follows the existing architecture patterns and includes appropriate documentation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Inspired by modern MOT systems:
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)

Built with:
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- [OpenCV](https://opencv.org/)
