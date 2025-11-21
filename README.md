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

3. Add your video file to the `data/` directory (or use the default `TownCent.mp4`)

### Running the Project

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Basic usage with default video and config
python -m src.main --video data/TownCent.mp4

# Use webcam (default camera index 0)
python -m src.main --webcam

# Use specific webcam (e.g., external camera at index 1)
python -m src.main --webcam 1

# Run with custom configuration
python -m src.main --config my_config.yaml --video data/your_video.mp4

# List available models
python -m src.main --list-models

# For development - run tests
pytest

# For development - check test coverage
pytest --cov=src --cov-report=html
```

**Command-line Arguments:**

**Video Source (mutually exclusive):**

- `--video`: Path to input video file (default: `data/TownCent.mp4` if no source specified)
- `--webcam`: Use webcam as input (optionally specify camera index, default: 0)

**Runtime Options:**

- `--config`: Path to YAML config file (default: `config/default.yaml`)
- `--save-output`: Save processed video to file
- `--output`: Output video path (default: auto-generated `output_YYYYMMDD_HHMMSS.mp4`)
- `--list-models`: List all available models and exit
- `--verbose`: Enable detailed logging

> **Note**: All model settings (model type, confidence, device, etc.) are now configured exclusively in the YAML config file.

**Interactive Controls:**

- `c` - Toggle continuous/step mode
- `SPACE` - Next frame (in step mode)
- `s` - Save current frame as image
- `q` - Quit

### Pushing Changes to GitHub

```bash
# Check status of your changes
git status

# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Your commit message here"

# Push to GitHub
git push origin base

# If you want to push to main branch instead
git checkout main
git merge base
git push origin main
```

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

**2. Model Registry (New)**
- Centralized database of all supported models
- Programmatically verifies parameter counts and device support
- Handles model aliases (e.g., 'mobilenet' -> 'fasterrcnn_mobilenet')
- Lazy-loads metadata to keep startup fast

**3. ObjectDetector (Implementation)**
- Supports multiple torchvision models (FasterRCNN, RetinaNet)
- Factory uses Registry to instantiate the correct backend (TorchVision or YOLO)
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
from src.detecting import DetectorFactory, Config
import cv2

# Load config
config = Config.from_default()
config.set('detection.model', 'mobilenet')  # Use alias
config.set('detection.device', 'cpu')

# Create detector via Factory
detector = DetectorFactory.create(config)

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

## Troubleshooting

### Installation Issues

**Missing dependencies:**
```bash
# If torchreid is missing
pip install torchreid

# If gdown is missing (for ReID model downloads)
pip install gdown

# If tensorboard is missing (required by torchreid)
pip install tensorboard
```

**Python version compatibility:**
- GenTbD requires Python 3.9 or higher
- Check your version: `python --version`

### Runtime Issues

**"Video file not found" error:**
- Ensure the video path is correct
- Try using absolute paths instead of relative paths
- Verify file permissions

**Low FPS on MPS (Apple Silicon):**
- First run is slower due to model compilation
- Subsequent runs should be significantly faster (~8-15 FPS)
- See `experiments/MPS_SOLUTION.md` for device-specific optimizations

**Webcam not opening:**
- Try different camera indices: `--webcam 0`, `--webcam 1`
- Check camera permissions in System Settings (macOS)
- Ensure no other application is using the camera
- Valid webcam indices are 0-10

**"COCO class not found" warnings:**
- Some models use different class IDs than COCO
- This is expected and doesn't affect detection
- Labels will show as "Class_X" for unknown IDs

### Performance Issues

**Detection is slow:**
- Use `--list-models` to find faster models (e.g., `yolo_v8n`)
- Reduce input resolution in config: `video.max_dimension: 640`
- Enable frame skipping in config: `video.skip_frames: 3`
- Use MPS (Apple Silicon) or CUDA (NVIDIA) instead of CPU

**High memory usage:**
- Close other applications
- Reduce `video.max_dimension` in config
- Use smaller models (mobilenet, yolo_v8n)

### Configuration Issues

**"Invalid detection type" error:**
- Use `object_detection` or `keypoint_detection` (not `object` or `keypoint`)
- Check `config/default.yaml` for valid values

**Model not found:**
- Run `python -m src.main --list-models` to see available models
- Check spelling and use exact model name or alias

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
