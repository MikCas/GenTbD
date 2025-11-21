# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2025-11-21

### Added
- Text backgrounds for all overlay text (FPS, frame count, mode) for better readability
- Detection labels showing class name and confidence on bounding boxes
- `--list-models` CLI flag to display all available models and aliases
- Webcam index validation (0-10 range)
- Troubleshooting section in README

### Changed
- **CLI Refactored**: Model configuration now exclusively in YAML config file
- Simplified CLI to runtime arguments only (`--video`, `--webcam`, `--config`, etc.)
- FPS calculation now measures full loop time (detection + rendering + display)
- Removed CLI arguments: `--model`, `--conf`, `--device`, `--classes`, `--max-dimension`, `--skip-frames`, `--detector-type`

### Fixed
- Accurate FPS measurement (was only measuring detection time)
- Duplicate logging import in `VideoProcessor`
- Default detector type value standardized to `object_detection`

## [1.1.0] - 2025-11-20

### Added
- **Model Registry**: Centralized system for managing model metadata and instantiation.
- Programmatic verification of model parameters (no more hardcoded values).
- Runtime device support checks (CPU/CUDA/MPS).
- `ModelSpec` dataclass for structured model definitions.
- `src/detecting/model_registry.py` module.

### Changed
- Refactored `DetectorFactory` to use `ModelRegistry` instead of string parsing.
- Simplified `factory.py` by removing complex conditional logic.
- Updated `README.md` to reflect the new architecture.

## [1.0.0] - 2025-11-16

### Added
- YOLO detector support (YOLOv8/YOLOv11) for Apple Silicon MPS optimization
- ReID feature extraction system with OSNet and ResNet50 models
- Separate reid/ module for appearance-based feature extraction
- ReID CLI flags: `--extract-reid`, `--reid-model`, `--reid-device`
- ReID configuration section in default.yaml
- DetectorFactory for centralized detector creation
- Comprehensive model naming convention (task-based types + explicit model names)
- Model weights (.pt, .pth, .weights) added to .gitignore

### Changed
- **BREAKING**: Removed ALL backward compatibility for detector types and model names
- **BREAKING**: ReID is now a feature extractor, not a detector type
- **BREAKING**: Detector types limited to: `object_detection`, `keypoint_detection`
- Reorganized architecture: detectors/ for detection, reid/ for feature extraction
- Updated configuration structure with separate reid section
- Improved naming: detector types are task-based, models are explicit
- Updated all documentation to reflect v1.0.0 architecture

### Removed
- Backward compatibility for legacy detector types ('object', 'keypoint', 'yolo', 'reid')
- Backward compatibility for legacy model names ('mobilenet', 'resnet50', 'yolov8n', etc.)
- OptimizedObjectDetector (dead code, never used)
- experiments/mps_debug/ (issue solved, documented in MPS_SOLUTION.md)
- Model weight file yolov8n.pt from repository
- All deprecation warnings and type/model mapping logic

### Fixed
- MPS performance issue on Apple Silicon (5000x speedup with YOLO)
- Config validation for new detector types
- Import organization after ReID restructuring

## [0.1.0] - 2025-11-07

### Added
- Object detection system with FasterRCNN and RetinaNet support
- Interactive video processor with real-time visualization
- Hardware acceleration support (CPU, CUDA, MPS)
- Class filtering for COCO dataset classes
- Confidence threshold configuration
- Extensible detector architecture with abstract base class
- Detection result container with dict-like interface
- BoundingBox class with IoU calculation and drawing
- Video output saving capability
- Frame-by-frame processing with step mode
- Frame capture to image files

### Architecture
- Standardized detection pipeline (preprocess → inference → postprocess)
- Modular structure with separate detecting package
- Factory methods for easy detector instantiation
- Support for custom detector implementations

[0.1.0]: https://github.com/mikhailcassar/GenTbD/releases/tag/v0.1.0
