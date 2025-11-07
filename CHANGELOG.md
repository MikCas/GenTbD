# Changelog

All notable changes to this project will be documented in this file.

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

[0.1.0]: https://github.com/your-username/GenTbD/releases/tag/v0.1.0
