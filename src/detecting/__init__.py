"""Detection system for GenTbD.

Components:
- Detector: Base class defining the detection pipeline
- Detection: Container for detection results (bbox, class_id, confidence)
- detectors/: Concrete detector implementations
- properties/: Detection properties (BoundingBox, etc.)
"""

from .detector import Detector
from .detection import Detection

__all__ = ['Detector', 'Detection']
