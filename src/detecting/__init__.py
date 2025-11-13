"""Detection system for GenTbD.

Components:
- Detector: Base class defining the detection pipeline
- Detection: Container for detection results (bbox, class_id, confidence)
- DetectorFactory: Factory for creating detector instances from config
- detectors/: Concrete detector implementations
"""

from .detector import Detector
from .detection import Detection
from .factory import DetectorFactory

__all__ = ['Detector', 'Detection', 'DetectorFactory']
