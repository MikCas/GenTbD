"""Detection system for GenTbD.

Components:
- Detector: Base class defining the detection pipeline
- FeatureExtractor: Base class for appearance-based feature extraction
- Detection: Container for detection results (bbox, class_id, confidence)
- DetectorFactory: Factory for creating detector instances from config
- detectors/: Concrete detector implementations
"""

from .detector import Detector
from .feature_extractor import FeatureExtractor
from .detection import Detection
from .factory import DetectorFactory

__all__ = ['Detector', 'FeatureExtractor', 'Detection', 'DetectorFactory']
