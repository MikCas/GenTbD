"""
Property classes for detection and tracking.

This module contains fundamental property classes like BoundingBox, Keypoints,
and Embedding that are used across detection, tracking, and ReID modules.
"""

from .bounding_box import BoundingBox
from .keypoints import Keypoints, KEYPOINT_NAMES, SKELETON
from .embedding import Embedding

__all__ = [
    'BoundingBox',
    'Keypoints',
    'KEYPOINT_NAMES',
    'SKELETON',
    'Embedding',
]
