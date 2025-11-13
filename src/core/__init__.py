"""
Core module containing shared domain objects and abstractions.

This module provides fundamental abstractions used across detection,
tracking, and other modules.
"""

from .frame import Frame
from .properties.bounding_box import BoundingBox
from .properties.keypoints import Keypoints, KEYPOINT_NAMES, SKELETON

__all__ = [
    'Frame',
    'BoundingBox',
    'Keypoints',
    'KEYPOINT_NAMES',
    'SKELETON',
]
