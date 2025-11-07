"""Detection properties for GenTbD.

Property classes used in Detection objects:
- BoundingBox: Bounding box implementation using PyTorch tensors
"""

from .bounding_box import BoundingBox

__all__ = ['BoundingBox']
