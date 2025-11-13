"""Unit tests for BoundingBox class."""

import pytest
import torch
from src.core.properties import BoundingBox


class TestBoundingBox:
    """Test BoundingBox functionality."""

    def test_initialization(self):
        """Test BoundingBox can be created with coordinates."""
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox is not None
        x1, y1, x2, y2 = bbox.xyxy
        assert x1 == 10.0
        assert y1 == 20.0
        assert x2 == 100.0
        assert y2 == 200.0

    def test_xyxy_property(self):
        """Test xyxy property returns coordinates as tuple."""
        bbox = BoundingBox(5, 10, 50, 100)
        coords = bbox.xyxy
        assert isinstance(coords, tuple)
        assert len(coords) == 4

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 100, 100)
        iou = bbox1.iou(bbox2)
        assert abs(iou - 1.0) < 0.001  # Allow for floating point error

    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        bbox1 = BoundingBox(0, 0, 50, 50)
        bbox2 = BoundingBox(100, 100, 150, 150)
        iou = bbox1.iou(bbox2)
        assert abs(iou - 0.0) < 0.001

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)
        iou = bbox1.iou(bbox2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143
        assert 0.14 < iou < 0.15

    def test_repr(self):
        """Test string representation."""
        bbox = BoundingBox(10, 20, 30, 40)
        repr_str = repr(bbox)
        assert "BoundingBox" in repr_str
        assert "10" in repr_str
        assert "20" in repr_str
        assert "30" in repr_str
        assert "40" in repr_str

    def test_coordinates_are_floats(self):
        """Test that coordinates are returned as floats."""
        bbox = BoundingBox(1, 2, 3, 4)
        x1, y1, x2, y2 = bbox.xyxy
        assert isinstance(x1, float)
        assert isinstance(y1, float)
        assert isinstance(x2, float)
        assert isinstance(y2, float
)

    def test_scaling_coordinates(self):
        """Test creating scaled bounding boxes."""
        bbox_original = BoundingBox(100, 200, 300, 400)
        x1, y1, x2, y2 = bbox_original.xyxy

        # Scale up by 2x
        scale_factor = 0.5  # Detection at 0.5x, so scale back by 1/0.5 = 2
        bbox_scaled = BoundingBox(
            x1 / scale_factor,
            y1 / scale_factor,
            x2 / scale_factor,
            y2 / scale_factor
        )

        x1_new, y1_new, x2_new, y2_new = bbox_scaled.xyxy
        assert x1_new == 200.0
        assert y1_new == 400.0
        assert x2_new == 600.0
        assert y2_new == 800.0

    def test_scale_method(self):
        """Test BoundingBox.scale() method."""
        bbox = BoundingBox(10, 20, 100, 200)

        # Scale by 2.0 (double size)
        scaled_up = bbox.scale(2.0)
        x1, y1, x2, y2 = scaled_up.xyxy
        assert x1 == 20.0
        assert y1 == 40.0
        assert x2 == 200.0
        assert y2 == 400.0

        # Scale by 0.5 (half size)
        scaled_down = bbox.scale(0.5)
        x1, y1, x2, y2 = scaled_down.xyxy
        assert x1 == 5.0
        assert y1 == 10.0
        assert x2 == 50.0
        assert y2 == 100.0

        # Original bbox should be unchanged (immutable)
        x1, y1, x2, y2 = bbox.xyxy
        assert x1 == 10.0
        assert y1 == 20.0
        assert x2 == 100.0
        assert y2 == 200.0
