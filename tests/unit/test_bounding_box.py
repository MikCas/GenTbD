"""Unit tests for BoundingBox class.

Unit tests are fast, isolated tests that test individual components
without dependencies on other parts of the system.
"""

import pytest
import torch
import numpy as np
from src.detecting.properties.bounding_box import BoundingBox


class TestBoundingBoxCreation:
    """Test BoundingBox initialization."""

    def test_create_bounding_box(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(10, 20, 100, 200)

        assert bbox is not None
        assert isinstance(bbox._tensor, torch.Tensor)

    def test_xyxy_property(self):
        """Test xyxy property returns correct coordinates."""
        bbox = BoundingBox(10, 20, 100, 200)
        x1, y1, x2, y2 = bbox.xyxy

        assert x1 == 10
        assert y1 == 20
        assert x2 == 100
        assert y2 == 200

    def test_coordinates_are_floats(self):
        """Test that internal tensor uses float dtype."""
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox._tensor.dtype == torch.float32

    @pytest.mark.parametrize("x1,y1,x2,y2", [
        (0, 0, 100, 100),           # Top-left corner
        (100, 100, 200, 200),       # Middle
        (500.5, 300.5, 600.5, 400.5),  # Float coordinates
    ])
    def test_various_coordinates(self, x1, y1, x2, y2):
        """Test bounding box with various coordinate values."""
        bbox = BoundingBox(x1, y1, x2, y2)
        result_x1, result_y1, result_x2, result_y2 = bbox.xyxy

        assert result_x1 == pytest.approx(x1)
        assert result_y1 == pytest.approx(y1)
        assert result_x2 == pytest.approx(x2)
        assert result_y2 == pytest.approx(y2)


class TestBoundingBoxIoU:
    """Test IoU (Intersection over Union) calculation."""

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0."""
        bbox1 = BoundingBox(10, 10, 100, 100)
        bbox2 = BoundingBox(10, 10, 100, 100)

        iou = bbox1.iou(bbox2)
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes should be 0.0."""
        bbox1 = BoundingBox(0, 0, 50, 50)
        bbox2 = BoundingBox(100, 100, 150, 150)

        iou = bbox1.iou(bbox2)
        assert iou == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        # Two 100x100 boxes with 50% overlap in one dimension
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 0, 150, 100)

        # Intersection: 50x100 = 5000
        # Union: 100x100 + 100x100 - 5000 = 15000
        # IoU = 5000/15000 = 0.333...

        iou = bbox1.iou(bbox2)
        assert iou == pytest.approx(0.333, rel=0.01)

    def test_iou_one_inside_another(self):
        """Test IoU when one box is inside another."""
        bbox1 = BoundingBox(0, 0, 100, 100)  # Large box
        bbox2 = BoundingBox(25, 25, 75, 75)  # Small box inside

        # Intersection: 50x50 = 2500
        # Union: 100x100 = 10000
        # IoU = 2500/10000 = 0.25

        iou = bbox1.iou(bbox2)
        assert iou == pytest.approx(0.25)

    def test_iou_is_symmetric(self):
        """Test that IoU(A, B) == IoU(B, A)."""
        bbox1 = BoundingBox(10, 10, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)

        iou1 = bbox1.iou(bbox2)
        iou2 = bbox2.iou(bbox1)

        assert iou1 == pytest.approx(iou2)


class TestBoundingBoxDrawing:
    """Test bounding box visualization."""

    def test_draw_on_image(self, sample_image):
        """Test drawing bounding box on an image."""
        bbox = BoundingBox(100, 100, 200, 200)
        image_copy = sample_image.copy()

        # Should not raise exception
        bbox.draw(image_copy, color=(255, 0, 0))

        # Image should be modified (not same as original)
        assert not np.array_equal(image_copy, sample_image)

    def test_draw_with_custom_color(self, sample_image):
        """Test drawing with different colors."""
        bbox = BoundingBox(100, 100, 200, 200)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for color in colors:
            image_copy = sample_image.copy()
            bbox.draw(image_copy, color=color)
            # Should complete without error
            assert image_copy is not None


class TestBoundingBoxStringRepresentation:
    """Test string representation of BoundingBox."""

    def test_repr(self):
        """Test __repr__ method."""
        bbox = BoundingBox(10.5, 20.5, 100.5, 200.5)
        repr_str = repr(bbox)

        assert "BoundingBox" in repr_str
        assert "10.5" in repr_str
        assert "20.5" in repr_str
        assert "100.5" in repr_str
        assert "200.5" in repr_str

    def test_repr_format(self):
        """Test that repr shows coordinates with one decimal place."""
        bbox = BoundingBox(10.123, 20.456, 100.789, 200.999)
        repr_str = repr(bbox)

        # Should round to 1 decimal place
        assert "10.1" in repr_str
        assert "20.5" in repr_str  # Rounds up
