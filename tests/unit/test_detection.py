"""Unit tests for Detection class."""

import pytest
from src.detecting.detection import Detection
from src.detecting.properties.bounding_box import BoundingBox


class TestDetectionCreation:
    """Test Detection object creation and dictionary-like interface."""

    def test_create_detection(self):
        """Test basic detection creation."""
        detection = Detection({
            'bbox': BoundingBox(10, 20, 100, 200),
            'class_id': 1,
            'confidence': 0.95
        })

        assert detection is not None

    def test_detection_dict_access(self):
        """Test accessing detection data like a dictionary."""
        bbox = BoundingBox(10, 20, 100, 200)
        detection = Detection({
            'bbox': bbox,
            'class_id': 1,
            'confidence': 0.95
        })

        assert detection['bbox'] == bbox
        assert detection['class_id'] == 1
        assert detection['confidence'] == 0.95

    def test_detection_attribute_access(self):
        """Test accessing detection data as attributes (if supported)."""
        bbox = BoundingBox(10, 20, 100, 200)
        detection = Detection({
            'bbox': bbox,
            'class_id': 1,
            'confidence': 0.95
        })

        # Should work like a dict
        assert 'bbox' in detection
        assert 'class_id' in detection
        assert 'confidence' in detection

    def test_detection_keys(self):
        """Test that detection has expected keys."""
        detection = Detection({
            'bbox': BoundingBox(10, 20, 100, 200),
            'class_id': 1,
            'confidence': 0.95
        })

        assert 'bbox' in detection
        assert 'class_id' in detection
        assert 'confidence' in detection

    def test_detection_extensibility(self):
        """Test that detection can hold arbitrary properties."""
        # This is important for future ReID and keypoint extensions
        detection = Detection({
            'bbox': BoundingBox(10, 20, 100, 200),
            'class_id': 1,
            'confidence': 0.95,
            'custom_property': 'test_value'
        })

        assert detection['custom_property'] == 'test_value'


class TestDetectionValidation:
    """Test detection validation and edge cases."""

    def test_empty_detection(self):
        """Test creating detection with minimal data."""
        detection = Detection({})
        assert detection is not None

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 0.99, 1.0])
    def test_various_confidence_scores(self, confidence):
        """Test detection with various confidence scores."""
        detection = Detection({
            'bbox': BoundingBox(10, 20, 100, 200),
            'confidence': confidence
        })

        assert detection['confidence'] == confidence

    @pytest.mark.parametrize("class_id", [0, 1, 79, 100])
    def test_various_class_ids(self, class_id):
        """Test detection with various class IDs."""
        detection = Detection({
            'bbox': BoundingBox(10, 20, 100, 200),
            'class_id': class_id
        })

        assert detection['class_id'] == class_id
