"""Unit tests for KeypointDetector class."""

import pytest
import numpy as np
import torch
from src.detecting.detectors.keypoint_detector import KeypointDetector
from src.core.properties import Keypoints


class TestKeypointDetector:
    """Test KeypointDetector functionality."""

    def test_init_resnet50(self):
        """Test KeypointDetector initializes with resnet50 model."""
        detector = KeypointDetector(model='resnet50', device='cpu')

        assert detector.model is not None
        assert detector.device == 'cpu'  # String comparison
        assert detector.conf_threshold == 0.5
        assert detector.keypoint_threshold == 0.5

    def test_init_with_custom_thresholds(self):
        """Test KeypointDetector with custom thresholds."""
        detector = KeypointDetector(
            model='resnet50',
            device='cpu',
            conf_threshold=0.7,
            keypoint_threshold=0.6
        )

        assert detector.conf_threshold == 0.7
        assert detector.keypoint_threshold == 0.6

    def test_init_invalid_model(self):
        """Test KeypointDetector raises error for invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            KeypointDetector(model='invalid_model')

    def test_preprocess(self):
        """Test preprocessing converts BGR to RGB tensor."""
        detector = KeypointDetector(model='resnet50')

        # Create BGR image with blue channel
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Blue channel (index 0 in BGR)

        # Preprocess
        tensor = detector.preprocess(image)

        # Check shape: (C, H, W)
        assert tensor.shape == (3, 480, 640)

        # Check normalization (0-1 range)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

        # Check BGR to RGB conversion
        # Blue was in channel 0 (BGR), should now be in channel 2 (RGB)
        assert tensor[2].max() == 1.0  # Blue channel (was BGR[0], now RGB[2])
        assert tensor[0].max() == 0.0  # Red channel should be empty
        assert tensor[1].max() == 0.0  # Green channel should be empty

    def test_postprocess_filters_confidence(self):
        """Test postprocessing filters by confidence threshold."""
        detector = KeypointDetector(model='resnet50', conf_threshold=0.7)

        # Create mock output with 2 detections, 1 low confidence
        output = {
            'boxes': torch.tensor([[100, 200, 300, 400], [50, 100, 150, 200]]),
            'labels': torch.tensor([1, 1]),  # Both are people
            'scores': torch.tensor([0.8, 0.5]),  # One above, one below threshold
            'keypoints': torch.rand(2, 17, 3),
            'keypoints_scores': torch.rand(2, 17)
        }

        # Postprocess
        detections = detector.postprocess(output, (480, 640))

        # Should only return 1 detection (confidence > 0.7)
        assert len(detections) == 1
        assert abs(detections[0]['confidence'] - 0.8) < 0.01  # Float comparison with tolerance

    def test_postprocess_creates_keypoints_object(self):
        """Test postprocessing creates Keypoints objects."""
        detector = KeypointDetector(model='resnet50')

        # Create mock output
        output = {
            'boxes': torch.tensor([[100, 200, 300, 400]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.9]),
            'keypoints': torch.rand(1, 17, 3),
            'keypoints_scores': torch.rand(1, 17)
        }

        # Postprocess
        detections = detector.postprocess(output, (480, 640))

        # Check keypoints property exists
        assert 'keypoints' in detections[0]

        # Check it's a Keypoints object
        assert isinstance(detections[0]['keypoints'], Keypoints)

        # Check keypoints have correct shape
        kpts = detections[0]['keypoints'].keypoints
        assert kpts.shape == (17, 3)

    def test_postprocess_empty_result(self):
        """Test postprocessing with no detections."""
        detector = KeypointDetector(model='resnet50', conf_threshold=0.9)

        # Create mock output with low confidence
        output = {
            'boxes': torch.tensor([[100, 200, 300, 400]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.3]),  # Below threshold
            'keypoints': torch.rand(1, 17, 3),
            'keypoints_scores': torch.rand(1, 17)
        }

        # Postprocess
        detections = detector.postprocess(output, (480, 640))

        # Should return empty list
        assert len(detections) == 0

    def test_postprocess_multiple_detections(self):
        """Test postprocessing with multiple people."""
        detector = KeypointDetector(model='resnet50', conf_threshold=0.5)

        # Create mock output with 3 people
        output = {
            'boxes': torch.tensor([[100, 200, 300, 400],
                                  [400, 200, 600, 400],
                                  [100, 450, 300, 650]]),
            'labels': torch.tensor([1, 1, 1]),
            'scores': torch.tensor([0.9, 0.8, 0.7]),
            'keypoints': torch.rand(3, 17, 3),
            'keypoints_scores': torch.rand(3, 17)
        }

        # Postprocess
        detections = detector.postprocess(output, (800, 800))

        # Should return all 3 detections
        assert len(detections) == 3

        # Check all have keypoints
        for det in detections:
            assert 'keypoints' in det
            assert isinstance(det['keypoints'], Keypoints)

    def test_postprocess_class_id_is_person(self):
        """Test that class_id is always 1 (person) for keypoint detector."""
        detector = KeypointDetector(model='resnet50')

        # Create mock output
        output = {
            'boxes': torch.tensor([[100, 200, 300, 400]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.9]),
            'keypoints': torch.rand(1, 17, 3),
            'keypoints_scores': torch.rand(1, 17)
        }

        # Postprocess
        detections = detector.postprocess(output, (480, 640))

        # Check class_id is 1 (person)
        assert detections[0]['class_id'] == 1
