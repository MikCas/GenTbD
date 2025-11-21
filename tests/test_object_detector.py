"""Unit tests for ObjectDetector class."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from src.detecting.object.torchvision import TorchVisionObjectDetector


class TestObjectDetector:
    """Test ObjectDetector functionality."""

    def test_init_resnet50(self):
        """Test ResNet50 model initialization."""
        detector = TorchVisionObjectDetector(model='resnet50', device='cpu', conf_threshold=0.6)

        assert detector is not None
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.6
        assert detector.model is not None

    def test_init_mobilenet(self):
        """Test MobileNet model initialization."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5)

        assert detector is not None
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.5
        assert detector.model is not None

    def test_init_retinanet(self):
        """Test RetinaNet model initialization."""
        detector = TorchVisionObjectDetector(model='retinanet', device='cpu', conf_threshold=0.7)

        assert detector is not None
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.7
        assert detector.model is not None

    def test_init_with_classes_filter(self):
        """Test initialization with class filtering."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', classes=[0, 1, 2])

        assert detector.classes == [0, 1, 2]

    def test_init_without_classes_filter(self):
        """Test initialization without class filtering."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu')

        assert detector.classes is None

    def test_init_invalid_model(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            TorchVisionObjectDetector(model='invalid_model', device='cpu')

    def test_preprocess_bgr_to_rgb(self):
        """Test preprocessing converts BGR to RGB."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu')

        # Create a simple BGR image (blue in top-left corner)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[0:50, 0:50, 0] = 255  # Blue channel

        tensor = detector.preprocess(image)

        # Check tensor shape (C, H, W)
        assert tensor.shape == (3, 100, 100)

        # Check values are normalized to [0, 1]
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

        # Check BGR was converted to RGB (blue should now be in channel 2)
        # Top-left corner should have high value in red channel (index 2 after BGR→RGB)
        assert tensor[2, 0, 0] > 0.9  # Red channel (was blue in BGR)

    def test_preprocess_normalization(self):
        """Test preprocessing normalizes values to [0, 1]."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu')

        # Create image with max values
        image = np.full((50, 50, 3), 255, dtype=np.uint8)

        tensor = detector.preprocess(image)

        # All values should be approximately 1.0
        assert torch.allclose(tensor, torch.ones_like(tensor), atol=0.01)

    def test_preprocess_tensor_permutation(self):
        """Test preprocessing permutes dimensions correctly."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu')

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor = detector.preprocess(image)

        # Should be (C, H, W) not (H, W, C)
        assert tensor.shape == (3, 100, 200)

    def test_postprocess_confidence_filtering(self):
        """Test postprocessing filters by confidence threshold."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.7)

        # Mock detection output
        output = {
            'boxes': torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80], [100, 110, 120, 130]]),
            'labels': torch.tensor([1, 2, 3]),
            'scores': torch.tensor([0.9, 0.6, 0.8])  # Only 0.9 and 0.8 pass 0.7 threshold
        }

        detections = detector.postprocess(output, (480, 640))

        # Should have 2 detections (scores 0.9 and 0.8)
        assert len(detections) == 2
        assert abs(detections[0]['confidence'] - 0.9) < 0.01
        assert abs(detections[1]['confidence'] - 0.8) < 0.01

    def test_postprocess_class_filtering(self):
        """Test postprocessing filters by class IDs."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5, classes=[1, 3])

        # Mock detection output with classes 1, 2, 3
        output = {
            'boxes': torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80], [100, 110, 120, 130]]),
            'labels': torch.tensor([1, 2, 3]),  # Only 1 and 3 should pass
            'scores': torch.tensor([0.9, 0.9, 0.9])
        }

        detections = detector.postprocess(output, (480, 640))

        # Should have 2 detections (class_id 1 and 3)
        assert len(detections) == 2
        assert detections[0]['class_id'] == 1
        assert detections[1]['class_id'] == 3

    def test_postprocess_no_class_filtering(self):
        """Test postprocessing without class filtering."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5)

        # Mock detection output
        output = {
            'boxes': torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8])
        }

        detections = detector.postprocess(output, (480, 640))

        # Should have all detections
        assert len(detections) == 2

    def test_postprocess_creates_detection_objects(self):
        """Test postprocessing creates proper Detection objects."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5)

        # Mock detection output
        output = {
            'boxes': torch.tensor([[10.5, 20.5, 30.5, 40.5]]),
            'labels': torch.tensor([5]),
            'scores': torch.tensor([0.95])
        }

        detections = detector.postprocess(output, (480, 640))

        assert len(detections) == 1
        det = detections[0]

        # Check detection properties
        assert 'bbox' in det
        assert 'class_id' in det
        assert 'confidence' in det

        assert det['class_id'] == 5
        assert abs(det['confidence'] - 0.95) < 0.01

        # Check bounding box coordinates
        x1, y1, x2, y2 = det['bbox'].xyxy
        assert abs(x1 - 10.5) < 0.01
        assert abs(y1 - 20.5) < 0.01
        assert abs(x2 - 30.5) < 0.01
        assert abs(y2 - 40.5) < 0.01

    def test_postprocess_empty_result(self):
        """Test postprocessing with no detections passing threshold."""
        detector = TorchVisionObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.9)

        # Mock detection output with low scores
        output = {
            'boxes': torch.tensor([[10, 20, 30, 40]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.3])  # Below threshold
        }

        detections = detector.postprocess(output, (480, 640))

        assert len(detections) == 0

    def test_default_model(self):
        """Test default model is mobilenet."""
        detector = TorchVisionObjectDetector(device='cpu')

        assert detector is not None
        assert detector.model is not None
        assert detector.conf_threshold == 0.5
        assert detector.classes is None
