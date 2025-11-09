"""Integration tests for ObjectDetector.

Integration tests test how multiple components work together.
These tests may be slower than unit tests because they load real models.
"""

import pytest
import torch
import numpy as np
from src.detecting.detectors import ObjectDetector
from src.detecting.detection import Detection


@pytest.mark.integration
class TestObjectDetectorLoading:
    """Test detector model loading."""

    def test_load_fasterrcnn_cpu(self):
        """Test loading FasterRCNN on CPU."""
        detector = ObjectDetector.from_fasterrcnn_resnet50(
            device='cpu',
            conf_threshold=0.5
        )

        assert detector is not None
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.5

    @pytest.mark.gpu
    def test_load_fasterrcnn_mps(self):
        """Test loading FasterRCNN on MPS (if available)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        detector = ObjectDetector.from_fasterrcnn_resnet50(
            device='mps',
            conf_threshold=0.5
        )

        assert detector is not None
        assert detector.device == 'mps'

    def test_load_retinanet_cpu(self):
        """Test loading RetinaNet on CPU."""
        detector = ObjectDetector.from_retinanet(
            device='cpu',
            conf_threshold=0.5
        )

        assert detector is not None
        assert detector.device == 'cpu'

    def test_detector_with_class_filter(self):
        """Test detector with class filtering."""
        detector = ObjectDetector.from_fasterrcnn_resnet50(
            device='cpu',
            conf_threshold=0.5,
            classes=[0]  # Person class only
        )

        assert detector.classes == [0]


@pytest.mark.integration
class TestObjectDetectorInference:
    """Test detector inference on images."""

    def test_detect_on_blank_image(self, pretrained_detector, sample_image):
        """Test detection on a blank image (should return few/no detections)."""
        detections = pretrained_detector.detect(sample_image)

        assert isinstance(detections, list)
        # Blank image shouldn't have many detections
        assert len(detections) < 10

    def test_detect_returns_detection_objects(self, pretrained_detector, sample_image):
        """Test that detect returns Detection objects."""
        detections = pretrained_detector.detect(sample_image)

        for det in detections:
            assert isinstance(det, Detection)
            assert 'bbox' in det
            assert 'class_id' in det
            assert 'confidence' in det

    def test_detect_confidence_filtering(self, sample_image):
        """Test that confidence threshold filters detections correctly."""
        detector_low = ObjectDetector.from_fasterrcnn_resnet50(
            device='cpu',
            conf_threshold=0.1
        )
        detector_high = ObjectDetector.from_fasterrcnn_resnet50(
            device='cpu',
            conf_threshold=0.9
        )

        detections_low = detector_low.detect(sample_image)
        detections_high = detector_high.detect(sample_image)

        # Lower threshold should give more detections
        assert len(detections_low) >= len(detections_high)

    @pytest.mark.slow
    def test_detect_on_multiple_images(self, pretrained_detector):
        """Test detection on multiple different images."""
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        for img in images:
            detections = pretrained_detector.detect(img)
            assert isinstance(detections, list)

    def test_detect_preserves_image(self, pretrained_detector, sample_image):
        """Test that detection doesn't modify original image."""
        original = sample_image.copy()
        _ = pretrained_detector.detect(sample_image)

        assert np.array_equal(sample_image, original)


@pytest.mark.integration
class TestObjectDetectorPipeline:
    """Test the detection pipeline stages."""

    def test_preprocess_output_shape(self, pretrained_detector, sample_image):
        """Test that preprocess returns correct tensor shape."""
        tensor = pretrained_detector.preprocess(sample_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 480, 640)  # (C, H, W)
        assert tensor.dtype == torch.float32

    def test_preprocess_value_range(self, pretrained_detector, sample_image):
        """Test that preprocessed values are in [0, 1] range."""
        tensor = pretrained_detector.preprocess(sample_image)

        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_inference_output_format(self, pretrained_detector, sample_image):
        """Test that inference returns expected output format."""
        tensor = pretrained_detector.preprocess(sample_image)
        output = pretrained_detector.inference(tensor)

        assert isinstance(output, dict)
        assert 'boxes' in output
        assert 'labels' in output
        assert 'scores' in output

    def test_postprocess_output_format(self, pretrained_detector, sample_image):
        """Test that postprocess returns list of Detection objects."""
        tensor = pretrained_detector.preprocess(sample_image)
        output = pretrained_detector.inference(tensor)
        detections = pretrained_detector.postprocess(output, sample_image.shape[:2])

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)


@pytest.mark.integration
@pytest.mark.gpu
class TestObjectDetectorDevices:
    """Test detector on different devices."""

    @pytest.fixture(params=['cpu', 'mps'])
    def detector_on_device(self, request):
        """Create detector on specified device."""
        device = request.param

        if device == 'mps' and not torch.backends.mps.is_available():
            pytest.skip(f"{device} not available")

        return ObjectDetector.from_fasterrcnn_resnet50(
            device=device,
            conf_threshold=0.5
        )

    def test_detector_device_consistency(self, detector_on_device, sample_image):
        """Test that detector works consistently across devices."""
        detections = detector_on_device.detect(sample_image)

        assert isinstance(detections, list)
        for det in detections:
            # Detections should be on CPU after postprocessing
            assert isinstance(det['bbox'].xyxy, tuple)


@pytest.mark.integration
@pytest.mark.benchmark
class TestObjectDetectorPerformance:
    """Benchmark tests for detector performance."""

    def test_detection_speed_cpu(self, benchmark, pretrained_detector, sample_image):
        """Benchmark detection speed on CPU."""
        result = benchmark(pretrained_detector.detect, sample_image)
        assert isinstance(result, list)

    @pytest.mark.slow
    def test_batch_vs_single_inference(self, pretrained_detector):
        """Compare batch vs single-image inference (for future optimization)."""
        import time

        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]

        # Single image inference
        start = time.time()
        for img in images:
            pretrained_detector.detect(img)
        single_time = time.time() - start

        # This test documents current behavior
        # When you add batch processing later, update this test
        assert single_time > 0
