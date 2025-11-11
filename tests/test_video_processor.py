"""Unit tests for VideoProcessor class."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.video_processor import VideoProcessor
from src.detecting.properties.bounding_box import BoundingBox
from src.detecting.detection import Detection


class TestVideoProcessor:
    """Test VideoProcessor functionality."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = Mock()
        detector.detect = Mock(return_value=[])
        return detector

    @pytest.fixture
    def processor(self, mock_detector, tmp_path):
        """Create a VideoProcessor instance for testing."""
        # Use a dummy path since we won't actually run the video
        video_path = str(tmp_path / "dummy.mp4")
        return VideoProcessor(
            video_path=video_path,
            detector=mock_detector,
            max_dimension=640,
            skip_frames=3
        )

    def test_initialization(self, processor):
        """Test VideoProcessor can be initialized."""
        assert processor.max_dimension == 640
        assert processor.skip_frames == 3
        assert processor.frame_num == 0
        assert processor.continuous_mode == False
        assert processor.fps_samples == []
        assert processor.window_name == 'Video Tracking'

    def test_prepare_detection_frame_no_resize(self, processor):
        """Test frame preparation when no resize is needed."""
        # Small frame that doesn't need resizing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection_frame, scale_factor = processor._prepare_detection_frame(frame)

        assert detection_frame.shape == frame.shape
        assert scale_factor == 1.0

    def test_prepare_detection_frame_with_resize(self, processor):
        """Test frame preparation with resizing."""
        # Large frame that needs resizing
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detection_frame, scale_factor = processor._prepare_detection_frame(frame)

        # Should be resized to max_dimension=640
        assert detection_frame.shape[1] == 640  # width
        assert detection_frame.shape[0] == 360  # height (proportional)
        assert abs(scale_factor - 640/1920) < 0.001

    def test_prepare_detection_frame_no_max_dimension(self, mock_detector, tmp_path):
        """Test frame preparation without max_dimension set."""
        processor = VideoProcessor(
            video_path=str(tmp_path / "dummy.mp4"),
            detector=mock_detector,
            max_dimension=None  # No resizing
        )

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detection_frame, scale_factor = processor._prepare_detection_frame(frame)

        assert detection_frame.shape == frame.shape
        assert scale_factor == 1.0

    def test_scale_detections(self, processor):
        """Test scaling detection bounding boxes."""
        # Create mock detections
        det1 = Detection({'bbox': BoundingBox(100, 50, 200, 150)})
        det2 = Detection({'bbox': BoundingBox(300, 100, 400, 200)})
        detections = [det1, det2]

        # Scale by factor of 0.5 (detection was at half resolution)
        scale_factor = 0.5
        scaled = processor._scale_detections(detections, scale_factor)

        # Check first detection scaled correctly
        x1, y1, x2, y2 = scaled[0]['bbox'].xyxy
        assert x1 == 200.0  # 100 / 0.5
        assert y1 == 100.0  # 50 / 0.5
        assert x2 == 400.0  # 200 / 0.5
        assert y2 == 300.0  # 150 / 0.5

        # Check second detection scaled correctly
        x1, y1, x2, y2 = scaled[1]['bbox'].xyxy
        assert x1 == 600.0  # 300 / 0.5
        assert y1 == 200.0  # 100 / 0.5
        assert x2 == 800.0  # 400 / 0.5
        assert y2 == 400.0  # 200 / 0.5

    def test_render_detections(self, processor):
        """Test rendering detections on frame."""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock detections
        mock_bbox = Mock()
        mock_bbox.draw = Mock()
        det = Detection({'bbox': mock_bbox})
        detections = [det]

        # Render detections
        processor._render_detections(frame, detections)

        # Verify draw was called
        mock_bbox.draw.assert_called_once()
        call_args = mock_bbox.draw.call_args
        assert call_args[0][0] is frame  # First arg should be the frame

    def test_detect_objects_with_no_detector(self, tmp_path):
        """Test detection with no detector returns empty list."""
        processor = VideoProcessor(
            video_path=str(tmp_path / "dummy.mp4"),
            detector=None
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, elapsed = processor._detect_objects(frame)

        assert detections == []
        assert elapsed >= 0

    def test_skip_frames_parameter(self, mock_detector, tmp_path):
        """Test skip_frames parameter is enforced to be >= 1."""
        # Should accept valid values
        proc1 = VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=5)
        assert proc1.skip_frames == 5

        # Should clamp invalid values to minimum 1
        proc2 = VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=0)
        assert proc2.skip_frames == 1

        proc3 = VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=-5)
        assert proc3.skip_frames == 1
