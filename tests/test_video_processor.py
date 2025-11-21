"""Unit tests for VideoProcessor class."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.video_processor import VideoProcessor
from src.core.properties import BoundingBox
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
        source = str(tmp_path / "dummy.mp4")
        return VideoProcessor(
            source=source,
            detector=mock_detector,
            max_dimension=640,
            skip_frames=3
        )

    def test_initialization(self, processor):
        """Test VideoProcessor can be initialized."""
        assert processor.max_dimension == 640
        assert processor.skip_frames == 3
        assert processor.frame_num == 0
        assert processor.continuous_mode == True
        assert len(processor.fps_samples) == 0  # deque is empty
        assert processor.window_name == 'Video Tracking'

    def test_frame_no_resize(self, processor):
        """Test Frame scaling when no resize is needed."""
        from src.core import Frame

        # Small frame that doesn't need resizing
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=frame_data, frame_id=0, timestamp=0.0, source_id="test")

        # With max_dimension=640, this frame doesn't need scaling
        if processor.max_dimension and max(frame.height, frame.width) > processor.max_dimension:
            scale = processor.max_dimension / max(frame.height, frame.width)
            scaled = frame.scaled(scale)
        else:
            scaled = frame

        assert scaled.shape == frame.shape
        assert scaled.scale_factor == 1.0

    def test_frame_with_resize(self, processor):
        """Test Frame scaling with resizing."""
        from src.core import Frame

        # Large frame that needs resizing
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame(data=frame_data, frame_id=0, timestamp=0.0, source_id="test")

        # Should be resized to max_dimension=640
        if processor.max_dimension and max(frame.height, frame.width) > processor.max_dimension:
            scale = processor.max_dimension / max(frame.height, frame.width)
            scaled = frame.scaled(scale)
        else:
            scaled = frame

        assert scaled.width == 640  # width
        assert scaled.height == 360  # height (proportional)
        assert abs(scaled.scale_factor - 640/1920) < 0.001

    def test_frame_no_max_dimension(self, mock_detector, tmp_path):
        """Test Frame scaling without max_dimension set."""
        from src.core import Frame

        processor = VideoProcessor(
            source=str(tmp_path / "dummy.mp4"),
            detector=mock_detector,
            max_dimension=None  # No resizing
        )

        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame(data=frame_data, frame_id=0, timestamp=0.0, source_id="test")

        # No scaling should happen
        if processor.max_dimension and max(frame.height, frame.width) > processor.max_dimension:
            scale = processor.max_dimension / max(frame.height, frame.width)
            scaled = frame.scaled(scale)
        else:
            scaled = frame

        assert scaled.shape == frame.shape
        assert scaled.scale_factor == 1.0

    def test_scale_detections(self, processor):
        """Test scaling detection bounding boxes using bbox.scale()."""
        # Create mock detections
        det1 = Detection({'bbox': BoundingBox(100, 50, 200, 150)})
        det2 = Detection({'bbox': BoundingBox(300, 100, 400, 200)})
        detections = [det1, det2]

        # Scale by factor of 0.5 (detection was at half resolution)
        scale_factor = 0.5
        scaled = processor._scale_detections(detections, scale_factor)

        # Check first detection scaled correctly (using bbox.scale())
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

    def test_render_frame(self, processor):
        """Test rendering all visualizations on frame."""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock detections
        mock_bbox = Mock()
        mock_bbox.draw = Mock()
        det = Detection({'bbox': mock_bbox})
        detections = [det]

        # Render frame
        processor._render_frame(frame, detections)

        # Verify draw was called
        mock_bbox.draw.assert_called_once()
        call_args = mock_bbox.draw.call_args
        assert call_args[0][0] is frame  # First arg should be the frame

    def test_skip_frames_parameter(self, mock_detector, tmp_path):
        """Test skip_frames parameter validation."""
        # Should accept valid values
        proc1 = VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=5)
        assert proc1.skip_frames == 5

        # Should raise error for invalid values
        with pytest.raises(ValueError):
            VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=0)

        with pytest.raises(ValueError):
            VideoProcessor(str(tmp_path / "dummy.mp4"), mock_detector, skip_frames=-5)

    def test_webcam_source(self, mock_detector):
        """Test VideoProcessor accepts camera index as source."""
        processor = VideoProcessor(
            source=0,  # Camera index
            detector=mock_detector
        )
        assert processor.source == 0
