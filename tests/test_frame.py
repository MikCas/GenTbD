"""Unit tests for Frame class."""

import pytest
import numpy as np
from src.core import Frame


class TestFrame:
    """Test Frame functionality."""

    def test_initialization(self):
        """Test Frame initializes with all required parameters."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            data=data,
            frame_id=42,
            timestamp=1.4,
            source_id="camera_1"
        )

        assert frame.frame_id == 42
        assert frame.timestamp == 1.4
        assert frame.source_id == "camera_1"
        assert frame.scale_factor == 1.0
        assert frame.original_shape == (480, 640, 3)

    def test_initialization_sets_original_shape(self):
        """Test original_shape is automatically set from data shape."""
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        assert frame.original_shape == (100, 200, 3)

    def test_initialization_preserves_original_shape(self):
        """Test original_shape is preserved when explicitly provided."""
        data = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(
            data=data,
            frame_id=0,
            timestamp=0.0,
            source_id="test",
            original_shape=(100, 200, 3),
            scale_factor=0.5
        )

        assert frame.original_shape == (100, 200, 3)
        assert frame.shape == (50, 100, 3)

    def test_initialization_invalid_data_none(self):
        """Test Frame raises error when data is None."""
        with pytest.raises(ValueError, match="Frame data must be a numpy array"):
            Frame(data=None, frame_id=0, timestamp=0.0, source_id="test")

    def test_initialization_invalid_data_not_array(self):
        """Test Frame raises error when data is not numpy array."""
        with pytest.raises(ValueError, match="Frame data must be a numpy array"):
            Frame(data=[1, 2, 3], frame_id=0, timestamp=0.0, source_id="test")

    def test_initialization_invalid_data_wrong_dimensions(self):
        """Test Frame raises error when data is not 3D."""
        with pytest.raises(ValueError, match="Frame data must be 3D"):
            data = np.zeros((480, 640), dtype=np.uint8)  # Only 2D
            Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

    def test_shape_property(self):
        """Test shape property returns correct tuple."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        assert frame.shape == (480, 640, 3)

    def test_height_property(self):
        """Test height property returns correct value."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        assert frame.height == 480

    def test_width_property(self):
        """Test width property returns correct value."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        assert frame.width == 640

    def test_channels_property(self):
        """Test channels property returns correct value."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        assert frame.channels == 3

    def test_scaled_half_size(self):
        """Test scaling frame to half size."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=42, timestamp=1.4, source_id="camera_1")

        scaled = frame.scaled(0.5)

        assert scaled.width == 320
        assert scaled.height == 240
        assert scaled.scale_factor == 0.5
        assert scaled.frame_id == 42  # Metadata preserved
        assert scaled.timestamp == 1.4
        assert scaled.source_id == "camera_1"
        assert scaled.original_shape == (480, 640, 3)  # Original preserved

    def test_scaled_double_size(self):
        """Test scaling frame to double size."""
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        scaled = frame.scaled(2.0)

        assert scaled.width == 400
        assert scaled.height == 200
        assert scaled.scale_factor == 2.0

    def test_scaled_chain_scaling(self):
        """Test chaining multiple scale operations."""
        data = np.zeros((400, 800, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        # Scale to half, then half again
        half = frame.scaled(0.5)
        quarter = half.scaled(0.5)

        assert quarter.width == 200
        assert quarter.height == 100
        assert quarter.scale_factor == 0.25  # Cumulative: 0.5 * 0.5
        assert quarter.original_shape == (400, 800, 3)

    def test_scaled_invalid_negative(self):
        """Test scaling with negative factor raises error."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        with pytest.raises(ValueError, match="Scale factor must be positive"):
            frame.scaled(-0.5)

    def test_scaled_invalid_zero(self):
        """Test scaling with zero factor raises error."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        with pytest.raises(ValueError, match="Scale factor must be positive"):
            frame.scaled(0.0)

    def test_scaled_minimum_dimensions(self):
        """Test scaling respects minimum 1x1 dimensions."""
        data = np.zeros((2, 2, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        # Scale very small - should clamp to 1x1
        scaled = frame.scaled(0.1)

        assert scaled.width >= 1
        assert scaled.height >= 1

    def test_scaled_preserves_original_data(self):
        """Test scaling does not modify original frame."""
        data = np.ones((100, 200, 3), dtype=np.uint8) * 255
        frame = Frame(data=data, frame_id=0, timestamp=0.0, source_id="test")

        scaled = frame.scaled(0.5)

        # Original frame unchanged
        assert frame.width == 200
        assert frame.height == 100
        assert frame.scale_factor == 1.0
        # Scaled frame changed
        assert scaled.width == 100
        assert scaled.height == 50

    def test_copy(self):
        """Test copying creates independent frame."""
        data = np.ones((100, 200, 3), dtype=np.uint8) * 128
        frame = Frame(
            data=data,
            frame_id=42,
            timestamp=1.4,
            source_id="camera_1",
            scale_factor=0.5
        )

        copied = frame.copy()

        # Same metadata
        assert copied.frame_id == 42
        assert copied.timestamp == 1.4
        assert copied.source_id == "camera_1"
        assert copied.scale_factor == 0.5
        assert copied.shape == (100, 200, 3)

        # Independent data
        copied.data[0, 0, 0] = 255
        assert frame.data[0, 0, 0] == 128  # Original unchanged

    def test_repr(self):
        """Test string representation contains key information."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data, frame_id=42, timestamp=1.456, source_id="camera_1")

        repr_str = repr(frame)

        assert "frame_id=42" in repr_str
        assert "1.456s" in repr_str
        assert "camera_1" in repr_str
        assert "(480, 640, 3)" in repr_str
        assert "scale=1.00" in repr_str

    def test_different_color_formats(self):
        """Test Frame works with different channel counts."""
        # Grayscale (1 channel) - should fail
        gray_data = np.zeros((100, 200, 1), dtype=np.uint8)
        frame_gray = Frame(data=gray_data, frame_id=0, timestamp=0.0, source_id="test")
        assert frame_gray.channels == 1

        # RGB (3 channels)
        rgb_data = np.zeros((100, 200, 3), dtype=np.uint8)
        frame_rgb = Frame(data=rgb_data, frame_id=0, timestamp=0.0, source_id="test")
        assert frame_rgb.channels == 3

        # RGBA (4 channels)
        rgba_data = np.zeros((100, 200, 4), dtype=np.uint8)
        frame_rgba = Frame(data=rgba_data, frame_id=0, timestamp=0.0, source_id="test")
        assert frame_rgba.channels == 4

    def test_metadata_for_tracking(self):
        """Test Frame provides all metadata needed for tracking."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            data=data,
            frame_id=100,
            timestamp=3.333,
            source_id="front_camera"
        )

        # Check all tracking-relevant metadata is accessible
        assert frame.frame_id == 100  # For temporal ordering
        assert frame.timestamp == 3.333  # For velocity calculation
        assert frame.source_id == "front_camera"  # For multi-camera
        assert frame.original_shape is not None  # For coordinate conversion
        assert frame.scale_factor == 1.0  # For coordinate scaling
