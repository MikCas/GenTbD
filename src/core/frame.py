"""
Frame abstraction for video processing with temporal metadata.

This module provides the Frame class that encapsulates video frame data
along with temporal and spatial metadata needed for tracking and multi-camera support.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional
import cv2


@dataclass
class Frame:
    """
    Encapsulates a video frame with metadata for tracking and processing.

    This class wraps a raw video frame (numpy array) with essential metadata
    for multi-object tracking, ReID, and multi-camera systems.

    Attributes:
        data: The frame image as numpy array (H, W, C) in BGR format
        frame_id: Sequential frame number starting from 0
        timestamp: Time in seconds since video/stream start
        source_id: Identifier for the video source (e.g., "camera_1", "video.mp4")
        original_shape: Original frame dimensions before any scaling (H, W, C)
        scale_factor: Cumulative scale factor applied to original frame (1.0 = no scaling)

    Example:
        >>> frame = Frame(
        ...     data=np.zeros((480, 640, 3), dtype=np.uint8),
        ...     frame_id=42,
        ...     timestamp=1.4,
        ...     source_id="camera_1"
        ... )
        >>> frame.width
        640
        >>> frame.height
        480
        >>> scaled = frame.scaled(0.5)  # Half resolution
        >>> scaled.width
        320
    """
    data: np.ndarray
    frame_id: int
    timestamp: float
    source_id: str
    original_shape: Optional[tuple] = None
    scale_factor: float = 1.0

    def __post_init__(self):
        """Validate frame data and set original_shape if not provided."""
        if self.data is None or not isinstance(self.data, np.ndarray):
            raise ValueError("Frame data must be a numpy array")

        if len(self.data.shape) != 3:
            raise ValueError(f"Frame data must be 3D (H, W, C), got shape {self.data.shape}")

        # Set original_shape if this is the first frame (not a scaled copy)
        if self.original_shape is None:
            self.original_shape = self.data.shape

    @property
    def shape(self) -> tuple:
        """Get current frame shape (H, W, C)."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Get current frame height in pixels."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Get current frame width in pixels."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Get number of color channels (typically 3 for BGR)."""
        return self.data.shape[2]

    def scaled(self, scale_factor: float) -> 'Frame':
        """
        Return a new Frame with scaled image data.

        This creates a NEW frame object with scaled data, preserving
        the original frame's metadata but updating scale_factor.

        Args:
            scale_factor: Scale multiplier (e.g., 0.5 = half size, 2.0 = double size)

        Returns:
            New Frame with scaled image and updated scale_factor

        Example:
            >>> original = Frame(data=frame_data, frame_id=0, timestamp=0.0, source_id="cam1")
            >>> half_size = original.scaled(0.5)
            >>> half_size.scale_factor
            0.5
            >>> quarter_size = half_size.scaled(0.5)  # Chain scaling
            >>> quarter_size.scale_factor
            0.25
        """
        if scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {scale_factor}")

        new_width = int(self.width * scale_factor)
        new_height = int(self.height * scale_factor)

        # Ensure minimum dimensions of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        scaled_data = cv2.resize(self.data, (new_width, new_height))

        return Frame(
            data=scaled_data,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            source_id=self.source_id,
            original_shape=self.original_shape,  # Preserve original
            scale_factor=self.scale_factor * scale_factor  # Cumulative
        )

    def copy(self) -> 'Frame':
        """
        Create a deep copy of this frame.

        Returns:
            New Frame with copied image data and same metadata
        """
        return Frame(
            data=self.data.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            source_id=self.source_id,
            original_shape=self.original_shape,
            scale_factor=self.scale_factor
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Frame(frame_id={self.frame_id}, "
            f"timestamp={self.timestamp:.3f}s, "
            f"source='{self.source_id}', "
            f"shape={self.shape}, "
            f"scale={self.scale_factor:.2f})"
        )
