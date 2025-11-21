"""Video processing with detection and visualization.

This module provides the VideoProcessor class for running object detection
on video files with interactive controls and optional output saving.
"""

import cv2
import time
import logging
import os
from collections import deque
from datetime import datetime
from typing import Optional
from .core import Frame
from .core.properties import BoundingBox
from .visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

# UI Constants
FPS_WINDOW = 30

class VideoProcessor:
    """Processes video with detection, visualization, and interactive controls.

    Features:
    - Object detection on each frame
    - FPS tracking and display
    - Interactive controls (play/pause)
    - Frame saving
    - Optional video output

    Controls:
    - SPACE: Play/Pause
    - →: Next frame (when paused)
    - 's': Save current frame
    - 'q': Quit
    """

    def __init__(self, source: str, detector, save_output: bool = False,
                 output_path: Optional[str] = None, max_dimension: Optional[int] = None,
                 skip_frames: int = 1):
        """Initialize video processor.

        Args:
            source: Path to video file, camera index (0, 1, etc.), or RTSP URL
            detector: Detector instance (e.g., ObjectDetector)
            save_output: Whether to save processed video
            output_path: Output video path (auto-generated if None)
            max_dimension: Resize frames to this max dimension before detection (None = no resize)
            skip_frames: Process every Nth frame (1 = all frames, 5 = every 5th)

        Raises:
            ValueError: If max_dimension or skip_frames are invalid
        """
        # Validate inputs
        if max_dimension is not None and max_dimension <= 0:
            raise ValueError(f"max_dimension must be positive, got {max_dimension}")
        if skip_frames <= 0:
            raise ValueError(f"skip_frames must be positive, got {skip_frames}")
        if skip_frames > 100:
            logger.warning(
                f"skip_frames={skip_frames} is very high - most frames will be skipped"
            )

        self.source = source
        self.detector = detector
        self.save_output = save_output
        self.output_path = output_path
        self.max_dimension = max_dimension
        self.skip_frames = skip_frames

        # Video resources
        self.cap = None
        self.out = None

        # Video properties (minimal - only what's needed)
        self.video_fps = None
        self.total_frames = None

        # State
        self.frame_num = 0
        self.continuous_mode = True  # Start playing automatically
        # Use deque with maxlen to prevent unbounded growth
        self.fps_samples = deque(maxlen=FPS_WINDOW)
        self.window_name = 'Video Tracking'
        self.visualizer = Visualizer(self.window_name)

    @property
    def is_live_stream(self):
        """Check if source is a live stream (camera/RTSP) vs video file."""
        return self.total_frames is None or self.total_frames == 0

    def run(self):
        """Main processing loop. Sets up video, processes frames, and cleans up."""
        try:
            self._setup_video()
            self._process_loop()
        finally:
            self._cleanup()

    def _setup_video(self):
        """Setup video capture and optional output writer."""
        # Handle different source types (file path, camera index, RTSP URL)
        if isinstance(self.source, int):
            # Camera index (0, 1, etc.) - validate range
            if self.source < 0 or self.source > 10:
                raise ValueError(
                    f"Invalid webcam index: {self.source}. Must be between 0 and 10."
                )
            self.cap = cv2.VideoCapture(self.source)
        elif self.source.startswith(('rtsp://', 'http://', 'https://')):
            # RTSP or HTTP stream
            self.cap = cv2.VideoCapture(self.source)
        else:
            # File path - check if it exists
            if not os.path.exists(self.source):
                raise ValueError(f"Video file not found: {self.source}")
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

        # Get video properties
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Log video info
        if self.is_live_stream:
            logger.info(f"Opened live stream: {width}x{height}, {self.video_fps:.1f} FPS")
        else:
            logger.info(f"Loaded video: {width}x{height}, {self.video_fps:.1f} FPS, {self.total_frames} frames")

        # Setup output writer if needed
        if self.save_output:
            self._setup_output_writer(width, height)

    def _setup_output_writer(self, width, height):
        """Setup video output writer.

        Args:
            width: Frame width
            height: Frame height
        """
        if not self.output_path:
            self.output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.video_fps, (width, height))

        if not self.out.isOpened():
            raise ValueError(f"Could not open video writer: {self.output_path}")

        logger.info(f"Saving to: {self.output_path}")

    def _process_loop(self):
        """Main processing loop - read frames, detect, visualize, handle input."""
        frame_count = 0  # 0-indexed frame counter
        detections = []  # Persist detections between frames

        while True:
            # Start timing for full loop (accurate FPS measurement)
            loop_start_time = time.time()
            
            ret, frame_data = self.cap.read()
            if not ret:
                break

            # Create Frame object with metadata (0-indexed)
            # Handle video_fps = 0 or None (some webcams/codecs)
            if self.video_fps and self.video_fps > 0:
                timestamp = (frame_count + 1) / self.video_fps
            else:
                timestamp = (frame_count + 1) * (1.0 / 30.0)  # Default 30 FPS estimate

            frame = Frame(
                data=frame_data,
                frame_id=frame_count,  # 0-indexed
                timestamp=timestamp,
                source_id=str(self.source)
            )

            self.frame_num = frame_count + 1  # 1-indexed for display
            should_detect = frame_count % self.skip_frames == 0

            if should_detect:
                # Run detection (no need to time separately anymore)
                detections, _ = self._detect_objects(frame)

            # Always render (current or cached) detections on the NEW frame
            # This prevents flickering by showing the last known detections on skipped frames
            display_frame = frame.data.copy()
            self._render_frame(display_frame, detections)

            # Display and save (only if frame available)
            if display_frame is not None:
                self._show_frame(display_frame)
                if self.out:
                    self.out.write(display_frame)

            # Handle keyboard input
            if self._handle_input(display_frame):
                break

            # Calculate full loop FPS (detection + rendering + display)
            loop_time = time.time() - loop_start_time
            self.fps_samples.append(1.0 / loop_time if loop_time > 0 else 0)

            # Increment frame counter
            frame_count += 1

    # =========================================================================
    # Detection Pipeline
    # =========================================================================

    def _detect_objects(self, frame: Frame):
        """Run object detection on frame.

        Handles frame resizing and bounding box scaling automatically using Frame abstraction.

        Args:
            frame: Input Frame object with metadata

        Returns:
            Tuple of (detections, elapsed_time)
        """
        start_time = time.time()

        # Scale frame for detection if requested (Frame handles scaling)
        if self.max_dimension and max(frame.height, frame.width) > self.max_dimension:
            scale = self.max_dimension / max(frame.height, frame.width)
            detection_frame = frame.scaled(scale)
        else:
            detection_frame = frame

        # Run detection on frame data
        detections = self.detector.detect(detection_frame.data)

        # Scale detections back if frame was scaled
        if detection_frame.scale_factor != 1.0:
            detections = self._scale_detections(detections, detection_frame.scale_factor)

        elapsed = time.time() - start_time
        return detections, elapsed

    def _scale_detections(self, detections, scale_factor):
        """Scale detection bounding boxes and keypoints from detection resolution to display resolution.

        Args:
            detections: List of Detection objects
            scale_factor: Factor used to resize frame (target_size / original_size)

        Returns:
            List of Detection objects with scaled bounding boxes and keypoints
        """
        for det in detections:
            # Scale bounding box
            det['bbox'] = det['bbox'].scale(1.0 / scale_factor)

            # Scale keypoints if present (for KeypointDetector)
            if 'keypoints' in det:
                det['keypoints'] = det['keypoints'].scale(1.0 / scale_factor)

        return detections

    # =========================================================================
    # VISUALIZATION (centralized for detections, tracks, keypoints)
    # =========================================================================

    def _render_frame(self, frame, detections, tracks=None):
        """Render all visualizations on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            detections: List of Detection objects to render
            tracks: List of Track objects to render (future)
        """
        # Calculate rolling average FPS
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        is_paused = not self.continuous_mode

        self.visualizer.render(
            frame=frame,
            detections=detections,
            fps=avg_fps,
            frame_num=self.frame_num,
            total_frames=self.total_frames,
            is_paused=is_paused,
            tracks=tracks
        )


    # =========================================================================
    # USER INTERACTION
    # =========================================================================

    def _show_frame(self, frame):
        """Display frame in window.

        Args:
            frame: Frame to display
        """
        self.visualizer.show(frame)

    def _handle_input(self, frame) -> bool:
        """Handle keyboard input.

        Args:
            frame: Current frame (for saving if 's' pressed)

        Returns:
            True if should quit, False otherwise
        """
        wait_time = 1 if self.continuous_mode else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            return True
        elif key == ord(' '):  # Spacebar: toggle play/pause
            self.continuous_mode = not self.continuous_mode
            mode = "PLAYING" if self.continuous_mode else "PAUSED"
            logger.info(f"Mode: {mode}")
        elif key == 83:  # Right arrow: next frame (when paused)
            if not self.continuous_mode:
                # In step mode, right arrow advances to next frame
                # Return False to continue loop (frame will advance naturally)
                pass
        elif key == ord('s'):
            filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Saved {filename}")

        return False

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def _cleanup(self):
        """Release video resources and print summary."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()

        # Clean up display window
        cv2.destroyAllWindows()

        # Print summary
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        logger.info(f"Processed {self.frame_num} frames @ {avg_fps:.1f} FPS average")
