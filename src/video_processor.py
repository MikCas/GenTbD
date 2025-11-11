"""Video processing with detection and visualization.

This module provides the VideoProcessor class for running object detection
on video files with interactive controls and optional output saving.
"""

import cv2
import time
import logging
import os
from datetime import datetime
from typing import Optional
from .detecting.properties.bounding_box import BoundingBox

logger = logging.getLogger(__name__)

# UI Constants
FPS_WINDOW = 30
TEXT_COLOR = (255, 0, 0)  # BGR blue
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

class VideoProcessor:
    """Processes video with detection, visualization, and interactive controls.

    Features:
    - Object detection on each frame
    - FPS tracking and display
    - Interactive controls (step/continuous mode)
    - Frame saving
    - Optional video output

    Controls:
    - 'c': Toggle continuous/step mode
    - SPACE: Next frame (in step mode)
    - 's': Save current frame as image
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
        """
        self.source = source
        self.detector = detector
        self.save_output = save_output
        self.output_path = output_path
        self.max_dimension = max_dimension
        self.skip_frames = max(1, skip_frames)

        # Video resources
        self.cap = None
        self.out = None

        # Video properties (minimal - only what's needed)
        self.video_fps = None
        self.total_frames = None

        # State
        self.frame_num = 0
        self.continuous_mode = False
        self.fps_samples = []
        self.window_name = 'Video Tracking'

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
            # Camera index (0, 1, etc.)
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
        cached_frame = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_num += 1
            should_detect = (self.frame_num - 1) % self.skip_frames == 0

            if should_detect:
                # Run detection and render
                detections, elapsed = self._detect_objects(frame)
                self.fps_samples.append(1.0 / elapsed if elapsed > 0 else 0)
                self._render_frame(frame, detections)

                # Cache only if frame skipping is enabled
                if self.skip_frames > 1:
                    cached_frame = frame.copy()

                display_frame = frame
            else:
                # Use cached frame (frame skipping mode)
                display_frame = cached_frame

            # Display and save
            self._show_frame(display_frame)
            if self.out:
                self.out.write(display_frame)

            # Handle keyboard input
            if self._handle_input(display_frame):
                break

    # =========================================================================
    # Detection Pipeline
    # =========================================================================

    def _detect_objects(self, frame):
        """Run object detection on frame.

        Handles frame resizing and bounding box scaling automatically.

        Args:
            frame: Input frame (will not be modified)

        Returns:
            Tuple of (detections, elapsed_time)
        """
        start_time = time.time()

        # Scale frame for detection if requested
        scaled_frame, scale_factor = self._scale_frame(frame)

        # Run detection
        detections = self.detector.detect(scaled_frame)

        # Scale bounding boxes back to original size if needed
        if scale_factor != 1.0:
            detections = self._scale_detections(detections, scale_factor)

        elapsed = time.time() - start_time
        return detections, elapsed

    def _scale_frame(self, frame):
        """Scale frame to max_dimension if needed.

        Args:
            frame: Original frame

        Returns:
            Tuple of (scaled_frame, scale_factor)
        """
        if not self.max_dimension:
            return frame, 1.0

        h, w = frame.shape[:2]
        max_dim = max(h, w)

        if max_dim <= self.max_dimension:
            return frame, 1.0

        # Calculate scale factor and resize
        scale_factor = self.max_dimension / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        scaled_frame = cv2.resize(frame, (new_w, new_h))

        return scaled_frame, scale_factor

    def _scale_detections(self, detections, scale_factor):
        """Scale detection bounding boxes from detection resolution to display resolution.

        Args:
            detections: List of Detection objects
            scale_factor: Factor used to resize frame (target_size / original_size)

        Returns:
            List of Detection objects with scaled bounding boxes
        """
        for det in detections:
            det['bbox'] = det['bbox'].scale(1.0 / scale_factor)
        return detections

    # =========================================================================
    # VISUALIZATION (centralized for detections, tracks, keypoints)
    # =========================================================================

    def _render_frame(self, frame, detections, tracks=None, keypoints=None):
        """Render all visualizations on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            detections: List of Detection objects to render
            tracks: List of Track objects to render (future)
            keypoints: List of Keypoint objects to render (future)
        """
        # Draw detections
        for det in detections:
            det['bbox'].draw(frame, color=(0, 255, 0))

        # Draw tracks (future)
        if tracks:
            self._draw_tracks(frame, tracks)

        # Draw keypoints (future)
        if keypoints:
            self._draw_keypoints(frame, keypoints)

        # Draw overlay
        self._draw_overlay(frame, detections, tracks)

    def _draw_tracks(self, frame, tracks):
        """Draw tracking IDs and trajectories on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            tracks: List of Track objects to render

        Future placeholder for tracking visualization.
        """
        pass

    def _draw_keypoints(self, frame, keypoints):
        """Draw pose keypoints and skeleton on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            keypoints: List of Keypoint objects to render

        Future placeholder for keypoint visualization.
        """
        pass

    def _draw_overlay(self, frame, detections, tracks=None):
        """Draw text overlay (FPS, frame info, mode, detection count) on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            detections: List of detections (for counting)
            tracks: List of tracks (for counting, future)
        """
        # Calculate rolling average FPS
        recent_fps = self.fps_samples[-FPS_WINDOW:] if self.fps_samples else [0]
        avg_fps = sum(recent_fps) / len(recent_fps)

        mode = "CONTINUOUS" if self.continuous_mode else "STEP"

        # Frame counter text (handle live streams)
        if self.is_live_stream:
            frame_text = f"Frame: {self.frame_num}"
        else:
            frame_text = f"Frame: {self.frame_num}/{self.total_frames}"

        # Draw text overlay
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, frame_text, (10, 70),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Mode: {mode}", (10, 110),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        if len(detections) > 0:
            cv2.putText(frame, f"Detections: {len(detections)}",
                       (10, 150), TEXT_FONT, 1, TEXT_COLOR, 2)

    # =========================================================================
    # USER INTERACTION
    # =========================================================================

    def _show_frame(self, frame):
        """Display frame in window.

        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)

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
        elif key == ord('c'):
            self.continuous_mode = not self.continuous_mode
            mode = "CONTINUOUS" if self.continuous_mode else "STEP"
            logger.info(f"Mode: {mode}")
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
