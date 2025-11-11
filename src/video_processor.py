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
from detecting.properties.bounding_box import BoundingBox
from ui.video_ui import VideoUI

logger = logging.getLogger(__name__)


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

    def __init__(self, video_path: str, detector, save_output: bool = False,
                 output_path: Optional[str] = None, max_dimension: Optional[int] = None,
                 skip_frames: int = 1, ui: Optional[VideoUI] = None):
        """Initialize video processor.

        Args:
            video_path: Path to input video file
            detector: Detector instance (e.g., ObjectDetector)
            save_output: Whether to save processed video
            output_path: Output video path (auto-generated if None)
            max_dimension: Resize frames to this max dimension before detection (None = no resize)
            skip_frames: Process every Nth frame (1 = all frames, 5 = every 5th)
            ui: UI instance for display (None = create default VideoUI)
        """
        self.video_path = video_path
        self.detector = detector
        self.save_output = save_output
        self.output_path = output_path
        self.max_dimension = max_dimension
        self.skip_frames = max(1, skip_frames)

        # UI layer (swappable)
        self.ui = ui if ui is not None else VideoUI()

        # Video resources
        self.cap = None
        self.out = None
        self.props = {}

        # State
        self.frame_num = 0

    def run(self):
        """Main processing loop. Sets up video, processes frames, and cleans up."""
        try:
            self._setup_video()
            self._process_loop()
        finally:
            self._cleanup()

    def _setup_video(self):
        """Setup video capture and optional output writer."""
        if not os.path.exists(self.video_path):
            raise ValueError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.props = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        logger.info(f"Loaded video: {self.props['width']}x{self.props['height']}, "
                   f"{self.props['fps']:.1f} FPS, {self.props['total_frames']} frames")

        if self.save_output:
            if not self.output_path:
                self.output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.props['fps'],
                                      (self.props['width'], self.props['height']))
            if not self.out.isOpened():
                raise ValueError(f"Could not open video writer: {self.output_path}")
            logger.info(f"Saving to: {self.output_path}")

    def _process_loop(self):
        """Main processing loop - read frames, detect, visualize, handle input.

        When skip_frames > 1, we cache the rendered frame and reuse it for skipped frames.
        This ensures bounding boxes match the frame they were detected on.
        """
        cached_display_frame = None  # Cache rendered frame when skipping

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_num += 1

            # Determine if we should run detection on this frame
            should_detect = (self.frame_num - 1) % self.skip_frames == 0

            if should_detect:
                # Run detection and render frame
                detections, elapsed = self._detect_objects(frame)
                self.ui.add_fps_sample(1.0 / elapsed if elapsed > 0 else 0)

                # Render boxes and overlay on frame
                self._render_detections(frame, detections)
                self.ui.draw_overlay(frame, self.frame_num, self.props['total_frames'], detections)

                # Cache this rendered frame for skip frames
                cached_display_frame = frame.copy()
                display_frame = frame
            else:
                # Reuse cached rendered frame (boxes match the detection frame)
                display_frame = cached_display_frame if cached_display_frame is not None else frame

            # Display and save
            self.ui.show_frame(display_frame)
            if self.out:
                self.out.write(display_frame)

            # Handle keyboard input
            key = self.ui.wait_for_input()
            action = self.ui.handle_key(key, display_frame)
            if action == 'quit':
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
        detections = []

        if not self.detector:
            return detections, time.time() - start_time

        # Resize frame for detection if requested
        detection_frame, scale_factor = self._prepare_detection_frame(frame)

        # Run detection on prepared frame
        detections = self.detector.detect(detection_frame)

        # Scale bounding boxes back to original size if needed
        if scale_factor != 1.0:
            detections = self._scale_detections(detections, scale_factor)

        elapsed = time.time() - start_time
        return detections, elapsed

    # =========================================================================
    # Frame Resizing for Detection Optimization
    # =========================================================================

    def _prepare_detection_frame(self, frame):
        """Prepare frame for detection by resizing if needed.

        Args:
            frame: Original frame

        Returns:
            Tuple of (detection_frame, scale_factor)
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
        detection_frame = cv2.resize(frame, (new_w, new_h))

        return detection_frame, scale_factor

    def _scale_detections(self, detections, scale_factor):
        """Scale detection bounding boxes from detection resolution to original resolution.

        Args:
            detections: List of Detection objects
            scale_factor: Factor used to resize frame (target_size / original_size)

        Returns:
            List of Detection objects with scaled bounding boxes
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].xyxy
            det['bbox'] = BoundingBox(
                x1 / scale_factor,
                y1 / scale_factor,
                x2 / scale_factor,
                y2 / scale_factor
            )
        return detections

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render_detections(self, frame, detections):
        """Render detection bounding boxes on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            detections: List of Detection objects to render
        """
        for det in detections:
            det['bbox'].draw(frame, color=(0, 255, 0))

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _cleanup(self):
        """Release video resources and print summary."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()

        # Clean up UI
        self.ui.cleanup()

        # Print summary
        avg_fps = self.ui.get_average_fps()
        logger.info(f"Processed {self.frame_num} frames @ {avg_fps:.1f} FPS average")
