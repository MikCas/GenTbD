"""Video processing with detection and visualization.

This module provides the VideoProcessor class for running object detection
on video files with interactive controls and optional output saving.
"""

import cv2
import time
import logging
from typing import Optional

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

    def __init__(self, video_path: str, detector, save_output: bool = False,
                 output_path: Optional[str] = None):
        """Initialize video processor.

        Args:
            video_path: Path to input video file
            detector: Detector instance (e.g., ObjectDetector)
            save_output: Whether to save processed video
            output_path: Output video path (auto-generated if None)
        """
        self.video_path = video_path
        self.detector = detector
        self.save_output = save_output
        self.output_path = output_path

        # Video resources
        self.cap = None
        self.out = None
        self.props = {}

        # State
        self.continuous_mode = False
        self.frame_num = 0
        self.fps_list = []

    def run(self):
        """Main processing loop. Sets up video, processes frames, and cleans up."""
        try:
            self._setup_video()
            self._process_loop()
        finally:
            self._cleanup()

    def _setup_video(self):
        """Setup video capture and optional output writer."""
        import os
        from datetime import datetime

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
        """Main processing loop - read frames, detect, visualize, handle input."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_num += 1

            # Run detection
            detections, elapsed = self._process_frame(frame)
            self.fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

            # Draw overlay
            self._draw_overlay(frame, detections)

            # Display and save
            cv2.imshow('Video Tracking', frame)
            if self.out:
                self.out.write(frame)

            # Handle keyboard input
            wait_time = 1 if self.continuous_mode else 0
            key = cv2.waitKey(wait_time) & 0xFF
            if self._handle_keyboard(key, frame):
                break  # Quit requested

    def _process_frame(self, frame):
        """Run detection on frame. Returns (detections, elapsed_time)."""
        start_time = time.time()

        detections = []
        if self.detector:
            detections = self.detector.detect(frame)

            # Draw bounding boxes
            for det in detections:
                det['bbox'].draw(frame, color=(0, 255, 0))

        elapsed = time.time() - start_time
        return detections, elapsed

    def _draw_overlay(self, frame, detections):
        """Draw FPS, frame info, mode, and detection count on frame."""
        # Calculate rolling average FPS
        recent_fps = self.fps_list[-FPS_WINDOW:] if self.fps_list else [0]
        avg_fps = sum(recent_fps) / len(recent_fps)

        mode = "CONTINUOUS" if self.continuous_mode else "STEP"

        # Draw text overlay
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Frame: {self.frame_num}/{self.props['total_frames']}",
                   (10, 70), TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Mode: {mode}", (10, 110),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        if len(detections) > 0:
            cv2.putText(frame, f"Detections: {len(detections)}",
                       (10, 150), TEXT_FONT, 1, TEXT_COLOR, 2)

    def _handle_keyboard(self, key, frame) -> bool:
        """Process keyboard input. Returns True if should quit."""
        if key == ord('q'):
            return True
        elif key == ord('c'):
            self.continuous_mode = not self.continuous_mode
            mode = "CONTINUOUS" if self.continuous_mode else "STEP"
            logger.info(f"Mode: {mode}")
        elif key == ord('s'):
            filename = f"frame_{self.frame_num:05d}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Saved {filename}")
        return False

    def _cleanup(self):
        """Release video resources and print summary."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

        if self.fps_list:
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            logger.info(f"Processed {self.frame_num} frames @ {avg_fps:.1f} FPS average")
