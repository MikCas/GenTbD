"""Enhanced video processor with interactive controls and real-time parameter adjustment.

This module extends VideoProcessor with:
- Real-time parameter adjustment via trackbars
- Enhanced keyboard controls (pause, seek, speed control)
- Multiple visualization modes
- Help overlay
- Statistics tracking
- Video seeking/scrubbing

Usage:
    # Drop-in replacement for VideoProcessor
    from src.interactive_video_processor import InteractiveVideoProcessor

    processor = InteractiveVideoProcessor(
        source='data/video.mp4',
        detector=detector,
        interactive=True  # Set False for headless mode
    )
    processor.run()

Controls:
    p       - Pause/Resume
    h       - Show/Hide help overlay
    d       - Toggle detections on/off
    v       - Cycle visualization modes
    +/-     - Adjust confidence threshold
    ←/→     - Seek backward/forward 10 frames
    ↑/↓     - Adjust playback speed
    0-9     - Jump to 0%, 10%, ..., 90% of video
    c       - Toggle continuous/step mode
    SPACE   - Next frame (step mode)
    s       - Save current frame
    r       - Reset parameters to defaults
    q       - Quit
"""

import cv2
import time
import logging
import os
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any
from .video_processor import VideoProcessor
from .core import Frame

logger = logging.getLogger(__name__)

# UI Constants
HELP_BACKGROUND_COLOR = (40, 40, 40)  # Dark gray
HELP_TEXT_COLOR = (255, 255, 255)  # White
STATS_BACKGROUND_COLOR = (30, 30, 30)  # Darker gray
STATS_TEXT_COLOR = (255, 255, 255)  # White
TRACKBAR_WINDOW = 'Controls'


class VisualizationMode:
    """Enumeration of visualization modes."""
    BOXES = 'boxes'  # Only bounding boxes
    LABELS = 'labels'  # Boxes + class labels
    CONFIDENCE = 'confidence'  # Boxes + confidence scores
    FULL = 'full'  # Boxes + labels + confidence
    MINIMAL = 'minimal'  # No overlays

    @classmethod
    def all_modes(cls):
        """Return list of all visualization modes."""
        return [cls.BOXES, cls.LABELS, cls.CONFIDENCE, cls.FULL, cls.MINIMAL]

    @classmethod
    def next_mode(cls, current_mode):
        """Get next mode in cycle."""
        modes = cls.all_modes()
        current_idx = modes.index(current_mode) if current_mode in modes else 0
        return modes[(current_idx + 1) % len(modes)]


class InteractiveVideoProcessor(VideoProcessor):
    """Enhanced video processor with interactive controls and real-time parameter adjustment.

    Extends VideoProcessor with:
    - Trackbars for real-time parameter adjustment
    - Pause/resume functionality
    - Video seeking (arrow keys)
    - Playback speed control
    - Visualization mode toggle
    - Help overlay
    - Enhanced statistics display
    """

    def __init__(self, source: str, detector, save_output: bool = False,
                 output_path: Optional[str] = None, max_dimension: Optional[int] = None,
                 skip_frames: int = 1, interactive: bool = True):
        """Initialize interactive video processor.

        Args:
            source: Path to video file, camera index (0, 1, etc.), or RTSP URL
            detector: Detector instance (e.g., ObjectDetector)
            save_output: Whether to save processed video
            output_path: Output video path (auto-generated if None)
            max_dimension: Resize frames to this max dimension before detection (None = no resize)
            skip_frames: Process every Nth frame (1 = all frames, 5 = every 5th)
            interactive: Enable interactive controls (trackbars, enhanced UI)
        """
        super().__init__(source, detector, save_output, output_path, max_dimension, skip_frames)

        # Interactive mode flag
        self.interactive = interactive

        # Enhanced state management
        self.paused = False
        self.show_help = False
        self.show_detections = True
        self.visualization_mode = VisualizationMode.FULL

        # Playback control
        self.playback_speed = 1.0  # 1.0 = normal speed
        self.seek_frames = 10  # Number of frames to seek with arrow keys

        # Parameter defaults (for reset functionality)
        self.default_conf_threshold = detector.conf_threshold if hasattr(detector, 'conf_threshold') else 0.5
        self.current_conf_threshold = self.default_conf_threshold

        # Statistics tracking
        self.total_detections = 0
        self.class_counts = {}  # Track detections per class
        self.frame_times = deque(maxlen=100)  # Track processing times

        # Cached frame for pause mode
        self.cached_display_frame = None

    def run(self):
        """Main processing loop with interactive controls."""
        try:
            self._setup_video()
            if self.interactive:
                self._setup_interactive_ui()
            self._process_loop()
        finally:
            self._cleanup()

    def _setup_interactive_ui(self):
        """Setup interactive UI elements (trackbars and control window)."""
        # Create control window for trackbars
        cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TRACKBAR_WINDOW, 400, 300)

        # Confidence threshold trackbar (0-100, representing 0.0-1.0)
        initial_conf = int(self.current_conf_threshold * 100)
        cv2.createTrackbar('Confidence', TRACKBAR_WINDOW, initial_conf, 100,
                          self._on_confidence_change)

        # Playback speed trackbar (1-50, representing 0.1x-5.0x)
        initial_speed = int(self.playback_speed * 10)
        cv2.createTrackbar('Speed (0.1x-5.0x)', TRACKBAR_WINDOW, initial_speed, 50,
                          self._on_speed_change)

        # Skip frames trackbar (1-30)
        cv2.createTrackbar('Skip Frames', TRACKBAR_WINDOW, self.skip_frames, 30,
                          self._on_skip_frames_change)

        # Visualization mode trackbar (0-4)
        mode_idx = VisualizationMode.all_modes().index(self.visualization_mode)
        cv2.createTrackbar('Viz Mode', TRACKBAR_WINDOW, mode_idx,
                          len(VisualizationMode.all_modes()) - 1,
                          self._on_viz_mode_change)

        logger.info("Interactive controls enabled. Press 'h' for help.")

    # Trackbar callbacks
    def _on_confidence_change(self, value):
        """Callback for confidence threshold trackbar."""
        self.current_conf_threshold = value / 100.0
        if hasattr(self.detector, 'conf_threshold'):
            self.detector.conf_threshold = self.current_conf_threshold

    def _on_speed_change(self, value):
        """Callback for playback speed trackbar."""
        self.playback_speed = max(0.1, value / 10.0)

    def _on_skip_frames_change(self, value):
        """Callback for skip frames trackbar."""
        self.skip_frames = max(1, value)

    def _on_viz_mode_change(self, value):
        """Callback for visualization mode trackbar."""
        modes = VisualizationMode.all_modes()
        if 0 <= value < len(modes):
            self.visualization_mode = modes[value]

    def _process_loop(self):
        """Enhanced processing loop with interactive controls."""
        frame_count = 0

        while True:
            # Handle pause mode
            if self.paused:
                if self.cached_display_frame is not None:
                    self._show_frame(self.cached_display_frame)
                if self._handle_input(self.cached_display_frame):
                    break
                continue

            # Read frame
            ret, frame_data = self.cap.read()
            if not ret:
                break

            # Create Frame object
            if self.video_fps and self.video_fps > 0:
                timestamp = (frame_count + 1) / self.video_fps
            else:
                timestamp = (frame_count + 1) * (1.0 / 30.0)

            frame = Frame(
                data=frame_data,
                frame_id=frame_count,
                timestamp=timestamp,
                source_id=str(self.source)
            )

            self.frame_num = frame_count + 1
            should_detect = frame_count % self.skip_frames == 0

            # Run detection and render
            if should_detect:
                start_time = time.time()
                detections, elapsed = self._detect_objects(frame)
                self.frame_times.append(elapsed)

                # Update statistics
                self._update_statistics(detections)

                # Make a copy for display
                display_frame = frame.data.copy()

                # Render based on visualization mode
                if self.show_detections and self.visualization_mode != VisualizationMode.MINIMAL:
                    self._render_frame_enhanced(display_frame, detections)

                # Always render overlay (FPS, controls, etc.)
                self._draw_overlay_enhanced(display_frame, detections)

                # Show help if requested
                if self.show_help:
                    self._draw_help_overlay(display_frame)

                # Cache for pause mode
                self.cached_display_frame = display_frame
            else:
                # Use previous frame (frame skipping mode)
                display_frame = self.cached_display_frame

            # Display and save
            self._show_frame(display_frame)
            if self.out:
                self.out.write(display_frame)

            # Handle keyboard input
            wait_time = max(1, int((1000 / self.video_fps) / self.playback_speed)) if self.video_fps > 0 else 1
            if self._handle_input_enhanced(display_frame, wait_time):
                break

            frame_count += 1

    def _render_frame_enhanced(self, frame, detections):
        """Render detections with enhanced visualization options.

        Args:
            frame: Frame to draw on (modified in-place)
            detections: List of Detection objects
        """
        for det in detections:
            # Draw bounding box
            bbox = det['bbox']
            class_id = det.get('class_id', 0)
            confidence = det.get('confidence', 0.0)

            # Choose color based on class (for variety)
            color = self._get_class_color(class_id)

            # Draw based on visualization mode
            if self.visualization_mode in [VisualizationMode.BOXES, VisualizationMode.LABELS,
                                          VisualizationMode.CONFIDENCE, VisualizationMode.FULL]:
                bbox.draw(frame, color=color)

            # Draw label
            if self.visualization_mode in [VisualizationMode.LABELS, VisualizationMode.FULL]:
                x1, y1, x2, y2 = map(int, bbox.xyxy)
                label = f"Class {class_id}"
                self._draw_label(frame, label, x1, y1 - 10, color)

            # Draw confidence
            if self.visualization_mode in [VisualizationMode.CONFIDENCE, VisualizationMode.FULL]:
                x1, y1, x2, y2 = map(int, bbox.xyxy)
                conf_text = f"{confidence:.2f}"
                offset = 30 if self.visualization_mode == VisualizationMode.FULL else 10
                self._draw_label(frame, conf_text, x1, y1 - offset, color)

            # Draw keypoints if present
            if 'keypoints' in det:
                det['keypoints'].draw(frame, color=(255, 0, 255))

    def _draw_overlay_enhanced(self, frame, detections):
        """Draw enhanced overlay with statistics and controls.

        Args:
            frame: Frame to draw on (modified in-place)
            detections: List of detections for current frame
        """
        h, w = frame.shape[:2]

        # Calculate average FPS
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0

        # Prepare text lines
        lines = []

        # Main stats
        lines.append(f"FPS: {avg_fps:.1f}")

        if self.is_live_stream:
            lines.append(f"Frame: {self.frame_num}")
        else:
            progress_pct = (self.frame_num / self.total_frames * 100) if self.total_frames > 0 else 0
            lines.append(f"Frame: {self.frame_num}/{self.total_frames} ({progress_pct:.1f}%)")

        # Mode and state
        mode = "CONTINUOUS" if self.continuous_mode else "STEP"
        state = "[PAUSED]" if self.paused else ""
        lines.append(f"Mode: {mode} {state}")

        # Detections
        if len(detections) > 0:
            lines.append(f"Detections: {len(detections)}")

        # Interactive parameters
        if self.interactive:
            lines.append(f"Conf: {self.current_conf_threshold:.2f}")
            lines.append(f"Speed: {self.playback_speed:.1f}x")
            lines.append(f"Skip: {self.skip_frames}")
            lines.append(f"Viz: {self.visualization_mode}")

        # Draw text with background for better readability
        self._draw_text_block(frame, lines, x=10, y=30, bg_color=(0, 0, 0, 180))

        # Draw statistics panel on the right
        if self.interactive and not self.paused:
            self._draw_statistics_panel(frame, w)

    def _draw_statistics_panel(self, frame, frame_width):
        """Draw statistics panel on the right side of frame.

        Args:
            frame: Frame to draw on (modified in-place)
            frame_width: Width of the frame
        """
        # Panel dimensions
        panel_width = 250
        panel_x = frame_width - panel_width - 10
        panel_y = 10

        # Prepare statistics
        stats = []
        stats.append("=== SESSION STATS ===")
        stats.append(f"Total Dets: {self.total_detections}")

        # Average detections per frame
        avg_dets = self.total_detections / max(1, self.frame_num)
        stats.append(f"Avg/Frame: {avg_dets:.1f}")

        # Average processing time
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            stats.append(f"Avg Time: {avg_time*1000:.1f}ms")

        # Top classes
        if self.class_counts:
            stats.append("")
            stats.append("Top Classes:")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_id, count in sorted_classes[:5]:
                pct = (count / self.total_detections * 100) if self.total_detections > 0 else 0
                stats.append(f"  {class_id}: {count} ({pct:.1f}%)")

        # Draw panel
        self._draw_text_block(frame, stats, x=panel_x, y=panel_y,
                             bg_color=(30, 30, 30, 200),
                             text_color=STATS_TEXT_COLOR)

    def _draw_help_overlay(self, frame):
        """Draw help overlay with all keyboard shortcuts.

        Args:
            frame: Frame to draw on (modified in-place)
        """
        h, w = frame.shape[:2]

        # Help text
        help_lines = [
            "=== KEYBOARD SHORTCUTS ===",
            "",
            "p       - Pause/Resume",
            "h       - Toggle this help",
            "d       - Toggle detections",
            "v       - Cycle visualization modes",
            "+/-     - Adjust confidence",
            "←/→     - Seek -/+ 10 frames",
            "↑/↓     - Adjust speed",
            "0-9     - Jump to 0%-90%",
            "c       - Toggle step/continuous",
            "SPACE   - Next frame (step mode)",
            "s       - Save current frame",
            "r       - Reset parameters",
            "q       - Quit",
            "",
            "Press 'h' to hide this help"
        ]

        # Calculate centered position
        center_x = w // 2 - 200
        center_y = h // 2 - (len(help_lines) * 15)

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (center_x - 20, center_y - 20),
                     (center_x + 400, center_y + len(help_lines) * 30),
                     HELP_BACKGROUND_COLOR, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw text
        for i, line in enumerate(help_lines):
            y = center_y + i * 30
            cv2.putText(frame, line, (center_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, HELP_TEXT_COLOR, 1, cv2.LINE_AA)

    def _handle_input_enhanced(self, frame, wait_time=1) -> bool:
        """Handle enhanced keyboard input with additional controls.

        Args:
            frame: Current frame (for saving if 's' pressed)
            wait_time: Milliseconds to wait for key press

        Returns:
            True if should quit, False otherwise
        """
        key = cv2.waitKey(wait_time) & 0xFF

        # Quit
        if key == ord('q'):
            return True

        # Pause/Resume
        elif key == ord('p'):
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RESUMED"
            logger.info(f"Playback {state}")

        # Toggle help
        elif key == ord('h'):
            self.show_help = not self.show_help

        # Toggle detections
        elif key == ord('d'):
            self.show_detections = not self.show_detections
            state = "ON" if self.show_detections else "OFF"
            logger.info(f"Detections {state}")

        # Cycle visualization mode
        elif key == ord('v'):
            self.visualization_mode = VisualizationMode.next_mode(self.visualization_mode)
            logger.info(f"Visualization mode: {self.visualization_mode}")

        # Toggle continuous mode
        elif key == ord('c'):
            self.continuous_mode = not self.continuous_mode
            mode = "CONTINUOUS" if self.continuous_mode else "STEP"
            logger.info(f"Mode: {mode}")

        # Save frame
        elif key == ord('s'):
            filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Saved {filename}")

        # Reset parameters
        elif key == ord('r'):
            self._reset_parameters()
            logger.info("Parameters reset to defaults")

        # Adjust confidence threshold
        elif key == ord('+') or key == ord('='):
            self.current_conf_threshold = min(1.0, self.current_conf_threshold + 0.05)
            if hasattr(self.detector, 'conf_threshold'):
                self.detector.conf_threshold = self.current_conf_threshold
            if self.interactive:
                cv2.setTrackbarPos('Confidence', TRACKBAR_WINDOW,
                                  int(self.current_conf_threshold * 100))
            logger.info(f"Confidence: {self.current_conf_threshold:.2f}")

        elif key == ord('-') or key == ord('_'):
            self.current_conf_threshold = max(0.0, self.current_conf_threshold - 0.05)
            if hasattr(self.detector, 'conf_threshold'):
                self.detector.conf_threshold = self.current_conf_threshold
            if self.interactive:
                cv2.setTrackbarPos('Confidence', TRACKBAR_WINDOW,
                                  int(self.current_conf_threshold * 100))
            logger.info(f"Confidence: {self.current_conf_threshold:.2f}")

        # Seek forward/backward (arrow keys)
        elif key == 83:  # Right arrow
            if not self.is_live_stream:
                new_pos = min(self.total_frames - 1, self.frame_num + self.seek_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                logger.info(f"Seek forward to frame {new_pos}")

        elif key == 81:  # Left arrow
            if not self.is_live_stream:
                new_pos = max(0, self.frame_num - self.seek_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                logger.info(f"Seek backward to frame {new_pos}")

        # Adjust playback speed (up/down arrows)
        elif key == 82:  # Up arrow
            self.playback_speed = min(5.0, self.playback_speed + 0.1)
            if self.interactive:
                cv2.setTrackbarPos('Speed (0.1x-5.0x)', TRACKBAR_WINDOW,
                                  int(self.playback_speed * 10))
            logger.info(f"Speed: {self.playback_speed:.1f}x")

        elif key == 84:  # Down arrow
            self.playback_speed = max(0.1, self.playback_speed - 0.1)
            if self.interactive:
                cv2.setTrackbarPos('Speed (0.1x-5.0x)', TRACKBAR_WINDOW,
                                  int(self.playback_speed * 10))
            logger.info(f"Speed: {self.playback_speed:.1f}x")

        # Jump to percentage (0-9 keys)
        elif key in [ord(str(i)) for i in range(10)]:
            if not self.is_live_stream:
                percentage = (key - ord('0')) * 10  # 0%, 10%, 20%, ..., 90%
                new_pos = int(self.total_frames * percentage / 100)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                logger.info(f"Jump to {percentage}% (frame {new_pos})")

        return False

    def _reset_parameters(self):
        """Reset all adjustable parameters to defaults."""
        self.current_conf_threshold = self.default_conf_threshold
        if hasattr(self.detector, 'conf_threshold'):
            self.detector.conf_threshold = self.default_conf_threshold

        self.playback_speed = 1.0
        self.skip_frames = 1
        self.visualization_mode = VisualizationMode.FULL

        # Update trackbars if in interactive mode
        if self.interactive:
            cv2.setTrackbarPos('Confidence', TRACKBAR_WINDOW,
                              int(self.default_conf_threshold * 100))
            cv2.setTrackbarPos('Speed (0.1x-5.0x)', TRACKBAR_WINDOW, 10)
            cv2.setTrackbarPos('Skip Frames', TRACKBAR_WINDOW, 1)
            cv2.setTrackbarPos('Viz Mode', TRACKBAR_WINDOW,
                              VisualizationMode.all_modes().index(VisualizationMode.FULL))

    def _update_statistics(self, detections):
        """Update running statistics with new detections.

        Args:
            detections: List of Detection objects from current frame
        """
        self.total_detections += len(detections)

        # Track class counts
        for det in detections:
            class_id = det.get('class_id', 0)
            self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1

    def _get_class_color(self, class_id):
        """Get consistent color for a class ID.

        Args:
            class_id: Class ID

        Returns:
            BGR color tuple
        """
        # Use class_id as seed for consistent colors
        np.random.seed(class_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color

    def _draw_label(self, frame, text, x, y, color):
        """Draw text label with background.

        Args:
            frame: Frame to draw on
            text: Text to draw
            x, y: Position
            color: Text color
        """
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(frame, (x, y - text_height - 4), (x + text_width, y), (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, text, (x, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_text_block(self, frame, lines, x, y, bg_color=(0, 0, 0, 180),
                        text_color=(255, 255, 255)):
        """Draw block of text with semi-transparent background.

        Args:
            frame: Frame to draw on
            lines: List of text lines
            x, y: Starting position
            bg_color: Background color (RGB or RGBA)
            text_color: Text color (BGR)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 30

        # Calculate background dimensions
        max_width = 0
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)

        bg_height = len(lines) * line_height + 10
        bg_width = max_width + 20

        # Draw semi-transparent background
        if len(bg_color) == 4:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 5, y - 25), (x + bg_width, y + bg_height - 25),
                         bg_color[:3], -1)
            alpha = bg_color[3] / 255.0
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            cv2.rectangle(frame, (x - 5, y - 25), (x + bg_width, y + bg_height - 25),
                         bg_color, -1)

        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            cv2.putText(frame, line, (x, y_pos), font, font_scale, text_color,
                       thickness, cv2.LINE_AA)

    def _cleanup(self):
        """Enhanced cleanup to close control window."""
        super()._cleanup()

        # Close control window if it exists
        if self.interactive:
            try:
                cv2.destroyWindow(TRACKBAR_WINDOW)
            except:
                pass
