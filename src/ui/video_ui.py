"""Video UI handling - display, overlay, and keyboard input.

This module separates UI concerns from processing logic, making it easy
to swap different UI implementations (Qt, web, headless, etc.)
"""

import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# UI Constants
FPS_WINDOW = 30
TEXT_COLOR = (255, 0, 0)  # BGR blue
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


class VideoUI:
    """Handles video display, overlays, and keyboard input.

    This class provides UI functionality that can be easily swapped
    for alternative implementations (Qt, web UI, headless processing, etc.)
    """

    def __init__(self, window_name: str = 'Video Tracking'):
        """Initialize video UI.

        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.continuous_mode = False
        self.fps_list = []

    def draw_overlay(self, frame, frame_num: int, total_frames: int, detections: list):
        """Draw FPS, frame info, mode, and detection count on frame.

        Args:
            frame: Frame to draw on (modified in-place)
            frame_num: Current frame number
            total_frames: Total frames in video
            detections: List of detections to count
        """
        # Calculate rolling average FPS
        recent_fps = self.fps_list[-FPS_WINDOW:] if self.fps_list else [0]
        avg_fps = sum(recent_fps) / len(recent_fps)

        mode = "CONTINUOUS" if self.continuous_mode else "STEP"

        # Draw text overlay
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                   (10, 70), TEXT_FONT, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Mode: {mode}", (10, 110),
                   TEXT_FONT, 1, TEXT_COLOR, 2)
        if len(detections) > 0:
            cv2.putText(frame, f"Detections: {len(detections)}",
                       (10, 150), TEXT_FONT, 1, TEXT_COLOR, 2)

    def show_frame(self, frame):
        """Display frame in window.

        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)

    def wait_for_input(self) -> int:
        """Wait for keyboard input.

        Returns:
            Key code pressed (use handle_key() to interpret)
        """
        wait_time = 1 if self.continuous_mode else 0
        return cv2.waitKey(wait_time) & 0xFF

    def handle_key(self, key: int, frame) -> Optional[str]:
        """Handle keyboard input and return action.

        Args:
            key: Key code from cv2.waitKey()
            frame: Current frame (for saving if 's' pressed)

        Returns:
            'quit' if should quit, 'toggle_mode' if mode toggled,
            'save' if frame saved, None otherwise
        """
        if key == ord('q'):
            return 'quit'
        elif key == ord('c'):
            self.continuous_mode = not self.continuous_mode
            mode = "CONTINUOUS" if self.continuous_mode else "STEP"
            logger.info(f"Mode: {mode}")
            return 'toggle_mode'
        elif key == ord('s'):
            # Note: frame_num not available here, using timestamp
            import time
            filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Saved {filename}")
            return 'save'
        return None

    def add_fps_sample(self, fps: float):
        """Add FPS sample to rolling average.

        Args:
            fps: FPS value to add
        """
        self.fps_list.append(fps)

    def cleanup(self):
        """Clean up UI resources."""
        cv2.destroyAllWindows()

    def get_average_fps(self) -> float:
        """Get average FPS across all samples.

        Returns:
            Average FPS
        """
        if not self.fps_list:
            return 0.0
        return sum(self.fps_list) / len(self.fps_list)
