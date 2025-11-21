"""Visualization module for video processing.

This module provides the Visualizer class for rendering detections,
overlays, and other visual elements on video frames.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..core.constants import COCO_CLASSES

# UI Constants
TEXT_COLOR = (255, 0, 0)  # BGR blue
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """Handles all visualization tasks for the video processor.
    
    Responsible for drawing:
    - Bounding boxes and labels
    - Keypoints
    - Text overlays (FPS, frame count)
    - Tracks (future)
    """

    def __init__(self, window_name: str = 'Video Tracking'):
        """Initialize visualizer.

        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name

    def render(self, frame: np.ndarray, detections: List[Dict], 
               fps: float, frame_num: int, total_frames: Optional[int],
               is_paused: bool, tracks: Optional[List] = None) -> None:
        """Render all visualizations on frame.

        Args:
            frame: Frame to draw on (will be modified in-place)
            detections: List of Detection objects to render
            fps: Current FPS value
            frame_num: Current frame number
            total_frames: Total frames in video (None for live stream)
            is_paused: True if in paused/step mode, False if playing
            tracks: List of Track objects to render (future)
        """
        # Draw detections and keypoints
        self._draw_detections(frame, detections)

        # Draw tracks (future)
        if tracks:
            self._draw_tracks(frame, tracks)

        # Draw overlay
        self._draw_overlay(frame, detections, fps, frame_num, total_frames, is_paused)

    def show(self, frame: np.ndarray) -> None:
        """Display frame in window.

        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Draw bounding boxes and keypoints."""
        for det in detections:
            # Build label with class name and confidence
            class_id = det.get('class_id', -1)
            class_name = COCO_CLASSES.get(class_id, f'Class_{class_id}')
            confidence = det.get('confidence', 0.0)
            label = f"{class_name} {confidence:.2f}"
            
            # Draw bounding box with label
            if 'bbox' in det:
                self._draw_bbox(frame, det['bbox'], color=(0, 255, 0), label=label)

            # Draw keypoints if present (for KeypointDetector)
            if 'keypoints' in det:
                det['keypoints'].draw(frame, color=(255, 0, 255))

    def _draw_bbox(self, image: np.ndarray, bbox, color=(0, 0, 255), label=None, 
                   label_bg_alpha=0.7, label_color=(255, 255, 255)) -> None:
        """Draw bounding box with optional label.
        
        Args:
            image: Image to draw on (modified in-place)
            bbox: BoundingBox object
            color: BGR color for box
            label: Optional text label to display above box
            label_bg_alpha: Background transparency for label (0-1)
            label_color: Text color for label (BGR)
        """
        x1, y1, x2, y2 = map(int, bbox.xyxy)
        
        # Draw bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Position label above box (or inside if at top edge)
            label_x = x1
            label_y = y1 - 5 if y1 > label_h + 10 else y1 + label_h + 5
            
            # Draw semi-transparent background for label
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (label_x, label_y - label_h - 3),
                (label_x + label_w + 6, label_y + 3),
                color,  # Match box color
                -1
            )
            cv2.addWeighted(overlay, label_bg_alpha, image, 1 - label_bg_alpha, 0, image)
            
            # Draw label text
            cv2.putText(image, label, (label_x + 3, label_y), 
                        font, font_scale, label_color, thickness)


    def _draw_tracks(self, frame: np.ndarray, tracks: List) -> None:
        """Draw tracking IDs and trajectories on frame (placeholder)."""
        pass

    def _draw_overlay(self, frame: np.ndarray, detections: List[Dict],
                      fps: float, frame_num: int, total_frames: Optional[int],
                      is_paused: bool) -> None:
        """Draw text overlay (FPS, frame info, mode, detection count).
        
        Shows minimal info when playing, detailed info when paused.
        """
        
        # Frame counter text
        if total_frames is None or total_frames == 0:
            frame_text = f"Frame: {frame_num}"
        else:
            # Add percentage when paused
            if is_paused:
                percentage = (frame_num / total_frames) * 100
                frame_text = f"Frame: {frame_num}/{total_frames} ({percentage:.1f}%)"
            else:
                frame_text = f"Frame: {frame_num}/{total_frames}"

        # Draw text overlay with backgrounds (stacked vertically)
        y_offset = 30
        line_height = 40
        
        self._draw_text_with_background(frame, f"FPS: {fps:.1f}", (10, y_offset))
        y_offset += line_height
        
        self._draw_text_with_background(frame, frame_text, (10, y_offset))
        y_offset += line_height
        
        # Show detailed info only when paused
        if is_paused:
            # Detection breakdown by class
            if len(detections) > 0:
                # Group detections by class
                class_counts = {}
                for det in detections:
                    class_id = det.get('class_id', -1)
                    class_name = COCO_CLASSES.get(class_id, f'Class_{class_id}')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Display total count
                self._draw_text_with_background(
                    frame, f"Detections: {len(detections)}", (10, y_offset)
                )
                y_offset += line_height
                
                # Display per-class counts
                for class_name, count in sorted(class_counts.items()):
                    self._draw_text_with_background(
                        frame, f"  {class_name}: {count}", (10, y_offset)
                    )
                    y_offset += line_height


    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                                    font_scale: float = 0.7, text_color: Tuple[int, int, int] = (255, 255, 255),
                                    bg_color: Tuple[int, int, int] = (0, 0, 0), bg_alpha: float = 0.6,
                                    thickness: int = 2, padding: int = 5) -> None:
        """Draw text with semi-transparent background for better readability."""
        font = TEXT_FONT
        
        # Calculate text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Background rectangle coordinates
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw text on top
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
