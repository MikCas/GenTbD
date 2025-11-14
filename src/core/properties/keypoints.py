"""Human pose keypoints representation and visualization.

This module provides the Keypoints class for storing and manipulating
COCO-format human pose keypoints (17 body landmarks).
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# COCO keypoint names and indices
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# Skeleton connections for drawing (pairs of keypoint indices)
SKELETON = [
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
    (0, 5), (0, 6),      # nose to shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # shoulders to hips
    (11, 12),            # hips to each other
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


class Keypoints:
    """Represents human pose keypoints (17 COCO keypoints).

    Stores keypoint coordinates, visibility flags, and confidence scores.
    Provides methods for scaling, visualization, and querying keypoints.
    """

    def __init__(self, keypoints: np.ndarray, scores: Optional[np.ndarray] = None):
        """Initialize keypoints from model output.

        Args:
            keypoints: Array of shape (17, 3) with [x, y, visibility] for each keypoint
            scores: Optional array of shape (17,) with confidence scores per keypoint

        Raises:
            ValueError: If keypoints shape is invalid
        """
        keypoints = np.array(keypoints, dtype=np.float32)

        if keypoints.shape != (17, 3):
            raise ValueError(f"Keypoints must have shape (17, 3), got {keypoints.shape}")

        self._keypoints = keypoints.copy()  # [x, y, visibility]

        if scores is not None:
            scores = np.array(scores, dtype=np.float32)
            if scores.shape != (17,):
                raise ValueError(f"Scores must have shape (17,), got {scores.shape}")
            self._scores = scores.copy()
        else:
            # Use visibility as scores if not provided
            self._scores = keypoints[:, 2].copy()

    @property
    def keypoints(self) -> np.ndarray:
        """Get keypoint coordinates and visibility.

        Returns:
            Array of shape (17, 3) with [x, y, visibility]
        """
        return self._keypoints.copy()

    @property
    def scores(self) -> np.ndarray:
        """Get keypoint confidence scores.

        Returns:
            Array of shape (17,) with confidence scores
        """
        return self._scores.copy()

    @property
    def num_visible(self) -> int:
        """Get number of visible/detected keypoints.

        Returns:
            Count of keypoints with visibility > 0.5
        """
        return int(np.sum(self._keypoints[:, 2] > 0.5))

    def get_keypoint(self, name: str) -> Tuple[float, float, float, float]:
        """Get specific keypoint by name.

        Args:
            name: Keypoint name (e.g., 'nose', 'left_shoulder')

        Returns:
            Tuple of (x, y, visibility, confidence)

        Raises:
            ValueError: If keypoint name is invalid
        """
        if name not in KEYPOINT_NAMES:
            valid_names = ', '.join(KEYPOINT_NAMES)
            raise ValueError(f"Invalid keypoint name '{name}'. Valid names: {valid_names}")

        idx = KEYPOINT_NAMES.index(name)
        x, y, vis = self._keypoints[idx]
        score = self._scores[idx]
        return (float(x), float(y), float(vis), float(score))

    def is_visible(self, index: int, threshold: float = 0.5) -> bool:
        """Check if keypoint is visible/detected.

        Args:
            index: Keypoint index (0-16)
            threshold: Visibility threshold (default: 0.5)

        Returns:
            True if keypoint visibility > threshold
        """
        if not 0 <= index < 17:
            raise ValueError(f"Keypoint index must be 0-16, got {index}")

        return float(self._keypoints[index, 2]) > threshold

    def scale(self, factor: float) -> 'Keypoints':
        """Return new Keypoints with scaled coordinates.

        Args:
            factor: Scale factor (e.g., 2.0 = double size, 0.5 = half size)

        Returns:
            New Keypoints instance with scaled coordinates
        """
        scaled_kpts = self._keypoints.copy()
        scaled_kpts[:, 0] *= factor  # Scale x coordinates
        scaled_kpts[:, 1] *= factor  # Scale y coordinates
        # Visibility unchanged

        return Keypoints(scaled_kpts, self._scores)

    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2, keypoint_radius: int = 4,
             draw_skeleton: bool = True, draw_keypoints: bool = True,
             visibility_threshold: float = 0.5):
        """Draw keypoints and skeleton on image (in-place).

        Args:
            image: Image to draw on (will be modified)
            color: RGB color for drawing
            thickness: Line thickness for skeleton
            keypoint_radius: Radius for keypoint circles
            draw_skeleton: Whether to draw skeleton connections
            draw_keypoints: Whether to draw keypoint circles
            visibility_threshold: Minimum visibility to draw keypoint
        """
        kpts = self._keypoints

        # Draw skeleton connections
        if draw_skeleton:
            for start_idx, end_idx in SKELETON:
                # Check if both keypoints are visible
                if (self.is_visible(start_idx, visibility_threshold) and
                    self.is_visible(end_idx, visibility_threshold)):

                    start_pt = (int(kpts[start_idx, 0]), int(kpts[start_idx, 1]))
                    end_pt = (int(kpts[end_idx, 0]), int(kpts[end_idx, 1]))

                    cv2.line(image, start_pt, end_pt, color, thickness)

        # Draw keypoint circles
        if draw_keypoints:
            for i in range(17):
                if self.is_visible(i, visibility_threshold):
                    center = (int(kpts[i, 0]), int(kpts[i, 1]))
                    cv2.circle(image, center, keypoint_radius, color, -1)  # Filled circle
                    # Draw white border for better visibility
                    cv2.circle(image, center, keypoint_radius, (255, 255, 255), 1)

    def __repr__(self) -> str:
        """String representation of keypoints."""
        return f"Keypoints(visible={self.num_visible}/17)"
