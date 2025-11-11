"""Unit tests for Keypoints class."""

import pytest
import numpy as np
from src.detecting.properties.keypoints import Keypoints, KEYPOINT_NAMES


class TestKeypoints:
    """Test Keypoints functionality."""

    @pytest.fixture
    def sample_keypoints(self):
        """Create sample keypoints for testing."""
        # Create 17 keypoints with [x, y, visibility]
        kpts = np.array([
            [100, 200, 1.0],  # nose
            [90, 190, 0.9],   # left_eye
            [110, 190, 0.9],  # right_eye
            [80, 195, 0.8],   # left_ear
            [120, 195, 0.8],  # right_ear
            [70, 250, 1.0],   # left_shoulder
            [130, 250, 1.0],  # right_shoulder
            [60, 300, 0.7],   # left_elbow
            [140, 300, 0.7],  # right_elbow
            [50, 350, 0.6],   # left_wrist
            [150, 350, 0.6],  # right_wrist
            [75, 350, 1.0],   # left_hip
            [125, 350, 1.0],  # right_hip
            [70, 450, 0.9],   # left_knee
            [130, 450, 0.9],  # right_knee
            [65, 550, 0.8],   # left_ankle
            [135, 550, 0.8],  # right_ankle
        ], dtype=np.float32)
        scores = np.array([1.0, 0.9, 0.9, 0.8, 0.8, 1.0, 1.0, 0.7, 0.7, 0.6, 0.6, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8], dtype=np.float32)
        return kpts, scores

    def test_initialization(self, sample_keypoints):
        """Test Keypoints initializes correctly."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        assert keypoints.keypoints.shape == (17, 3)
        assert keypoints.scores.shape == (17,)
        assert np.allclose(keypoints.keypoints, kpts)
        assert np.allclose(keypoints.scores, scores)

    def test_initialization_without_scores(self, sample_keypoints):
        """Test Keypoints initializes without explicit scores (uses visibility)."""
        kpts, _ = sample_keypoints
        keypoints = Keypoints(kpts)

        assert keypoints.scores.shape == (17,)
        # Scores should be the visibility values
        assert np.allclose(keypoints.scores, kpts[:, 2])

    def test_initialization_invalid_shape(self):
        """Test Keypoints raises error for invalid shape."""
        invalid_kpts = np.array([[100, 200, 1.0]])  # Only 1 keypoint

        with pytest.raises(ValueError, match="Keypoints must have shape \\(17, 3\\)"):
            Keypoints(invalid_kpts)

    def test_initialization_invalid_scores_shape(self, sample_keypoints):
        """Test Keypoints raises error for invalid scores shape."""
        kpts, _ = sample_keypoints
        invalid_scores = np.array([1.0, 0.9])  # Only 2 scores

        with pytest.raises(ValueError, match="Scores must have shape \\(17,\\)"):
            Keypoints(kpts, invalid_scores)

    def test_keypoints_property_is_copy(self, sample_keypoints):
        """Test keypoints property returns copy (immutable)."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Get keypoints
        kpts_copy = keypoints.keypoints

        # Modify the copy
        kpts_copy[0, 0] = 999.0

        # Original should be unchanged
        assert keypoints.keypoints[0, 0] != 999.0
        assert keypoints.keypoints[0, 0] == kpts[0, 0]

    def test_scores_property_is_copy(self, sample_keypoints):
        """Test scores property returns copy (immutable)."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Get scores
        scores_copy = keypoints.scores

        # Modify the copy
        scores_copy[0] = 0.0

        # Original should be unchanged
        assert keypoints.scores[0] != 0.0
        assert keypoints.scores[0] == scores[0]

    def test_num_visible(self, sample_keypoints):
        """Test num_visible counts keypoints correctly."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # All keypoints have visibility > 0.5, so all should be visible
        assert keypoints.num_visible == 17

        # Create keypoints with some invisible
        kpts_low_vis = kpts.copy()
        kpts_low_vis[0, 2] = 0.3  # nose invisible
        kpts_low_vis[1, 2] = 0.4  # left_eye invisible
        keypoints_low = Keypoints(kpts_low_vis, scores)

        assert keypoints_low.num_visible == 15

    def test_get_keypoint(self, sample_keypoints):
        """Test get_keypoint returns correct values."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Get nose keypoint
        x, y, vis, score = keypoints.get_keypoint('nose')
        assert x == 100.0
        assert y == 200.0
        assert vis == 1.0
        assert score == 1.0

        # Get left_shoulder
        x, y, vis, score = keypoints.get_keypoint('left_shoulder')
        assert x == 70.0
        assert y == 250.0
        assert vis == 1.0
        assert score == 1.0

    def test_get_keypoint_invalid_name(self, sample_keypoints):
        """Test get_keypoint raises error for invalid name."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        with pytest.raises(ValueError, match="Invalid keypoint name"):
            keypoints.get_keypoint('invalid_name')

    def test_is_visible(self, sample_keypoints):
        """Test is_visible checks visibility correctly."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # All keypoints have visibility > 0.5
        assert keypoints.is_visible(0) is True   # nose
        assert keypoints.is_visible(5) is True   # left_shoulder

        # Test with custom threshold
        kpts_low_vis = kpts.copy()
        kpts_low_vis[0, 2] = 0.3  # nose low visibility
        keypoints_low = Keypoints(kpts_low_vis, scores)

        assert keypoints_low.is_visible(0, threshold=0.5) is False
        assert keypoints_low.is_visible(0, threshold=0.2) is True

    def test_is_visible_invalid_index(self, sample_keypoints):
        """Test is_visible raises error for invalid index."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        with pytest.raises(ValueError, match="Keypoint index must be 0-16"):
            keypoints.is_visible(17)

        with pytest.raises(ValueError, match="Keypoint index must be 0-16"):
            keypoints.is_visible(-1)

    def test_scale(self, sample_keypoints):
        """Test scale returns new Keypoints with scaled coordinates."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Scale by 2.0 (double size)
        scaled_up = keypoints.scale(2.0)

        # Check scaled coordinates
        assert scaled_up.keypoints[0, 0] == 200.0  # nose x: 100 * 2
        assert scaled_up.keypoints[0, 1] == 400.0  # nose y: 200 * 2
        assert scaled_up.keypoints[0, 2] == 1.0    # visibility unchanged

        # Check scores unchanged
        assert np.allclose(scaled_up.scores, scores)

        # Original should be unchanged (immutable)
        assert keypoints.keypoints[0, 0] == 100.0
        assert keypoints.keypoints[0, 1] == 200.0

    def test_scale_down(self, sample_keypoints):
        """Test scale with factor < 1 (shrink)."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Scale by 0.5 (half size)
        scaled_down = keypoints.scale(0.5)

        # Check scaled coordinates
        assert scaled_down.keypoints[0, 0] == 50.0   # nose x: 100 * 0.5
        assert scaled_down.keypoints[0, 1] == 100.0  # nose y: 200 * 0.5
        assert scaled_down.keypoints[0, 2] == 1.0    # visibility unchanged

    def test_repr(self, sample_keypoints):
        """Test string representation."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        repr_str = repr(keypoints)
        assert "Keypoints" in repr_str
        assert "17" in repr_str  # total keypoints
        assert "visible" in repr_str.lower()

    def test_draw_does_not_crash(self, sample_keypoints):
        """Test draw method runs without errors."""
        kpts, scores = sample_keypoints
        keypoints = Keypoints(kpts, scores)

        # Create a dummy frame
        frame = np.zeros((600, 300, 3), dtype=np.uint8)

        # Should not raise any exceptions
        keypoints.draw(frame, color=(255, 0, 0), thickness=2)

        # Frame should be modified (not all zeros anymore)
        assert np.any(frame > 0)
