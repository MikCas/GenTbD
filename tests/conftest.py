"""Pytest fixtures shared across all tests.

Fixtures are reusable test components that set up test data,
create mock objects, or provide common functionality.

Learn more: https://docs.pytest.org/en/stable/fixture.html
"""

import pytest
import numpy as np
import torch
import cv2
from pathlib import Path


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample RGB image (640x480)."""
    # Create a simple gradient image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Red gradient
    image[:, :, 1] = 128  # Green constant
    image[:, :, 2] = np.linspace(255, 0, 640, dtype=np.uint8)  # Blue gradient
    return image


@pytest.fixture
def sample_image_with_objects():
    """Create an image with simple rectangular objects."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw some rectangles (simulating objects)
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue box
    cv2.rectangle(image, (300, 150), (450, 300), (0, 255, 0), -1)  # Green box
    cv2.rectangle(image, (500, 350), (600, 450), (0, 0, 255), -1)  # Red box

    return image


@pytest.fixture
def sample_tensor():
    """Create a sample PyTorch tensor."""
    return torch.randn(3, 640, 480)


@pytest.fixture
def sample_bounding_boxes():
    """Create sample bounding box data."""
    return [
        (100, 100, 200, 200),  # (x1, y1, x2, y2)
        (300, 150, 450, 300),
        (500, 350, 600, 450),
    ]


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@pytest.fixture
def cpu_device():
    """Force CPU device for testing."""
    return 'cpu'


@pytest.fixture(params=['cpu', 'mps', 'cuda'])
def all_devices(request):
    """Parametrized fixture to test on all available devices."""
    device = request.param

    # Skip if device not available
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    elif device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    return device


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def pretrained_detector():
    """Load pretrained detector once per test session.

    Note: 'session' scope means this is created once and reused
    across all tests, which is faster than creating it each time.
    """
    from src.detecting.detectors import ObjectDetector
    return ObjectDetector.from_fasterrcnn_resnet50(
        device='cpu',  # Use CPU for tests (faster startup)
        conf_threshold=0.5
    )


@pytest.fixture
def mock_detector(mocker):
    """Create a mock detector for fast testing without loading real models."""
    from src.detecting.detections import Detection
    from src.detecting.properties.bounding_box import BoundingBox

    mock = mocker.MagicMock()

    # Mock detect() to return fake detections
    mock.detect.return_value = [
        Detection({
            'bbox': BoundingBox(100, 100, 200, 200),
            'class_id': 0,
            'confidence': 0.95
        })
    ]

    return mock


# ============================================================================
# File Path Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_video_path(test_data_dir):
    """Path to sample test video."""
    video_path = test_data_dir / "sample.mp4"

    # Create a simple test video if it doesn't exist
    if not video_path.exists():
        test_data_dir.mkdir(exist_ok=True)
        create_test_video(video_path)

    return str(video_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs.

    tmp_path is a pytest built-in fixture that creates
    a temporary directory unique to each test.
    """
    return tmp_path


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_video(output_path, num_frames=30, fps=10):
    """Create a simple test video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

    for i in range(num_frames):
        # Create frame with moving rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(100 + i * 10)
        cv2.rectangle(frame, (x, 200), (x + 100, 300), (0, 255, 0), -1)
        out.write(frame)

    out.release()


# ============================================================================
# Performance Fixtures
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'rounds': 10,
        'warmup': 3,
        'min_time': 0.1,
    }
