# Testing Guide for GenTbD

This guide explains how to write and run tests for the GenTbD project. Testing is crucial for maintaining code quality and catching bugs early.

## Table of Contents

- [Quick Start](#quick-start)
- [Testing Philosophy](#testing-philosophy)
- [Types of Tests](#types-of-tests)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Best Practices](#best-practices)
- [CI/CD Integration](#cicd-integration)

---

## Quick Start

### Install Testing Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_bounding_box.py

# Run specific test
pytest tests/unit/test_bounding_box.py::TestBoundingBoxIoU::test_iou_identical_boxes
```

### Common Commands

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests in parallel (faster!)
pytest -n auto

# Generate HTML coverage report
pytest --cov-report=html
# Then open: htmlcov/index.html

# Run tests with verbose output
pytest -vv

# Run tests and stop at first failure
pytest -x

# Run tests that failed last time
pytest --lf
```

---

## Testing Philosophy

We follow the **Testing Pyramid** approach:

```
        /\
       /E2E\        ← Few end-to-end tests
      /------\
     /Integr.\     ← Some integration tests
    /----------\
   /   Unit     \  ← Many unit tests
  /--------------\
```

**Key Principles:**

1. **Fast Feedback** - Unit tests should run in milliseconds
2. **Isolation** - Tests shouldn't depend on each other
3. **Clarity** - Test names should describe what they test
4. **Coverage** - Aim for 80%+ code coverage
5. **Maintainability** - Tests are code too - keep them clean!

---

## Types of Tests

### 1. Unit Tests

**Purpose:** Test individual components in isolation

**Location:** `tests/unit/`

**Characteristics:**
- Very fast (< 10ms each)
- No external dependencies
- Use mocks for dependencies
- Test one thing at a time

**Example:**
```python
def test_bounding_box_iou():
    """Test IoU calculation for identical boxes."""
    bbox1 = BoundingBox(10, 10, 100, 100)
    bbox2 = BoundingBox(10, 10, 100, 100)

    assert bbox1.iou(bbox2) == 1.0  # Perfect overlap
```

**When to write:**
- Testing utility functions
- Testing data structures (BoundingBox, Detection)
- Testing algorithms (IoU, NMS)

### 2. Integration Tests

**Purpose:** Test how components work together

**Location:** `tests/integration/`

**Characteristics:**
- Slower (100ms - 1s)
- Load real models
- Test component interactions
- May use test data files

**Example:**
```python
@pytest.mark.integration
def test_detector_pipeline(pretrained_detector, sample_image):
    """Test full detection pipeline."""
    detections = pretrained_detector.detect(sample_image)

    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)
```

**When to write:**
- Testing detector with real models
- Testing video processor with detector
- Testing tracking with multiple components

### 3. End-to-End Tests

**Purpose:** Test complete workflows

**Location:** `tests/e2e/`

**Characteristics:**
- Slowest (seconds to minutes)
- Test complete user workflows
- Use real data files
- Test CLI interfaces

**Example:**
```python
@pytest.mark.slow
def test_process_video_end_to_end(sample_video_path):
    """Test processing a complete video."""
    processor = VideoProcessor(sample_video_path, detector)
    processor.run()

    # Verify output exists and is valid
    assert output_exists
```

**When to write:**
- Testing complete video processing
- Testing CLI commands
- Testing batch processing

---

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test categories
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Skip slow tests
pytest -m "unit or integration"  # Multiple markers
```

### Test Selection

```bash
# By file
pytest tests/unit/test_bounding_box.py

# By class
pytest tests/unit/test_bounding_box.py::TestBoundingBoxIoU

# By function
pytest tests/unit/test_bounding_box.py::TestBoundingBoxIoU::test_iou_identical_boxes

# By keyword (name matching)
pytest -k "iou"             # Run all tests with 'iou' in name
pytest -k "not gpu"         # Skip GPU tests
```

### Output Control

```bash
# Verbose output
pytest -v                   # Show test names
pytest -vv                  # Extra verbose

# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Stop after first failure
pytest -x

# Show slowest 10 tests
pytest --durations=10
```

### Parallel Execution

```bash
# Run tests in parallel (much faster!)
pytest -n auto              # Auto-detect CPU count
pytest -n 4                 # Use 4 workers
```

### Coverage Reports

```bash
# Terminal report
pytest --cov=src --cov-report=term

# HTML report (recommended!)
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

---

## Writing Tests

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
def test_bounding_box_iou():
    # ARRANGE - Set up test data
    bbox1 = BoundingBox(0, 0, 100, 100)
    bbox2 = BoundingBox(50, 50, 150, 150)

    # ACT - Perform the action
    iou = bbox1.iou(bbox2)

    # ASSERT - Verify the result
    assert iou == pytest.approx(0.143, rel=0.01)
```

### Test Naming

Use descriptive names that explain **what** is being tested:

```python
# Good names ✅
def test_iou_identical_boxes_returns_one():
def test_detector_filters_low_confidence_detections():
def test_video_processor_handles_missing_file():

# Bad names ❌
def test_iou():
def test_detector():
def test_case_1():
```

### Using Fixtures

Fixtures are reusable test components defined in `conftest.py`:

```python
def test_detection_on_sample_image(sample_image, pretrained_detector):
    """Fixtures are automatically passed as arguments."""
    detections = pretrained_detector.detect(sample_image)
    assert len(detections) >= 0
```

### Parametrized Tests

Test multiple cases with one test function:

```python
@pytest.mark.parametrize("x1,y1,x2,y2,expected_area", [
    (0, 0, 10, 10, 100),
    (0, 0, 20, 20, 400),
    (5, 5, 15, 15, 100),
])
def test_bounding_box_area(x1, y1, x2, y2, expected_area):
    """Test area calculation for various boxes."""
    bbox = BoundingBox(x1, y1, x2, y2)
    assert bbox.area() == expected_area
```

### Using Markers

Mark tests for selective execution:

```python
@pytest.mark.unit
def test_fast_unit_test():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_with_real_model():
    """Integration test with model loading."""
    pass

@pytest.mark.slow
def test_long_running():
    """Test that takes a while."""
    pass

@pytest.mark.gpu
def test_requires_mps():
    """Test that requires GPU/MPS."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
```

### Testing Exceptions

```python
def test_invalid_video_path_raises_error():
    """Test that invalid path raises ValueError."""
    with pytest.raises(ValueError, match="Video file not found"):
        processor = VideoProcessor("nonexistent.mp4", detector)
```

### Using Mocks

Mock expensive operations for faster tests:

```python
def test_video_processor_calls_detector(mocker, sample_image):
    """Test that processor calls detector.detect()."""
    mock_detector = mocker.MagicMock()
    mock_detector.detect.return_value = []

    processor = VideoProcessor("video.mp4", mock_detector)
    # ... test logic ...

    # Verify detector was called
    mock_detector.detect.assert_called()
```

---

## Best Practices

### 1. Keep Tests Independent

```python
# Bad ❌ - Tests depend on each other
class TestDetector:
    def test_create_detector(self):
        self.detector = ObjectDetector(...)

    def test_detect(self):
        # Depends on test_create_detector!
        self.detector.detect(image)

# Good ✅ - Each test is independent
class TestDetector:
    @pytest.fixture
    def detector(self):
        return ObjectDetector(...)

    def test_detect(self, detector):
        detector.detect(image)
```

### 2. Test One Thing at a Time

```python
# Bad ❌ - Testing multiple things
def test_detector():
    detector = ObjectDetector(...)
    detections = detector.detect(image)
    assert len(detections) > 0
    assert detections[0]['confidence'] > 0.5
    assert detections[0]['class_id'] == 0

# Good ✅ - Separate tests
def test_detector_returns_detections(detector):
    detections = detector.detect(image)
    assert len(detections) > 0

def test_detector_filters_by_confidence(detector):
    detections = detector.detect(image)
    assert all(d['confidence'] > 0.5 for d in detections)

def test_detector_filters_by_class(detector):
    detections = detector.detect(image)
    assert all(d['class_id'] == 0 for d in detections)
```

### 3. Use Descriptive Assertions

```python
# Bad ❌ - Unclear failure message
assert result == expected

# Good ✅ - Clear failure message
assert result == expected, f"Expected {expected}, got {result}"

# Better ✅ - Use pytest helpers
assert result == pytest.approx(expected, rel=0.01)
```

### 4. Don't Test Implementation Details

```python
# Bad ❌ - Testing internal implementation
def test_detector_uses_fasterrcnn():
    assert isinstance(detector.model, FasterRCNN)

# Good ✅ - Testing behavior
def test_detector_returns_valid_detections():
    detections = detector.detect(image)
    assert all(isinstance(d, Detection) for d in detections)
```

### 5. Test Edge Cases

```python
def test_detection_on_empty_image():
    """Test with blank image."""

def test_detection_on_very_large_image():
    """Test with 4K image."""

def test_detection_with_zero_confidence_threshold():
    """Test edge case threshold."""
```

---

## CI/CD Integration

### GitHub Actions

Tests automatically run on every push/PR. See `.github/workflows/tests.yml`.

```yaml
# Runs on: push to main, pull requests, manual trigger
# Tests on: Python 3.8, 3.9, 3.10, 3.11
# Reports: Coverage to Codecov
```

### Pre-commit Hooks

Run tests before committing:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Tests will run automatically on git commit
```

### Local CI Simulation

Test locally before pushing:

```bash
# Run the same tests as CI
pytest --cov=src --cov-report=xml -m "not slow"

# Run code quality checks
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

---

## Troubleshooting

### Common Issues

**Tests fail with "No module named 'src'"**
```bash
# Solution: Install package in development mode
pip install -e .
```

**MPS/GPU tests fail**
```bash
# Solution: Skip GPU tests on non-GPU machines
pytest -m "not gpu"
```

**Tests are too slow**
```bash
# Solution 1: Run only fast tests
pytest -m "not slow"

# Solution 2: Run in parallel
pytest -n auto

# Solution 3: Use last-failed mode
pytest --lf  # Only run tests that failed last time
```

**Coverage too low**
```bash
# Find uncovered lines
pytest --cov=src --cov-report=term-missing

# Generate HTML report for detailed view
pytest --cov=src --cov-report=html
```

---

## Learning Resources

### Pytest Documentation
- Official docs: https://docs.pytest.org/
- Fixtures: https://docs.pytest.org/en/stable/fixture.html
- Parametrize: https://docs.pytest.org/en/stable/parametrize.html

### Testing Best Practices
- "Test Driven Development" by Kent Beck
- "Python Testing with pytest" by Brian Okken
- Real Python Testing Guide: https://realpython.com/pytest-python-testing/

### Related Topics
- Mocking: https://docs.python.org/3/library/unittest.mock.html
- Coverage: https://coverage.readthedocs.io/
- TDD: https://testdriven.io/test-driven-development/

---

## Next Steps

1. **Run the existing tests** to make sure everything works
2. **Write tests for new features** before implementing them (TDD)
3. **Add tests when fixing bugs** to prevent regression
4. **Aim for 80%+ coverage** on new code
5. **Review test results** in CI/CD pipelines

Happy testing! 🧪
