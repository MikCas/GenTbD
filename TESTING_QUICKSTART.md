# Testing Quick Start Guide

Get started with testing in GenTbD in 5 minutes! 🚀

## Setup (One-time)

```bash
# 1. Install development dependencies
pip install -r requirements-dev.txt

# 2. Install package in development mode
pip install -e .

# 3. Verify installation
pytest --version
```

## Running Tests

```bash
# Run all tests (use this most often)
pytest

# Run fast tests only
make test-fast

# Run with coverage report
make test

# Run specific test file
pytest tests/unit/test_bounding_box.py
```

## Understanding Test Output

```
tests/unit/test_bounding_box.py::TestBoundingBoxIoU::test_iou_identical_boxes PASSED [ 10%]
                                    ^                     ^                        ^      ^
                                    |                     |                        |      |
                                File path          Test class            Test function  Status  Progress
```

**Status meanings:**
- ✅ `PASSED` - Test passed
- ❌ `FAILED` - Test failed (check output for details)
- ⏭️ `SKIPPED` - Test skipped (e.g., GPU not available)

## Writing Your First Test

1. **Create test file** in `tests/unit/` or `tests/integration/`
2. **Name it** `test_<component>.py`
3. **Write test function** starting with `test_`

Example:

```python
# tests/unit/test_my_feature.py

def test_addition():
    """Test that 1 + 1 equals 2."""
    assert 1 + 1 == 2

def test_string_concatenation():
    """Test string joining."""
    result = "Hello" + " " + "World"
    assert result == "Hello World"
```

Run it:
```bash
pytest tests/unit/test_my_feature.py -v
```

## Common Patterns

### Testing a Function

```python
from src.my_module import my_function

def test_my_function_returns_expected_value():
    """Test my_function with valid input."""
    result = my_function(input_data)
    assert result == expected_output
```

### Testing a Class

```python
from src.detecting.properties.bounding_box import BoundingBox

class TestBoundingBox:
    def test_creation(self):
        """Test BoundingBox creation."""
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox is not None

    def test_coordinates(self):
        """Test coordinate access."""
        bbox = BoundingBox(10, 20, 100, 200)
        x1, y1, x2, y2 = bbox.xyxy
        assert x1 == 10
```

### Using Fixtures

```python
def test_detection_on_sample_image(sample_image, pretrained_detector):
    """Fixtures provide reusable test data."""
    detections = pretrained_detector.detect(sample_image)
    assert isinstance(detections, list)
```

Available fixtures (see `tests/conftest.py`):
- `sample_image` - A 640x480 test image
- `pretrained_detector` - Loaded FasterRCNN model
- `device` - Best available device (cuda/mps/cpu)
- `temp_output_dir` - Temporary directory for test files

## Development Workflow

### Before Starting Work

```bash
# Make sure tests pass
make test-fast
```

### While Developing

```bash
# Run tests related to what you're working on
pytest tests/unit/test_bounding_box.py -v

# Or run specific test
pytest tests/unit/test_bounding_box.py::test_iou_identical_boxes -v

# Run continuously (re-run on file changes)
pytest-watch
```

### Before Committing

```bash
# Run quality checks
make dev-check

# Or run individually:
make format          # Format code
make lint           # Check style
make test-fast      # Run fast tests
```

### Before Creating PR

```bash
# Run full test suite
make test

# Check coverage
make coverage
```

## Debugging Failed Tests

### 1. Read the Error Message

```
FAILED tests/unit/test_bounding_box.py::test_iou - AssertionError: assert 0.5 == 1.0
```

This tells you:
- **File**: `tests/unit/test_bounding_box.py`
- **Test**: `test_iou`
- **Error**: Expected 1.0, got 0.5

### 2. Run with Verbose Output

```bash
# Show more details
pytest tests/unit/test_bounding_box.py::test_iou -vv

# Show print statements
pytest tests/unit/test_bounding_box.py::test_iou -s

# Show local variables
pytest tests/unit/test_bounding_box.py::test_iou -l
```

### 3. Use Python Debugger

```python
def test_my_function():
    result = my_function(input)

    # Drop into debugger
    import pdb; pdb.set_trace()

    assert result == expected
```

Then run:
```bash
pytest tests/unit/test_my_feature.py -s
```

## Test Categories (Markers)

```bash
# Unit tests only (fastest)
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"

# Combine markers
pytest -m "unit and not slow"
```

## Useful Commands Cheatsheet

```bash
# Running tests
pytest                          # All tests
pytest -v                       # Verbose
pytest -x                       # Stop at first failure
pytest --lf                     # Last failed tests only
pytest -k "iou"                # Tests matching "iou"
pytest -n auto                  # Parallel execution

# Coverage
pytest --cov=src                # Show coverage
pytest --cov-report=html        # HTML report
make coverage                   # Open HTML report in browser

# Code quality
make format                     # Format code
make lint                       # Check style
make type-check                # Check types

# Shortcuts
make test-fast                  # Quick test
make test                       # Full test with coverage
make dev-check                  # Pre-commit checks
```

## Getting Help

- **Full guide**: See `TESTING.md`
- **Pytest docs**: https://docs.pytest.org/
- **Examples**: Check existing tests in `tests/`
- **Fixtures**: See `tests/conftest.py`

## Next Steps

1. ✅ Run `make test-fast` to verify setup
2. ✅ Read through existing tests in `tests/unit/`
3. ✅ Write a simple test for practice
4. ✅ Read `TESTING.md` for advanced topics
5. ✅ Set up pre-commit hooks for automatic testing

Happy testing! 🧪
