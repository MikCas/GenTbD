"""GenTbD test suite.

This package contains unit tests, integration tests, and end-to-end tests
for the GenTbD tracking-by-detection framework.

Test Organization:
    tests/unit/          - Fast, isolated unit tests
    tests/integration/   - Integration tests with real models
    tests/e2e/          - End-to-end workflow tests
    tests/fixtures/     - Shared test data
    conftest.py         - Shared pytest fixtures

Running Tests:
    pytest                      # Run all tests
    pytest -m unit             # Run only unit tests
    pytest -m integration      # Run only integration tests
    pytest -m "not slow"       # Skip slow tests

For more information, see TESTING.md
"""

__version__ = "0.1.0"
