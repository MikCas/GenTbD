# Makefile for GenTbD project
# Common development commands

.PHONY: help install install-dev test test-unit test-integration test-fast coverage clean format lint type-check

help:  ## Show this help message
	@echo "GenTbD Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:  ## Run all tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-unit:  ## Run only unit tests (fast)
	pytest -m unit -v

test-integration:  ## Run only integration tests
	pytest -m integration -v

test-fast:  ## Run fast tests only (skip slow tests)
	pytest -m "not slow" -v

test-parallel:  ## Run tests in parallel
	pytest -n auto

test-verbose:  ## Run tests with verbose output
	pytest -vv -s

test-lf:  ## Run only tests that failed last time
	pytest --lf

coverage:  ## Generate coverage report and open in browser
	pytest --cov=src --cov-report=html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html || open htmlcov/index.html || xdg-open htmlcov/index.html

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check:  ## Check code formatting without changing files
	black --check src/ tests/
	isort --check-only src/ tests/

lint:  ## Run flake8 linter
	flake8 src/ tests/ --max-line-length=100

type-check:  ## Run mypy type checker
	mypy src/

quality:  ## Run all code quality checks
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) type-check

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

benchmark:  ## Run performance benchmarks
	pytest -m benchmark --benchmark-only

watch:  ## Run tests continuously (requires pytest-watch)
	pytest-watch

ci:  ## Run CI pipeline locally
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-fast

# Development workflow commands
dev-setup: install-dev  ## Complete development setup

dev-check: format lint type-check test-fast  ## Quick development check before commit
