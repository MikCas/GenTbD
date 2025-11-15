"""
Pytest-benchmark based performance tests.

Install: pip install pytest-benchmark

Usage:
    pytest benchmarks/test_performance.py --benchmark-compare
    pytest benchmarks/test_performance.py --benchmark-only
    pytest benchmarks/test_performance.py --benchmark-autosave
    pytest benchmarks/test_performance.py --benchmark-compare=0001
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from src.detecting.detectors import ObjectDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def sample_frame_small():
    """Small 480x640 test frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope='session')
def sample_frame_medium():
    """Medium 720x1280 test frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture(scope='session')
def sample_frame_large():
    """Large 1080x1920 test frame."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture(scope='session')
def detector_cpu():
    """CPU detector."""
    return ObjectDetector(model='mobilenet', device='cpu', conf_threshold=0.5)


@pytest.fixture(scope='session')
def detector_mps():
    """MPS detector (if available)."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return ObjectDetector(model='mobilenet', device='mps', conf_threshold=0.5)


@pytest.fixture(scope='session')
def detector_cuda():
    """CUDA detector (if available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return ObjectDetector(model='mobilenet', device='cuda', conf_threshold=0.5)


# ============================================================================
# CPU Benchmarks
# ============================================================================

def test_cpu_small_frame(benchmark, detector_cpu, sample_frame_small):
    """Benchmark CPU on 480x640 frame."""
    result = benchmark(detector_cpu.detect, sample_frame_small)
    assert isinstance(result, list)


def test_cpu_medium_frame(benchmark, detector_cpu, sample_frame_medium):
    """Benchmark CPU on 720x1280 frame."""
    result = benchmark(detector_cpu.detect, sample_frame_medium)
    assert isinstance(result, list)


def test_cpu_large_frame(benchmark, detector_cpu, sample_frame_large):
    """Benchmark CPU on 1080x1920 frame."""
    result = benchmark(detector_cpu.detect, sample_frame_large)
    assert isinstance(result, list)


# ============================================================================
# MPS Benchmarks
# ============================================================================

def test_mps_small_frame(benchmark, detector_mps, sample_frame_small):
    """Benchmark MPS on 480x640 frame."""
    result = benchmark(detector_mps.detect, sample_frame_small)
    assert isinstance(result, list)


def test_mps_medium_frame(benchmark, detector_mps, sample_frame_medium):
    """Benchmark MPS on 720x1280 frame."""
    result = benchmark(detector_mps.detect, sample_frame_medium)
    assert isinstance(result, list)


def test_mps_large_frame(benchmark, detector_mps, sample_frame_large):
    """Benchmark MPS on 1080x1920 frame."""
    result = benchmark(detector_mps.detect, sample_frame_large)
    assert isinstance(result, list)


# ============================================================================
# CUDA Benchmarks
# ============================================================================

def test_cuda_small_frame(benchmark, detector_cuda, sample_frame_small):
    """Benchmark CUDA on 480x640 frame."""
    result = benchmark(detector_cuda.detect, sample_frame_small)
    assert isinstance(result, list)


def test_cuda_medium_frame(benchmark, detector_cuda, sample_frame_medium):
    """Benchmark CUDA on 720x1280 frame."""
    result = benchmark(detector_cuda.detect, sample_frame_medium)
    assert isinstance(result, list)


def test_cuda_large_frame(benchmark, detector_cuda, sample_frame_large):
    """Benchmark CUDA on 1080x1920 frame."""
    result = benchmark(detector_cuda.detect, sample_frame_large)
    assert isinstance(result, list)


# ============================================================================
# Component Benchmarks
# ============================================================================

def test_cpu_preprocess_only(benchmark, detector_cpu, sample_frame_medium):
    """Benchmark preprocessing only (CPU)."""
    result = benchmark(detector_cpu.preprocess, sample_frame_medium)
    assert result is not None


def test_mps_preprocess_only(benchmark, detector_mps, sample_frame_medium):
    """Benchmark preprocessing only (MPS)."""
    result = benchmark(detector_mps.preprocess, sample_frame_medium)
    assert result is not None


def test_cpu_inference_only(benchmark, detector_cpu, sample_frame_medium):
    """Benchmark inference only (CPU)."""
    input_tensor = detector_cpu.preprocess(sample_frame_medium)

    def run_inference():
        with torch.no_grad():
            return detector_cpu.inference(input_tensor)

    result = benchmark(run_inference)
    assert result is not None


def test_mps_inference_only(benchmark, detector_mps, sample_frame_medium):
    """Benchmark inference only (MPS)."""
    input_tensor = detector_mps.preprocess(sample_frame_medium)

    def run_inference():
        with torch.no_grad():
            output = detector_mps.inference(input_tensor)
        torch.mps.synchronize()
        return output

    result = benchmark(run_inference)
    assert result is not None
