# Performance Benchmarking Guide

This directory contains comprehensive benchmarking tools for analyzing detector performance across different devices and configurations.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Run Quick Benchmark
```bash
python benchmarks/comprehensive_benchmark.py --quick
```

### 3. Run Full Benchmark
```bash
python benchmarks/comprehensive_benchmark.py --video data/TownCent.mp4
```

## Benchmark Scripts

### comprehensive_benchmark.py
**Purpose**: Compare all detector types across devices

**Features**:
- Tests object detection (mobilenet, resnet50, retinanet)
- Tests keypoint detection (resnet50)
- Tests ReID feature extraction (osnet variants)
- Compares CPU vs MPS vs CUDA (if available)
- Tests multiple resolutions (320p, 640p, 960p, 720p)
- Generates performance summary tables
- Saves results to JSON

**Usage Examples**:
```bash
# Quick test (5 iterations, 640p only)
python benchmarks/comprehensive_benchmark.py --quick

# Full test (10 iterations, all resolutions)
python benchmarks/comprehensive_benchmark.py

# Test specific devices
python benchmarks/comprehensive_benchmark.py --devices cpu mps

# Custom iterations
python benchmarks/comprehensive_benchmark.py --iterations 20
```

## Expected Performance (640x360 resolution)

### Object Detection
| Model | CPU | MPS | CUDA |
|-------|-----|-----|------|
| mobilenet | 5-6 | 20-30 | 60-100 |
| resnet50 | 0.5-2 | 8-15 | 30-60 |
| retinanet | 1-3 | 10-20 | 40-80 |

### Keypoint Detection  
| Model | CPU | MPS | CUDA |
|-------|-----|-----|------|
| resnet50 | 0.3-1 | 5-10 | 20-40 |

### ReID (128x256 crops)
| Model | CPU | MPS | CUDA |
|-------|-----|-----|------|
| osnet_x1_0 | 50-100 | 200-400 | 500-1000 |

## Performance Tips

1. **Use GPU**: `--device mps` or `--device cuda` (10-20x faster)
2. **Lower resolution**: `--max-dimension 480` (faster, slight accuracy loss)
3. **Frame skipping**: `--skip-frames 2` (2x effective FPS)
4. **Lighter models**: mobilenet vs resnet50 (3-5x faster)
