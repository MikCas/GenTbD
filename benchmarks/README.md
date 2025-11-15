# GenTbD Performance Benchmarking

This directory contains all performance testing and profiling tools.

## Quick Start

### 1. Install Dependencies

```bash
pip install pytest-benchmark py-spy line-profiler memory-profiler
```

### 2. Run Device Comparison

```bash
# Compare all available devices (CPU, MPS, CUDA)
python benchmarks/benchmark_devices.py

# Benchmark specific device
python benchmarks/benchmark_devices.py --device mps --frames 100

# With custom video
python benchmarks/benchmark_devices.py --video path/to/video.mp4
```

### 3. Run Pytest Benchmarks

```bash
# Run all benchmarks
pytest benchmarks/test_performance.py --benchmark-only

# Save results
pytest benchmarks/test_performance.py --benchmark-autosave

# Compare with previous run
pytest benchmarks/test_performance.py --benchmark-compare=0001
```

### 4. Profile with py-spy

```bash
# Profile MPS device
bash benchmarks/profile_with_pyspy.sh mps

# Profile CPU device
bash benchmarks/profile_with_pyspy.sh cpu

# Open flamegraph (shows where time is spent)
open benchmarks/profiles/flamegraph_mps_*.svg
```

---

## Tools Overview

### `benchmark_devices.py`
**Purpose**: Compare CPU vs MPS vs CUDA performance
**Output**: Detailed timing breakdown + speedup comparison
**Use when**: You want to know which device is faster

**Example output:**
```
Results: MPS
  Preprocess:  8.5 ms
  Inference:   45.2 ms
  Postprocess: 18.3 ms
  Total:       72.0 ± 4.2 ms
  FPS: 13.89

⚠️ MPS is SLOWER than CPU!
→ Postprocess is 18.3ms (too high!)
→ Fix: Keep tensors on GPU longer
```

### `test_performance.py`
**Purpose**: Automated regression testing with pytest-benchmark
**Output**: Statistical comparison across runs
**Use when**: You want to track performance over time

**Example:**
```bash
pytest benchmarks/test_performance.py --benchmark-autosave
# Make code changes
pytest benchmarks/test_performance.py --benchmark-compare=0001
```

### `profile_with_pyspy.sh`
**Purpose**: Visual profiling (flamegraph)
**Output**: SVG flamegraph showing time per function
**Use when**: You want to see WHERE the slowdown is

**Identifies:**
- Which functions take most time (wide bars)
- CPU-GPU transfer overhead (`.cpu()` calls)
- Python loops vs vectorized operations

---

## Common Workflows

### Diagnose MPS Slowdown

```bash
# 1. Confirm MPS is actually slower
python benchmarks/benchmark_devices.py

# 2. Profile to see WHERE time is spent
bash benchmarks/profile_with_pyspy.sh mps

# 3. Look at flamegraph
open benchmarks/profiles/flamegraph_mps_*.svg

# 4. Fix identified bottlenecks (usually postprocess)

# 5. Re-benchmark to confirm improvement
python benchmarks/benchmark_devices.py --device mps
```

### Track Optimization Progress

```bash
# Baseline before optimizations
pytest benchmarks/test_performance.py --benchmark-autosave
# Output: Saved as 0001_*.json

# Make optimization changes to code

# Compare with baseline
pytest benchmarks/test_performance.py --benchmark-compare=0001
# Shows % improvement/regression
```

### Profile Specific Component

```python
# Add to test_performance.py:
def test_postprocess_only(benchmark, detector_mps):
    # ... setup ...
    result = benchmark(detector_mps.postprocess, output, (480, 640))
```

---

## Understanding Results

### Benchmark Output

```
Results: MPS
  Preprocess:  8.5 ms   ← Should be <5ms on GPU
  Inference:   45.2 ms  ← Main computation (OK)
  Postprocess: 18.3 ms  ← Should be <5ms on GPU!
  FPS: 13.89
```

**Red flags:**
- Postprocess > 10ms → Too many CPU-GPU transfers
- Preprocess > 5ms on GPU → Not using GPU for preprocessing
- MPS slower than CPU → Device overhead > computation benefit

### Flamegraph Interpretation

**Wide bars = hotspots (time-consuming functions)**

Look for:
- `torch.Tensor.cpu` calls (red flag for MPS)
- Python loops in `postprocess`
- `cv2.cvtColor` taking significant time

---

## Optimization Checklist

Based on benchmark results:

- [ ] **Postprocess > 10ms?** → Implement GPU filtering (OPTIMIZATION_ANALYSIS.md #1)
- [ ] **Preprocess > 5ms on GPU?** → Move to GPU (OPTIMIZATION_ANALYSIS.md #2)
- [ ] **MPS slower than CPU?** → Check for `.cpu()` calls
- [ ] **Inference > 100ms?** → Reduce frame size or use MobileNet
- [ ] **Total > 150ms?** → Enable frame skipping

---

## Advanced Profiling

### Line-by-line profiling

```bash
# Install
pip install line-profiler

# Add @profile decorator to function
# Run
kernprof -l -v src/detecting/detectors/object_detector.py
```

### Memory profiling

```bash
# Install
pip install memory-profiler

# Run
python -m memory_profiler benchmarks/benchmark_devices.py
```

### PyTorch profiler

```python
# Add to code:
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    detector.detect(frame)

print(prof.key_averages().table(sort_by="cpu_time_total"))
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

---

## Expected Performance

### M3 Chip

| Configuration | Expected FPS | Notes |
|---------------|--------------|-------|
| CPU, mobilenet, 1080p | 5-10 | Baseline |
| CPU, mobilenet, 480p | 15-30 | Resize optimization |
| MPS, mobilenet, 1080p (current) | 5-15 | May be slower due to bugs |
| MPS, mobilenet, 1080p (optimized) | 50-150 | After fixes |

### Optimization Impact

| Optimization | Expected Improvement |
|--------------|---------------------|
| GPU filtering (keep tensors on device) | 3-5x |
| GPU preprocessing | 1.5-2x |
| Batching (batch_size=4) | 2-3x |
| FP16 precision | 1.5-2x |

---

## Troubleshooting

**"py-spy not installed"**
```bash
pip install py-spy
```

**"Permission denied" on macOS**
```bash
sudo py-spy record ...
# Or disable SIP (not recommended)
```

**"Video not found"**
```bash
# Specify video path
python benchmarks/benchmark_devices.py --video /full/path/to/video.mp4
```

**"MPS not available"**
- Only works on Apple Silicon (M1/M2/M3)
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

---

For detailed optimization strategies, see: `OPTIMIZATION_ANALYSIS.md`
