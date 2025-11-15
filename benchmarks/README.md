# Performance Benchmarking Suite

This directory contains comprehensive benchmarking tools to analyze the performance of the GenTbD detection system.

## Benchmark Files

### 1. `timed_detector.py` - Per-Stage Timing Analysis
**Purpose:** Measure time spent in each pipeline stage (preprocess, inference, postprocess)

**What it reveals:**
- Which stage is the bottleneck
- Percentage breakdown of time
- FPS statistics (mean, P50, P95, P99)
- High variance detection

**Run:**
```bash
python benchmarks/timed_detector.py
```

**Key insights:**
- If inference > 70%: Model is bottleneck → use lighter model or lower resolution
- If preprocess > 20%: Image conversion is slow → optimize cv2/tensor operations
- If postprocess > 15%: GPU→CPU transfer or object creation is slow

---

### 2. `preprocessing_bench.py` - Preprocessing Optimization
**Purpose:** Compare different preprocessing approaches

**What it tests:**
- cv2.cvtColor vs numpy slicing (BGR→RGB)
- Division vs multiplication normalization
- Different tensor creation methods
- CPU→GPU transfer overhead

**Run:**
```bash
python benchmarks/preprocessing_bench.py
```

**Key insights:**
- Identifies fastest preprocessing method
- Measures transfer overhead (CPU→MPS/CUDA)
- Component-level timing (color conversion, normalization, permute)

---

### 3. `inference_bench.py` - Inference Performance
**Purpose:** Benchmark model inference under different conditions

**What it tests:**
- Single-frame vs batched inference
- Different resolutions (320p, 480p, 720p, 1080p)
- Different models (MobileNet, ResNet50)
- CPU vs MPS/CUDA performance

**Run:**
```bash
python benchmarks/inference_bench.py
```

**Key insights:**
- Resolution scaling behavior
- Batching efficiency (important for throughput)
- Model comparison (speed vs accuracy tradeoff)
- Device performance comparison

---

### 4. `postprocessing_bench.py` - Postprocessing Breakdown
**Purpose:** Analyze postprocessing bottlenecks

**What it tests:**
- GPU→CPU transfer time
- Confidence filtering performance
- Class filtering performance
- Detection object creation overhead

**Run:**
```bash
python benchmarks/postprocessing_bench.py
```

**Key insights:**
- Is GPU→CPU transfer significant?
- Is filtering slow for many detections?
- Is Python object creation a bottleneck?
- Optimization opportunities (vectorized filtering)

---

### 5. `e2e_benchmark.py` - End-to-End Comparison
**Purpose:** Compare all model/device/resolution combinations

**What it tests:**
- All combinations of:
  - Models: MobileNet, ResNet50
  - Devices: CPU, MPS, CUDA
  - Resolutions: 640x480, 1280x720

**Run:**
```bash
python benchmarks/e2e_benchmark.py
```

**Output:**
- Comprehensive results table (CSV)
- Device comparison
- Model comparison
- Resolution scaling analysis
- Optimization recommendations

**Key insights:**
- Best configuration for your hardware
- Real-time capable configurations (≥30 FPS)
- MPS vs CPU speedup
- Resolution efficiency

---

### 6. `profiler_analysis.py` - PyTorch Profiler
**Purpose:** Operation-level profiling using PyTorch's profiler

**What it reveals:**
- Top 20 slowest operations
- Pipeline stage breakdown
- Memory usage analysis
- Automatic bottleneck detection
- Chrome trace export for visualization

**Run:**
```bash
python benchmarks/profiler_analysis.py
```

**Output:**
- Operation timing table
- Bottleneck identification
- Memory usage breakdown
- Optimization recommendations
- `trace_*.json` files for chrome://tracing

**Key insights:**
- Exact PyTorch operations that are slow
- Data transfer overhead (aten::to, aten::copy_)
- Memory allocation patterns
- Unexpected operations (debugging)

---

## Quick Start

### 1. Identify Overall Bottleneck
```bash
python benchmarks/timed_detector.py
```
Look at the stage breakdown. This tells you where to focus optimization efforts.

### 2. Deep Dive into Bottleneck
If **inference** is slow:
```bash
python benchmarks/inference_bench.py
```

If **preprocessing** is slow:
```bash
python benchmarks/preprocessing_bench.py
```

If **postprocessing** is slow:
```bash
python benchmarks/postprocessing_bench.py
```

### 3. Operation-Level Analysis
```bash
python benchmarks/profiler_analysis.py
```
Use this to identify specific PyTorch operations causing slowness.

### 4. Compare All Configurations
```bash
python benchmarks/e2e_benchmark.py
```
Find the best model/device/resolution for your use case.

---

## Understanding Results

### FPS Metrics
- **Mean FPS**: Average frames per second
- **P50 FPS**: Median performance (typical case)
- **P95 FPS**: 95th percentile (most frames faster than this)
- **P99 FPS**: 99th percentile (worst-case latency)

### Performance Targets
- **Real-time video**: ≥30 FPS
- **Smooth tracking**: ≥20 FPS
- **Acceptable**: ≥10 FPS
- **Too slow**: <10 FPS

### Bottleneck Interpretation

**Inference dominates (>70%):**
- ✓ Model is the bottleneck
- → Use MobileNet instead of ResNet50
- → Lower resolution (640x480 instead of 1080p)
- → Frame skipping (process every Nth frame)

**Preprocessing significant (>20%):**
- ✓ Image conversion is slow
- → Optimize BGR→RGB conversion
- → Use faster normalization
- → Reduce resolution before preprocessing

**Postprocessing significant (>15%):**
- ✓ GPU→CPU transfer or object creation slow
- → Keep tensors on GPU longer
- → Vectorize filtering operations
- → Batch detection creation

**High variance (CV > 0.2):**
- ✓ Inconsistent performance
- → JIT compilation not warmed up
- → Memory allocation spikes
- → Thermal throttling (long runs)

---

## Troubleshooting

### MPS (Apple Silicon) Appears Slow

**Symptom:** MPS slower than CPU or very low FPS

**Common causes:**
1. **Missing synchronization** → Measuring kernel launch, not execution
2. **CPU fallback** → Some ops not MPS-compatible
3. **Transfer overhead** → Excessive CPU↔MPS copying
4. **Small batch size** → MPS optimized for batches

**Diagnostics:**
```bash
# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i 100

# Look for:
# - GPU Power: 5-15W (GPU working)
# - GPU Active Residency: 80-99% (GPU utilized)

# If GPU Power <1W → MPS not being used!
```

**Solutions:**
- Ensure all tensors on MPS: `tensor.to('mps')`
- Add synchronization: `torch.mps.synchronize()`
- Try batched inference
- Update to PyTorch 2.4+ and macOS 15+

### Low FPS on All Devices

**Check:**
1. Is warmup being done? (First runs are always slow)
2. Is model in eval mode? (`model.eval()`)
3. Is torch.no_grad() used? (Disables gradient computation)
4. Is resolution too high? (Try 640x480)

### High Memory Usage

**Run:**
```bash
python benchmarks/profiler_analysis.py
```

Look at "MEMORY USAGE ANALYSIS" section for memory-hungry operations.

---

## Best Practices

### 1. Always Warmup
```python
# Bad: No warmup
for frame in frames:
    time = measure(detector.detect(frame))  # First iteration slow!

# Good: Warmup first
for _ in range(5):
    detector.detect(frame)  # Warmup JIT compilation

for frame in frames:
    time = measure(detector.detect(frame))  # Accurate now
```

### 2. Always Synchronize (GPU/MPS)
```python
# Bad: Measures kernel launch only
start = time.time()
output = model(input)
elapsed = time.time() - start  # WRONG!

# Good: Wait for GPU completion
start = time.time()
output = model(input)
torch.mps.synchronize()  # or torch.cuda.synchronize()
elapsed = time.time() - start  # Correct
```

### 3. Report Percentiles, Not Just Mean
```python
# Good: Report distribution
print(f"Mean: {np.mean(times)} ms")
print(f"P50:  {np.percentile(times, 50)} ms")
print(f"P95:  {np.percentile(times, 95)} ms")
print(f"P99:  {np.percentile(times, 99)} ms")
```

---

## Export Results

All benchmarks save results to files:
- `e2e_benchmark.py` → `benchmark_results.csv`
- `profiler_analysis.py` → `trace_*.json` (visualize in chrome://tracing)

Share results:
```bash
# Include in performance reports
cat benchmark_results.csv

# Visualize profiler trace
# 1. Open chrome://tracing in Chrome
# 2. Load trace_*.json file
# 3. Zoom and pan to see operation timeline
```

---

## Next Steps

After running benchmarks:

1. **Identify bottleneck** from `timed_detector.py`
2. **Analyze root cause** from `profiler_analysis.py`
3. **Test optimizations** by modifying code and re-benchmarking
4. **Compare configurations** with `e2e_benchmark.py`
5. **Document improvements** in experiments/

---

For questions or issues, see:
- PyTorch Profiler docs: https://pytorch.org/tutorials/beginner/profiler.html
- Performance optimization guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
