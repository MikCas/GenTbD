# Benchmarking & Performance Analysis Guide

**A comprehensive guide to proper benchmarking methodology for performance optimizations**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Proper Benchmarking Matters](#why-proper-benchmarking-matters)
3. [Benchmarking Methodology](#benchmarking-methodology)
4. [Common Pitfalls](#common-pitfalls)
5. [Statistical Analysis](#statistical-analysis)
6. [Interpreting Results](#interpreting-results)
7. [Using the Framework](#using-the-framework)

---

## Introduction

Performance optimization without proper benchmarking is like driving with your eyes closed. You might get lucky, but you're more likely to make things worse. This guide teaches you how to properly measure performance improvements.

### Key Principles

1. **Measure Before Optimizing**: Never optimize without a baseline
2. **Use Statistics**: Single measurements are meaningless
3. **Control Variables**: Test one thing at a time
4. **Be Skeptical**: Small improvements might just be noise
5. **Document Everything**: Future you will thank present you

---

## Why Proper Benchmarking Matters

### The Problem with "Quick Tests"

```python
# ❌ BAD: Single measurement
start = time.time()
result = my_function()
print(f"Took {time.time() - start} seconds")
```

**Problems**:
- One measurement tells you nothing about variability
- No warmup (first run is always slower)
- Python GC can trigger at any time
- OS scheduler can interrupt your process
- CPU frequency scaling affects results

### The Right Way

```python
# ✅ GOOD: Proper benchmark
def benchmark_function(func, num_warmup=5, num_samples=30):
    # Warmup
    for _ in range(num_warmup):
        func()

    # Collect samples
    samples = []
    for _ in range(num_samples):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)

    # Statistics
    mean = statistics.mean(samples)
    std = statistics.stdev(samples)

    return mean, std, samples
```

---

## Benchmarking Methodology

### Step 1: Warmup Iterations

**Purpose**: Stabilize the system before measuring

```python
# Why warmup is critical:
# - GPU kernels are JIT-compiled on first use
# - CPU caches need to be populated
# - Python bytecode is compiled on first execution
# - CUDA/MPS streams need initialization
# - File system caches need population

def warmup(func, iterations=5):
    """Run function multiple times to warm up."""
    for _ in range(iterations):
        func()

    # Synchronize GPU if needed
    if using_gpu:
        torch.cuda.synchronize()  # or torch.mps.synchronize()
```

**Example**: First PyTorch model call is always slower:

```
Iteration 1: 150ms  ← JIT compilation happening
Iteration 2:  50ms  ← Stabilized
Iteration 3:  48ms
Iteration 4:  49ms
Iteration 5:  51ms
```

### Step 2: Collect Multiple Samples

**Purpose**: Understand variability and detect outliers

```python
# Recommended sample sizes:
# - Quick test: 10-20 samples
# - Standard: 30-50 samples
# - Rigorous: 100+ samples

samples = []
for i in range(30):
    start = time.perf_counter()  # Use perf_counter, not time()!
    result = my_function()
    elapsed = time.perf_counter() - start
    samples.append(elapsed)
```

**Why 30+ samples?**
- Central Limit Theorem applies (normal distribution)
- Can detect outliers reliably
- Can calculate confidence intervals
- Statistical tests become valid

### Step 3: Synchronize GPU Operations

**Purpose**: Ensure GPU work completes before stopping timer

```python
# ❌ BAD: Timer stops before GPU finishes
start = time.perf_counter()
output = model(input)  # Returns immediately (async GPU call)
elapsed = time.perf_counter() - start  # Wrong!

# ✅ GOOD: Wait for GPU to finish
start = time.perf_counter()
output = model(input)
torch.cuda.synchronize()  # Wait for GPU
elapsed = time.perf_counter() - start  # Correct!
```

### Step 4: Calculate Statistics

**Purpose**: Understand performance distribution

```python
import numpy as np

samples = [...]  # Your timing samples

# Central tendency
mean = np.mean(samples)
median = np.median(samples)

# Variability
std = np.std(samples)
min_time = np.min(samples)
max_time = np.max(samples)

# Percentiles (important for real-world performance)
p95 = np.percentile(samples, 95)  # 95% of runs are faster than this
p99 = np.percentile(samples, 99)  # 99% of runs are faster than this
```

**Why percentiles matter**: In production, you care about "worst-case" performance, not just average.

### Step 5: Detect Outliers

**Purpose**: Identify anomalous measurements (GC, OS interrupts)

```python
def detect_outliers(samples):
    """Detect outliers using IQR method."""
    q1, q3 = np.percentile(samples, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [s for s in samples if s < lower_bound or s > upper_bound]
    return outliers
```

**Should you remove outliers?**
- **For analysis**: Yes, to understand "normal" performance
- **For reporting**: No, report both with and without outliers
- **For comparison**: Include outliers (they happen in production too)

---

## Common Pitfalls

### Pitfall 1: Not Controlling for External Factors

```python
# ❌ BAD: Running benchmarks while other apps are running
# ❌ BAD: CPU frequency scaling enabled (laptop on battery)
# ❌ BAD: Different CPU temperature between runs
# ❌ BAD: Network activity affecting I/O benchmarks

# ✅ GOOD:
# - Close unnecessary applications
# - Plug in laptop (disable power saving)
# - Let system cool down between runs
# - Disable automatic updates/backups
```

### Pitfall 2: Optimizing the Wrong Thing

```python
# Example: You optimize postprocessing by 50%
# But postprocessing is only 10% of total time!
# Result: Only 5% total improvement

# Always profile FIRST to find bottlenecks
import cProfile
cProfile.run('my_function()')
```

### Pitfall 3: Comparing Apples to Oranges

```python
# ❌ BAD: Different conditions
baseline = benchmark(func, device='cpu')
optimized = benchmark(func, device='gpu')  # Not a fair comparison!

# ✅ GOOD: Same conditions
baseline = benchmark(func, device='cpu', resolution=720)
optimized = benchmark(func, device='cpu', resolution=720)
```

### Pitfall 4: Ignoring Statistical Significance

```python
# Baseline:  100ms ± 5ms
# Optimized: 98ms ± 5ms

# Is 2ms improvement real? NO!
# The ranges overlap completely [95-105] vs [93-103]
# This is just measurement noise
```

---

## Statistical Analysis

### Calculating Speedup

```python
# Simple speedup ratio
speedup = baseline_time / optimized_time

# Example:
# Baseline:  100ms
# Optimized:  50ms
# Speedup:   100/50 = 2.0x faster
```

### Statistical Significance (Simplified t-test)

```python
def is_significant(baseline_samples, optimized_samples):
    """Check if improvement is statistically significant."""
    mean_b = np.mean(baseline_samples)
    mean_o = np.mean(optimized_samples)
    std_b = np.std(baseline_samples)
    std_o = np.std(optimized_samples)
    n = len(baseline_samples)

    # Pooled standard error
    pooled_std = np.sqrt((std_b**2 + std_o**2) / 2)
    std_error = pooled_std / np.sqrt(n)

    # t-statistic
    t = abs(mean_b - mean_o) / std_error

    # Rule of thumb: t > 2 means significant
    return t > 2, t
```

**Interpretation**:
- `t < 2`: Improvement likely just noise
- `t = 2-4`: Likely real improvement
- `t > 4`: Definitely real improvement

### Confidence Intervals

```python
from scipy import stats

def confidence_interval(samples, confidence=0.95):
    """Calculate confidence interval for mean."""
    mean = np.mean(samples)
    sem = stats.sem(samples)  # Standard error of mean
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(samples) - 1)
    return mean - interval, mean + interval

# Example:
# CI = (95ms, 105ms)
# Interpretation: 95% confident true mean is in this range
```

---

## Interpreting Results

### What Constitutes a "Good" Improvement?

| Speedup | Assessment | Action |
|---------|------------|--------|
| < 1.1x | Marginal | Probably not worth the complexity |
| 1.1-1.5x | Moderate | Worth it if code stays clean |
| 1.5-2x | Good | Definitely worth implementing |
| 2-5x | Excellent | Major win! |
| > 5x | Outstanding | You found a major bottleneck |

### Reading the Statistics

```
Mean:   50.2ms  ← Average performance (most important)
Median: 48.5ms  ← Middle value (robust to outliers)
Std:     5.3ms  ← Variability (lower is better)
Min:    45.1ms  ← Best case
Max:    67.8ms  ← Worst case (check for outliers!)
P95:    58.2ms  ← 95% of runs are faster than this
P99:    63.1ms  ← 99% of runs are faster than this
```

**What to focus on**:
- **Mean**: For average performance
- **Median**: If you have outliers
- **P95/P99**: For worst-case guarantees (important in production)
- **Std**: For consistency (low std = predictable performance)

### Red Flags

🚩 **High Standard Deviation** (std > 20% of mean)
- System is unstable
- Background processes interfering
- Measurement methodology issue

🚩 **Mean ≠ Median** (differ by > 10%)
- Outliers are present
- Distribution is skewed
- Need more samples or better control

🚩 **Speedup < 1.0** (slower after "optimization")
- Optimization didn't work
- Changed test conditions accidentally
- Introduced new overhead

---

## Using the Framework

### Quick Start

```bash
# 1. Run baseline benchmarks (BEFORE optimization)
python benchmarking/run_benchmarks.py --mode baseline

# 2. Apply your optimizations to the code
# (Edit the source files)

# 3. Run optimized benchmarks (AFTER optimization)
python benchmarking/run_benchmarks.py --mode optimized

# 4. Compare results
python benchmarking/run_benchmarks.py --mode compare
```

### Custom Benchmarks

```python
from benchmarking import BenchmarkRunner, BenchmarkConfig

# Create custom config
config = BenchmarkConfig(
    name="my_test",
    resolution=(1920, 1080),
    model="mobilenet",
    device="cpu",
    conf_threshold=0.5,
    classes=[1, 2, 3],
    num_warmup=10,      # More warmup for GPU
    num_samples=50,     # More samples for accuracy
)

# Run benchmark
runner = BenchmarkRunner()
result = runner.run_benchmark(detector, config)

# Access results
print(f"Mean: {result.mean_ms:.2f}ms")
print(f"FPS: {result.fps:.1f}")
print(f"P95: {result.p95_ms:.2f}ms")
```

### Analyzing Individual Components

```python
# Benchmark individual pipeline stages
def benchmark_stage(func, *args, **kwargs):
    samples = []

    # Warmup
    for _ in range(5):
        func(*args, **kwargs)

    # Measure
    for _ in range(30):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        samples.append((time.perf_counter() - start) * 1000)

    return samples

# Example: Benchmark each stage
preprocess_times = benchmark_stage(detector.preprocess, frame)
inference_times = benchmark_stage(detector.inference, tensor)
postprocess_times = benchmark_stage(detector.postprocess, output, shape)

# Find bottleneck
print(f"Preprocess:  {np.mean(preprocess_times):.2f}ms")
print(f"Inference:   {np.mean(inference_times):.2f}ms")
print(f"Postprocess: {np.mean(postprocess_times):.2f}ms")
```

---

## Best Practices Checklist

Before running benchmarks:
- [ ] Close unnecessary applications
- [ ] Disable automatic updates/backups
- [ ] Plug in laptop (disable power saving)
- [ ] Let system stabilize (wait 1-2 minutes)
- [ ] Check CPU temperature (don't benchmark while hot)

During benchmarking:
- [ ] Run warmup iterations (5-10)
- [ ] Collect sufficient samples (30+)
- [ ] Synchronize GPU operations
- [ ] Monitor for system interruptions
- [ ] Save raw samples (not just statistics)

After benchmarking:
- [ ] Calculate comprehensive statistics
- [ ] Detect and document outliers
- [ ] Check for statistical significance
- [ ] Compare multiple scenarios
- [ ] Document test conditions
- [ ] Save results with git commit hash

---

## Further Reading

### Books
- "Systems Performance" by Brendan Gregg
- "The Art of Computer Systems Performance Analysis" by Raj Jain

### Tools
- **cProfile**: Python profiling (`python -m cProfile script.py`)
- **line_profiler**: Line-by-line profiling
- **memory_profiler**: Memory usage profiling
- **torch.profiler**: PyTorch-specific profiling
- **nvprof/nsys**: NVIDIA GPU profiling

### Techniques
- **Microbenchmarking**: Test small components in isolation
- **System benchmarking**: Test full pipeline end-to-end
- **Profiling**: Find where time is actually spent
- **A/B testing**: Compare two implementations directly

---

## Conclusion

Proper benchmarking is a skill that separates amateur optimization from professional performance engineering. Key takeaways:

1. **Always measure before optimizing** - profile first, optimize second
2. **Use statistics** - single measurements are meaningless
3. **Control variables** - change one thing at a time
4. **Be skeptical** - small improvements need statistical validation
5. **Document everything** - future maintainers will thank you

Remember: **"In God we trust, all others must bring data."** - W. Edwards Deming

---

**Happy Benchmarking! 🚀**
