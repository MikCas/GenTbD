# GenTbD - Quick Start Guide

## Setup (5 minutes)

```bash
# 1. Navigate to project
cd GenTbD

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r benchmarks/requirements.txt

# 4. Verify
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Run Detection

```bash
# Webcam
python -m src.main --webcam --device cpu

# Video file
python -m src.main --video data/TownCent.mp4 --device cpu

# Optimized for speed
python -m src.main --video data/TownCent.mp4 \
    --device cpu \
    --model mobilenet \
    --max-dimension 480 \
    --skip-frames 2
```

---

## Benchmark Performance

```bash
# Compare all devices (CPU, MPS, CUDA)
python benchmarks/benchmark_devices.py

# MPS only
python benchmarks/benchmark_devices.py --device mps --frames 100

# With pytest-benchmark
pytest benchmarks/test_performance.py --benchmark-only
```

---

## Profile & Find Bottlenecks

```bash
# Profile MPS
bash benchmarks/profile_with_pyspy.sh mps

# Open flamegraph
open benchmarks/profiles/flamegraph_mps_*.svg
```

---

## Controls

- `c` - Toggle continuous/step mode
- `SPACE` - Next frame (step mode)
- `s` - Save frame
- `q` - Quit

---

## Common Issues

**MPS slower than CPU?**
→ See `OPTIMIZATION_ANALYSIS.md` Task 2

**Import errors?**
→ Run as module: `python -m src.main` (not `python src/main.py`)

**Video not found?**
→ Use full path: `--video /full/path/to/video.mp4`

---

For detailed docs: `README.md`, `CLAUDE.md`, `OPTIMIZATION_ANALYSIS.md`
