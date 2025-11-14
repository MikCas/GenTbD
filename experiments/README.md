# Experiments

This directory contains experimental scripts, performance tests, and documentation for exploring GenTbD features.

---

## Current Experiments

### 1. Performance Optimization (November 2025)

**[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - Complete writeup of CPU performance optimization

**Goal**: Achieve real-time object detection on CPU without GPU acceleration

**Result**: **270x speedup** (0.5 FPS → 135 FPS)

**Techniques**:
1. Model selection (ResNet50 → MobileNet): 15x faster
2. Frame resizing (process at lower resolution): 6x faster
3. Frame skipping (temporal reuse): 3x faster
4. Combined multiplicative gain: 15 × 6 × 3 = 270x

**Key Contributions**:
- Three orthogonal optimization strategies
- Minimal code changes (<200 lines)
- Configurable speed/accuracy tradeoffs
- Validated on real-world video

**Read the full experiment**: [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)

---

## Testing & Reproduction

### Run Performance Tests

```bash
# From project root
source venv/bin/activate
python experiments/test_performance.py
```

This script runs 4 configurations:
1. Baseline (ResNet50, full-res, all frames)
2. MobileNet only
3. MobileNet + resize to 640px
4. MobileNet + resize + skip frames

Watch the FPS counter in each test, press 'q' to move to next.

### Try Different Configurations

```bash
# Fast mode (recommended)
python src/process_video.py --model mobilenet --max-dimension 640 --skip-frames 3

# Balanced mode
python src/process_video.py --model mobilenet --max-dimension 640 --skip-frames 2

# High accuracy (slower)
python src/process_video.py --model resnet50 --conf 0.7

# Apple Silicon acceleration
python src/process_video.py --model mobilenet --device mps
```

---

## Experiment Structure

Each experiment in this directory should follow this structure:

### Documentation
- **Problem statement**: What we're trying to solve
- **Hypothesis**: Expected solution and benefits
- **Implementation**: Technical details and code changes
- **Results**: Benchmarks, measurements, comparisons
- **Analysis**: What worked, what didn't, lessons learned
- **Conclusion**: Key takeaways and next steps

### Code
- Test scripts for reproduction
- Benchmark utilities
- Visualization tools (if applicable)

### Data
- Sample inputs/outputs
- Performance metrics
- Comparison tables

---

## Future Experiments

Planned experiments for this directory:

### Tracking Algorithms
- Compare IoU-based vs Kalman-based tracking
- Benchmark track ID persistence
- Evaluate handling of occlusions
- Test on crowded scenes

### Re-Identification (ReID)
- Compare ReID models (OSNet, FastReID, torchreid)
- Measure embedding quality vs speed
- Test appearance-based re-matching
- Evaluate on person re-identification benchmarks

### Keypoint Detection
- Compare keypoint models (Keypoint R-CNN, HRNet, etc.)
- Measure pose estimation accuracy
- Test temporal smoothness of keypoints
- Benchmark on action recognition tasks

### Multi-Camera Tracking
- Camera calibration and synchronization
- Cross-camera object matching
- Global trajectory optimization
- Scalability testing

### Dataset Evaluation
- Benchmark on MOT (Multi-Object Tracking) datasets
- Calculate MOTA, MOTP, IDF1 metrics
- Compare with state-of-the-art methods
- Error analysis and failure cases

---

## Adding New Experiments

When adding a new experiment:

1. **Create a markdown document** (e.g., `MY_EXPERIMENT.md`)
   - Follow the structure above
   - Include problem, solution, results, analysis

2. **Add test/benchmark scripts**
   - Make them runnable from project root
   - Document dependencies
   - Include usage examples

3. **Update this README**
   - Add entry to "Current Experiments"
   - Link to your document
   - Brief summary of results

4. **Document in git**
   - Commit with clear message
   - Tag if it's a major milestone
   - Update CHANGELOG if appropriate

---

## Experiment Guidelines

### Scientific Rigor
- Measure baseline before changes
- Use consistent test conditions
- Report both speed and accuracy
- Include error bars/variance where applicable

### Reproducibility
- Document exact environment (Python version, hardware, etc.)
- Provide commands to reproduce results
- Include sample data if possible
- Version control all code

### Documentation
- Explain *why* not just *what*
- Include diagrams/visualizations
- Document failed approaches (negative results are valuable!)
- Write for future you and others

### Code Quality
- Clean, readable code
- Add comments for complex logic
- Follow project style
- Include docstrings

---

## Resources

### Performance Analysis
- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION.md)
- [Test Script](test_performance.py)
- Main source: [src/](../src/)

### Related Papers
- [Tracking paper](../Tracking_paper.pdf) - Original inspiration
- ByteTrack, DeepSORT, StrongSORT - Referenced in ARCHIVE

### External Benchmarks
- MOT Challenge: https://motchallenge.net/
- COCO Detection: https://cocodataset.org/
- Keypoint Detection: MPII, COCO-Pose

---

## Summary

This experiments directory serves as:
1. **Documentation** of optimization techniques and design decisions
2. **Validation** through benchmarks and performance tests
3. **Knowledge base** for future development
4. **Reproducibility** with runnable scripts and clear procedures

All major features should be prototyped and validated here before integration into main codebase.

---

**Current Status**: ✅ Phase 1 Complete (Performance Optimization)
**Next**: Phase 2 - YAML Configuration System
**Future**: Tracking, ReID, Keypoints experiments
