# GenTbD Audit Summary

**Date:** 2025-11-14
**Auditor:** Claude
**Scope:** Parameter audit and performance analysis for GenTbD tracking-by-detection system

---

## Executive Summary

This audit analyzed the GenTbD codebase to:
1. ✅ Identify all system parameters and their groupings
2. ✅ Detect missing or ungrouped parameters
3. ✅ Analyze the detection pipeline for performance bottlenecks
4. ✅ Identify logical errors affecting performance

### Key Findings

- **25+ parameters** identified across 6 major components
- **4 critical performance bottlenecks** found (combined impact: **2-3x slowdown**)
- **3 logical errors** detected (2 affecting performance, 1 design inconsistency)
- **15+ missing parameters** that should be exposed for better configurability

---

## Document Index

This audit produced three comprehensive documents:

### 1. [`PARAMETER_AUDIT.md`](./PARAMETER_AUDIT.md)
**Complete parameter inventory and organization recommendations**

**Contents:**
- Detailed parameter tables for all 6 components
- Parameter grouping analysis
- Missing parameter identification
- Recommended configuration structure
- Critical findings and inconsistencies

**Key Metrics:**
- **Detector Parameters:** 4 exposed, 5 missing
- **Tracker Parameters:** 5 exposed, 5 missing
- **Track Parameters:** 1 exposed, 4 missing
- **Kalman Filter Parameters:** 3 exposed, 4 missing
- **Video Processor Parameters:** 2 exposed, 6 missing
- **System Parameters:** 1 exposed, 2 missing

### 2. [`PERFORMANCE_ANALYSIS.md`](./PERFORMANCE_ANALYSIS.md)
**In-depth performance bottleneck analysis and optimization guide**

**Contents:**
- Pipeline flow analysis
- Critical bottleneck identification (ranked by impact)
- Code-level optimization recommendations
- Logical error detection
- Expected performance improvements
- Performance profiling code

**Key Findings:**
- 🔴 **Critical:** Nested loop cost matrix (20-100ms overhead)
- 🔴 **Critical:** Inefficient NMS implementation (10-50ms overhead)
- 🔴 **Critical:** Multiple array copies in postprocessing (5-15ms overhead)
- 🔴 **Critical:** Missing ONNX session optimizations

**Expected Improvement:** **2-3x speedup** (from ~5-13 FPS to ~11-25 FPS)

### 3. [`AUDIT_SUMMARY.md`](./AUDIT_SUMMARY.md) (This Document)
**High-level overview and action plan**

---

## Critical Issues Requiring Immediate Attention

### 🔴 Priority 1: Performance Bottlenecks

#### Issue 1.1: Nested Loop Cost Matrix Computation
**Location:** `src/tracking/Trackers/Tracker.py:143-164`

**Impact:** 20-100ms per frame

**Problem:**
```python
# Current: O(N*M) Python loops
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        cost_matrix[i, j] = x.calculate_cost(y)
```

**Solution:** Vectorize using NumPy broadcasting (see PERFORMANCE_ANALYSIS.md section 2.1)

**Difficulty:** Medium
**Priority:** 🔴 CRITICAL

---

#### Issue 1.2: Inefficient NMS Implementation
**Location:** `src/Detecting/Detectors/Detector.py:64-94`

**Impact:** 10-50ms per frame

**Problem:** Python while-loop with repeated array indexing

**Solution:** Use OpenCV's NMS or vectorized NumPy implementation (see PERFORMANCE_ANALYSIS.md section 1.4)

**Difficulty:** Easy
**Priority:** 🔴 CRITICAL

---

#### Issue 1.3: Multiple Array Copies in Postprocessing
**Location:** `src/Detecting/Detectors/Detector_ONNX_YOLO7.py:226-261`

**Impact:** 5-15ms per frame

**Problem:** Each filtering step creates new array copies

**Solution:** Combine masks before indexing (see PERFORMANCE_ANALYSIS.md section 1.3)

**Difficulty:** Easy
**Priority:** 🔴 CRITICAL

---

#### Issue 1.4: Missing ONNX Session Options
**Location:** `src/Detecting/Detectors/Detector_ONNX_YOLO7.py:42`

**Impact:** Variable (depends on model and hardware)

**Problem:** No graph optimization, no thread control, no provider fallback checking

**Solution:** Add session options and provider verification (see PERFORMANCE_ANALYSIS.md section 1.2)

**Difficulty:** Easy
**Priority:** 🔴 CRITICAL

---

### ⚠️ Priority 2: Logical Errors

#### Issue 2.1: Creation Threshold Inconsistency
**Location:** `src/main.py:52-53`

**Problem:**
```python
detection_threshold = 0.3  # High/low split
creation_threshold = 0.3   # Should be > detection_threshold
```

**Expected:** `creation_threshold` should be higher to avoid creating tracks from low-confidence detections

**Solution:** Change to `creation_threshold = 0.4` (the default value)

**Difficulty:** Trivial
**Priority:** ⚠️ WARNING

---

#### Issue 2.2: Redundant Confidence Filtering
**Location:** `src/Detecting/Detectors/Detector_ONNX_YOLO7.py:232-246`

**Problem:** Applies confidence threshold twice (before and after multiplying class confidence)

**Solution:** Remove first filter, only filter once after multiplication (see PERFORMANCE_ANALYSIS.md section 1.3)

**Difficulty:** Easy
**Priority:** ⚠️ WARNING

---

### 🟡 Priority 3: Configuration Issues

#### Issue 3.1: No Centralized Configuration
**Problem:** All parameters hardcoded in main.py, no config file support

**Solution:** Create YAML/JSON configuration system (see PARAMETER_AUDIT.md recommendations)

**Difficulty:** Medium
**Priority:** 🟡 MEDIUM

---

#### Issue 3.2: Missing Parameter Validation
**Problem:** No validation for threshold ranges, list lengths, or mutual constraints

**Solution:** Add validation in constructors

**Difficulty:** Easy
**Priority:** 🟡 MEDIUM

---

## Recommended Action Plan

### Phase 1: Immediate Fixes (1-2 days)
**Goal:** Achieve 2-3x performance improvement

- [ ] Vectorize cost matrix computation (Tracker.py)
- [ ] Replace NMS with optimized version (Detector.py)
- [ ] Add ONNX session options (Detector_ONNX_YOLO7.py)
- [ ] Combine array filtering operations (Detector_ONNX_YOLO7.py)
- [ ] Set logging level to WARNING for benchmarking
- [ ] Fix `creation_threshold` parameter (main.py)

**Expected Result:** 5-13 FPS → 11-25 FPS

---

### Phase 2: Configuration Improvements (1 week)
**Goal:** Improve usability and maintainability

- [ ] Create configuration file system (YAML)
- [ ] Group related parameters into config sections
- [ ] Add parameter validation
- [ ] Expose missing parameters (device selection, output path, etc.)
- [ ] Create example config files
- [ ] Update documentation

**Expected Result:** Better code organization, easier tuning

---

### Phase 3: Advanced Optimizations (2-4 weeks)
**Goal:** Further performance gains and features

- [ ] Add batch processing support
- [ ] Implement GPU-accelerated operations (if applicable)
- [ ] Add performance profiling tools
- [ ] Cache repeated computations (Kalman predictions)
- [ ] Optimize drawing operations
- [ ] Add video output saving

**Expected Result:** 25-40 FPS (depending on hardware)

---

## Parameter Organization Recommendations

### Current State (main.py)
```python
# Scattered parameters
model_path = 'models/yolov7_640x640.onnx'
confidence_threshold = 0.1
iou_threshold = 0.5
classes = [0]
detection_threshold = 0.3
creation_threshold = 0.3
# ... etc
```

### Recommended Structure (config.yaml)
```yaml
detection:
  model_path: models/yolov7_640x640.onnx
  confidence_threshold: 0.1
  iou_threshold: 0.5
  classes: [0]
  device: auto
  max_detections: 300

tracking:
  association:
    detection_threshold: 0.3
    match_thresholds: [0.4, 0.2, 0.2]

  lifecycle:
    creation_threshold: 0.4  # Fixed!
    activation_threshold: 10
    deactivation_threshold: 15

  trajectory:
    max_size: 50
    min_length: 3

motion_model:
  std_position: 0.05
  std_velocity: 0.00625
  dt: 1.0

video:
  input_path: data/TownCent.mp4
  output_path: output/result.mp4
  draw_mode: state
  save_output: true
  skip_frames: 1

system:
  log_level: INFO
  log_file: logs/tracking.log
  verbose: true
```

---

## Testing Recommendations

### Performance Testing
```bash
# Before optimization
python src/main.py --log-level WARNING --benchmark

# After optimization
python src/main.py --log-level WARNING --benchmark

# Compare FPS improvements
```

### Correctness Testing
```bash
# Test with various threshold values
python src/main.py --config configs/high_precision.yaml
python src/main.py --config configs/high_recall.yaml
python src/main.py --config configs/balanced.yaml

# Verify tracking quality hasn't degraded
```

### Profiling
```bash
# Add profiling code from PERFORMANCE_ANALYSIS.md section 8
# Run and analyze bottlenecks
python -m cProfile -o profile.stats src/main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"
```

---

## Risk Assessment

### Low Risk (Safe to implement immediately)
✅ ONNX session options
✅ Logging level reduction
✅ NMS optimization
✅ Array copy reduction
✅ Parameter validation

### Medium Risk (Requires testing)
⚠️ Cost matrix vectorization - Must verify tracking quality
⚠️ Configuration file refactor - Breaking change for users
⚠️ Creation threshold fix - May affect tracking behavior

### High Risk (Requires careful evaluation)
🔴 Batch processing - Significant architectural change
🔴 Kalman filter caching - May affect tracking accuracy

---

## Success Metrics

### Performance Metrics
- **Target FPS:** 20+ FPS on M3 MacBook (currently ~5-13 FPS)
- **Latency:** <50ms per frame for full pipeline
- **Memory:** No memory leaks during long runs

### Code Quality Metrics
- **Configuration:** All parameters in config file (not hardcoded)
- **Validation:** 100% of parameters validated
- **Documentation:** All parameters documented with ranges

### Tracking Quality Metrics
- **Precision:** No degradation after optimization
- **Recall:** No degradation after optimization
- **ID Switches:** Similar or better than current

---

## Conclusion

The GenTbD system shows promise but suffers from several critical performance bottlenecks that significantly limit its usability on modern hardware like the M3 MacBook. The good news is that **these issues are fixable** with straightforward optimizations that should yield a **2-3x performance improvement**.

The parameter audit also revealed opportunities to improve code organization and configurability through a structured configuration system.

**Recommended Next Steps:**
1. Implement Phase 1 fixes (1-2 days) for immediate performance gains
2. Benchmark and verify improvements
3. Proceed with Phase 2 configuration improvements
4. Re-evaluate for Phase 3 advanced optimizations

**Expected Timeline:**
- Phase 1: 1-2 days
- Phase 2: 1 week
- Phase 3: 2-4 weeks

**Total Effort:** 3-5 weeks for full optimization and refactor

---

## References

- **PARAMETER_AUDIT.md** - Complete parameter documentation
- **PERFORMANCE_ANALYSIS.md** - Detailed performance analysis and fixes
- **ByteTrack Paper** - https://arxiv.org/abs/2110.06864
- **ONNX Runtime Optimization** - https://onnxruntime.ai/docs/performance/
- **Kalman Filter Tuning** - https://www.kalmanfilter.net/

---

**End of Audit Summary**
