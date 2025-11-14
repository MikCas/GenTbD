# GenTbD Performance Analysis

## Executive Summary

This document analyzes the detection and tracking pipeline for performance bottlenecks, particularly on Apple M3 hardware. Several **critical performance issues** have been identified that significantly slow down both CPU and GPU execution.

**Severity Levels:**
- 🔴 **CRITICAL**: Major performance bottleneck, must fix
- 🟡 **WARNING**: Moderate performance impact, should optimize
- 🟢 **INFO**: Minor optimization opportunity

---

## 1. DETECTION PIPELINE ANALYSIS

### Pipeline Flow
```
Frame → Preprocess → Inference → Postprocess → Create Detections
```

### 1.1 Preprocessing (Detector_ONNX_YOLO7.py:132-181)

**Current Implementation:**
```python
def preprocess(self, image: cv2.Mat) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    resized_image = cv2.resize(image_rgb, (0,0), fx=scale, fy=scale)  # Resize
    padded_image = cv2.copyMakeBorder(...)  # Padding
    blob = padded_image.transpose(2,0,1)[np.newaxis,...].astype(np.float32) / 255.0
```

**Performance Assessment:** ✅ **GOOD**
- Uses optimized OpenCV functions
- Minimal memory allocations
- Standard operations, difficult to optimize further

---

### 1.2 Inference (Detector_ONNX_YOLO7.py:182-204)

**Current Implementation:**
```python
def inference(self, blob: np.ndarray) -> list:
    start_time = time.perf_counter()
    outputs = self._session.run(self._output_names, {self._input_names[0]: blob})
    inference_time_ms = (time.perf_counter() - start_time) * 1000
```

**Issues Identified:**

🔴 **CRITICAL: Execution Provider Selection**
```python
# Line 42: Hardcoded provider list
self._session = ort.InferenceSession(
    model_path,
    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
)
```

**Problems:**
1. **CoreMLExecutionProvider may not be optimal** for M3 - Consider testing alternatives
2. **No provider priority checking** - May fall back to CPU without warning
3. **No session options** - Missing performance optimizations

**Fix:**
```python
# Recommended approach
import onnxruntime as ort

# Check available providers
available_providers = ort.get_available_providers()
print(f"Available providers: {available_providers}")

# Set session options for performance
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4  # Adjust based on CPU cores
session_options.inter_op_num_threads = 4

# Provider selection with options
providers = []
if 'CoreMLExecutionProvider' in available_providers:
    providers.append(('CoreMLExecutionProvider', {
        'device_id': 0,
    }))
providers.append('CPUExecutionProvider')

self._session = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=providers
)
```

🟡 **WARNING: No Batch Processing**
- Processing one frame at a time
- Could batch multiple frames for better GPU utilization
- Recommendation: Add batch processing option

---

### 1.3 Postprocessing (Detector_ONNX_YOLO7.py:205-262)

**Current Implementation:**
```python
def postprocess(self, outputs: list) -> tuple:
    predictions = outputs[0][0]  # Shape: [num_anchors, 5 + num_classes]

    # Step 1: Filter by object confidence
    object_confidence_mask = predictions[:, 4] > self._confidence_threshold
    predictions = predictions[object_confidence_mask]

    # Step 2: Multiply class confidence with bbox confidence
    confidence_scores = predictions[:, 4]
    predictions[:, 5:] *= confidence_scores[:, np.newaxis]

    # Step 3: Filter by class confidence
    scores = np.max(predictions[:, 5:], axis=1)
    class_confidence_mask = scores > self._confidence_threshold
    predictions = predictions[class_confidence_mask]
    scores = scores[class_confidence_mask]

    # Step 4: Filter by class IDs
    class_ids = np.argmax(predictions[:, 5:], axis=1).astype(np.int32)
    class_filter_mask = np.isin(class_ids, self._classes)
    predictions = predictions[class_filter_mask]
    scores = scores[class_filter_mask]
    class_ids = class_ids[class_filter_mask]

    # Step 5: Extract boxes and NMS
    boxes = self.extract_boxes(predictions)
    selected_indices = Detector.nms(boxes, scores, self._iou_threshold)

    return boxes[selected_indices], class_ids[selected_indices], scores[selected_indices]
```

**Issues Identified:**

🔴 **CRITICAL: Redundant Filtering** (Lines 232-254)
The code applies `confidence_threshold` **TWICE**:
1. **Step 1** (line 232): Filter by `object_confidence > threshold`
2. **Step 3** (line 244): Filter by `class_confidence > threshold` (which is already filtered!)

**Problem:**
- After Step 2, `predictions[:, 5:]` already contains `class_conf * object_conf`
- Step 1 filters out low `object_conf`, but Step 3 re-checks against the same threshold
- This is **logically correct** but creates confusion

**Optimization:**
```python
def postprocess(self, outputs: list) -> tuple:
    predictions = outputs[0][0]

    # Single-step filtering: Apply threshold to final confidence
    confidence_scores = predictions[:, 4]
    predictions[:, 5:] *= confidence_scores[:, np.newaxis]  # Multiply first

    scores = np.max(predictions[:, 5:], axis=1)
    valid_mask = scores > self._confidence_threshold  # Single filter

    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])

    predictions = predictions[valid_mask]
    scores = scores[valid_mask]

    # Filter by class
    class_ids = np.argmax(predictions[:, 5:], axis=1).astype(np.int32)
    class_mask = np.isin(class_ids, self._classes)

    if not np.any(class_mask):
        return np.array([]), np.array([]), np.array([])

    predictions = predictions[class_mask]
    scores = scores[class_mask]
    class_ids = class_ids[class_mask]

    # Extract boxes and NMS
    boxes = self.extract_boxes(predictions)
    selected_indices = Detector.nms(boxes, scores, self._iou_threshold)

    return boxes[selected_indices], class_ids[selected_indices], scores[selected_indices]
```

🔴 **CRITICAL: Multiple Array Copies**
Every filtering step creates a new array copy:
```python
predictions = predictions[mask]  # Copy 1
scores = scores[mask]            # Copy 2
class_ids = class_ids[mask]      # Copy 3
```

**Impact:** With 8400 anchors (typical for 640x640), this creates unnecessary memory allocations.

**Optimization:** Combine masks before indexing
```python
# Compute all masks first
confidence_mask = ...
class_mask = ...
combined_mask = confidence_mask & class_mask

# Apply once
predictions = predictions[combined_mask]
scores = scores[combined_mask]
class_ids = class_ids[combined_mask]
```

---

### 1.4 Non-Maximum Suppression (Detector.py:64-94)

🔴 **CRITICAL: Inefficient NMS Implementation**

**Current Implementation:**
```python
@staticmethod
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        current_box_index = sorted_indices[0]
        keep_boxes.append(current_box_index)

        # Compute IoU with ALL remaining boxes
        ious = Detector.compute_iou(boxes[current_box_index], boxes[sorted_indices[1:]])

        remaining_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[remaining_indices + 1]

    return keep_boxes
```

**Problems:**
1. **Python while loop** - Slow for large numbers of detections
2. **Repeated array indexing** - `sorted_indices[1:]` creates new array each iteration
3. **Custom IoU computation** - Slower than optimized implementations

**Performance Impact:**
- With 100 detections: ~50-100 iterations
- Each iteration: Array copy + IoU computation
- **Estimated overhead: 10-50ms per frame**

🔴 **CRITICAL FIX: Use OpenCV or Torchvision NMS**
```python
# Option 1: OpenCV (if available in your version)
import cv2
indices = cv2.dnn.NMSBoxes(
    boxes.tolist(),
    scores.tolist(),
    score_threshold=0.0,  # Already filtered
    nms_threshold=iou_threshold
)
keep_boxes = indices.flatten() if len(indices) > 0 else []

# Option 2: Vectorized NumPy implementation (faster than current)
def nms_fast(boxes, scores, iou_threshold):
    """Fast vectorized NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Vectorized IoU computation
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# This is still not optimal - consider using:
# - torchvision.ops.nms (if PyTorch available)
# - onnxruntime NMS op
# - TensorRT NMS (for NVIDIA GPUs)
```

**Expected Speedup: 2-5x faster**

---

## 2. TRACKING PIPELINE ANALYSIS

### Pipeline Flow
```
Detections → Update → Association → Track Management → Draw
```

### 2.1 Cost Matrix Computation (Tracker.py:143-164)

🔴 **CRITICAL: Nested Loop Cost Matrix**

**Current Implementation:**
```python
def create_cost_matrix(self, xs: List[Detection], ys: List[Detection]) -> np.ndarray:
    num_xs = len(xs)
    num_ys = len(ys)
    cost_matrix = np.ones((num_xs, num_ys))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cost_matrix[i, j] = x.calculate_cost(y)  # Calls Kalman filter + IoU

    return cost_matrix
```

**Problems:**
1. **Double nested loop in Python** - Very slow
2. **No vectorization** - Processes one pair at a time
3. **Repeated Kalman predictions** - Each track predicts state multiple times

**Performance Impact:**
- With 50 tracks and 100 detections: 5,000 iterations
- Each iteration: Kalman predict + IoU computation
- **Estimated overhead: 20-100ms per frame**

🔴 **CRITICAL FIX: Vectorize Cost Matrix**
```python
def create_cost_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Vectorized cost matrix computation."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.ones((len(tracks), len(detections)))

    # Get all predicted boxes at once
    track_boxes = np.array([
        track.get_kalman_filter_state().bounding_box.xyxy()
        for track in tracks
    ])  # Shape: (num_tracks, 4)

    detection_boxes = np.array([
        det.bounding_box.xyxy()
        for det in detections
    ])  # Shape: (num_detections, 4)

    # Vectorized IoU computation
    cost_matrix = 1.0 - self._compute_iou_matrix(track_boxes, detection_boxes)

    return cost_matrix

def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes using broadcasting."""
    # boxes1: (N, 4), boxes2: (M, 4) -> output: (N, M)

    # Expand dimensions for broadcasting
    boxes1 = boxes1[:, None, :]  # (N, 1, 4)
    boxes2 = boxes2[None, :, :]  # (1, M, 4)

    # Compute intersection
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute union
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection

    iou = intersection / np.maximum(union, 1e-6)

    return iou.squeeze()
```

**Expected Speedup: 10-50x faster**

---

### 2.2 Track State Prediction (Track.py:93-109)

🟡 **WARNING: Redundant Kalman Predictions**

**Current Issue:**
In `create_cost_matrix`, each track calls `get_kalman_filter_state()`, which may call `predict()`:
```python
def calculate_cost(self, detection: Detection) -> float:
    predicted_detection = self.get_kalman_filter_state()  # Calls predict internally
    similarity = detection.calculate_similarity(predicted_detection)
    return 1 - similarity
```

If `predict()` is called in the Kalman filter during `get_kalman_filter_state()`, this happens for **every detection pair**.

**Fix:** Ensure predictions are cached or called once per track per frame.

---

### 2.3 Cascaded Association (SimpleTracker.py:204-253)

🟡 **WARNING: Multiple Association Passes**

**Current Implementation:**
```python
def bytetrack_association(self, timestep, detections):
    # Split detections
    detections_high, detections_low = self.partition_detections(detections)

    # Pass 1: High-conf detections with matched+lost tracks
    partition1 = self.association(timestep, matched_lost_tracks, detections_high, threshold1)

    # Pass 2: Low-conf detections with unmatched tracks
    partition2 = self.association(timestep, unmatched_tracks1, detections_low, threshold2)

    # Pass 3: Unmatched high-conf detections with new tracks
    partition3 = self.association(timestep, new_tracks, unmatched_high, threshold3)

    # Combine results
    ...
```

**Performance Assessment:** ⚠️ **ACCEPTABLE BUT NOT OPTIMAL**
- This is the **ByteTrack algorithm design** - necessary for good tracking
- However, each association call:
  1. Creates cost matrix (nested loops) 🔴
  2. Runs Hungarian algorithm
  3. Processes results

**Optimization:** Focus on speeding up cost matrix computation (see 2.1)

---

## 3. LOGGING OVERHEAD

🟡 **WARNING: Excessive Logging**

**Current Implementation:**
```python
# main.py:16
logger.setLevel(logging.DEBUG)

# Many DEBUG logs throughout:
self.log(logging.INFO, "\t|| PREPROCESSING")
self.log(logging.INFO, "\t|| INFERENCE")
self.log(logging.INFO, "\t|| POSTPROCESS")
# ... etc
```

**Impact:**
- Logging on every frame adds overhead
- String formatting for debug messages
- Console I/O is slow

**Fix:**
```python
# For production/benchmarking:
logger.setLevel(logging.WARNING)  # or ERROR

# Only log important events, not every frame
```

**Expected Speedup: 5-15% in total pipeline**

---

## 4. SUMMARY OF CRITICAL BOTTLENECKS

### Ranked by Performance Impact

| Priority | Issue | Location | Estimated Impact | Fix Complexity |
|----------|-------|----------|------------------|----------------|
| 🔴 **1** | Nested loop cost matrix | Tracker.py:143-164 | **20-100ms/frame** | Medium |
| 🔴 **2** | Inefficient NMS loop | Detector.py:64-94 | **10-50ms/frame** | Easy |
| 🔴 **3** | Multiple array copies in postprocess | Detector_ONNX_YOLO7.py:226-261 | **5-15ms/frame** | Easy |
| 🔴 **4** | Missing ONNX session options | Detector_ONNX_YOLO7.py:42 | **Variable** | Easy |
| 🟡 **5** | Redundant confidence filtering | Detector_ONNX_YOLO7.py:232-246 | **2-5ms/frame** | Easy |
| 🟡 **6** | Excessive logging | Throughout | **5-10%** | Trivial |
| 🟡 **7** | No batch processing | Detector_ONNX_YOLO7.py:182 | **Depends on use case** | Hard |

---

## 5. LOGICAL ERRORS FOUND

### Error 1: Incorrect Confidence Threshold Logic ❌

**Location:** Detector_ONNX_YOLO7.py:232-246

**Issue:**
The two-stage filtering may not achieve the intended behavior:
```python
# Step 1: Filter by object confidence
object_confidence_mask = predictions[:, 4] > self._confidence_threshold
predictions = predictions[object_confidence_mask]

# Step 2: Multiply confidences
predictions[:, 5:] *= confidence_scores[:, np.newaxis]

# Step 3: Filter AGAIN by class confidence
scores = np.max(predictions[:, 5:], axis=1)
class_confidence_mask = scores > self._confidence_threshold
```

**Problem:** After multiplication in Step 2, `scores` will be **lower** than `confidence_scores` (since class probs ≤ 1). So Step 3 may filter out boxes that passed Step 1.

**Correct Behavior:** This is actually correct for YOLO! The final confidence should be `obj_conf * class_conf`. But Step 1 is unnecessary if Step 3 will filter anyway.

**Fix:** Remove Step 1, only filter once after multiplication.

---

### Error 2: Creation Threshold < Detection Threshold ⚠️

**Location:** main.py:52-53

```python
detection_threshold = 0.3
creation_threshold = 0.3  # Default was 0.4!
```

**Issue:**
- Detections with confidence ≥ 0.3 are "high confidence"
- But new tracks can be created with confidence ≥ 0.3
- This means **all high-confidence detections** can create tracks

**Expected Behavior:** Typically, `creation_threshold > detection_threshold` to avoid creating tracks from marginally-confident detections.

**Recommendation:** Set `creation_threshold = 0.4` or higher.

---

### Error 3: Activation Threshold Terminology Confusion 🤔

**Location:** SimpleTracker.py:117

```python
if (track._trajectory.size > self._activation_threshold):
    track.activation(timestep, detection)
```

**Issue:**
- Parameter is called `activation_threshold` (value: 10)
- But it's compared with `trajectory.size` (number of detections)
- Should be called `activation_count` or `min_trajectory_size` for clarity

**Not a bug, but confusing naming.**

---

## 6. RECOMMENDED FIXES - PRIORITY ORDER

### Immediate (Day 1):
1. ✅ **Vectorize cost matrix computation** (Tracker.py)
2. ✅ **Replace NMS with optimized version** (Detector.py)
3. ✅ **Add ONNX session options** (Detector_ONNX_YOLO7.py)
4. ✅ **Reduce logging to WARNING level** for performance testing

### Short-term (Week 1):
5. ✅ **Combine array filtering operations** (Detector_ONNX_YOLO7.py)
6. ✅ **Remove redundant confidence filtering** (Detector_ONNX_YOLO7.py)
7. ✅ **Fix creation_threshold parameter** (main.py)
8. ✅ **Add performance profiling** to measure actual bottlenecks

### Long-term (Month 1):
9. ⏰ **Add batch processing support**
10. ⏰ **Implement GPU-accelerated NMS** (if using CUDA)
11. ⏰ **Profile and optimize Kalman filter** operations
12. ⏰ **Add caching for repeated computations**

---

## 7. EXPECTED PERFORMANCE IMPROVEMENT

### Current Performance (Estimated):
- **Preprocessing:** ~5ms
- **Inference:** ~20-50ms (model-dependent)
- **Postprocessing:** ~15-30ms 🔴
- **Tracking:** ~30-100ms 🔴
- **Drawing:** ~5-10ms
- **Total:** ~75-200ms per frame → **5-13 FPS**

### After Optimization (Estimated):
- **Preprocessing:** ~5ms
- **Inference:** ~20-50ms (same)
- **Postprocessing:** ~5-10ms ✅ (3x faster)
- **Tracking:** ~5-15ms ✅ (6-10x faster)
- **Drawing:** ~5-10ms
- **Total:** ~40-90ms per frame → **11-25 FPS**

### Expected Speedup: **2-3x overall** 🚀

---

## 8. PERFORMANCE PROFILING RECOMMENDATIONS

Add this to measure actual bottlenecks:

```python
import time
import numpy as np

class PerformanceProfiler:
    def __init__(self):
        self.times = {}

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        pass

    def record(self, name):
        if name not in self.times:
            self.times[name] = []
        elapsed = (time.perf_counter() - self.start) * 1000
        self.times[name].append(elapsed)
        self.start = time.perf_counter()

    def report(self):
        print("\n=== Performance Report ===")
        for name, times in self.times.items():
            print(f"{name:30s}: {np.mean(times):6.2f}ms ± {np.std(times):5.2f}ms")

# Usage:
profiler = PerformanceProfiler()
with profiler:
    blob = detector.preprocess(image)
    profiler.record("Preprocess")

    output = detector.inference(blob)
    profiler.record("Inference")

    data = detector.postprocess(output)
    profiler.record("Postprocess")

    detections = detector.create_detections(data)
    profiler.record("Create Detections")

    tracker.update(timestep, detections)
    profiler.record("Tracking")

profiler.report()
```

---

## CONCLUSION

The GenTbD system has **several critical performance bottlenecks** that significantly impact performance on M3 hardware:

1. **Non-vectorized cost matrix computation** - Major bottleneck
2. **Inefficient NMS implementation** - Major bottleneck
3. **Redundant array operations** - Moderate bottleneck
4. **Missing ONNX optimizations** - Variable impact

By implementing the recommended fixes, you should see a **2-3x performance improvement**, bringing the system from ~5-13 FPS to ~11-25 FPS on an M3 MacBook.

The fixes are relatively straightforward and can be implemented incrementally, with immediate benefits from vectorizing the cost matrix and replacing the NMS implementation.
