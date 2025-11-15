# Performance Fixes for Main App

## Issue: MPS works (43ms) but main app is slow

### Root Causes

1. **Frame object overhead** - Created every iteration
2. **cv2.resize creates copy** - Needed for scaling but adds overhead
3. **cv2.imshow() blocking** - Display synchronization
4. **Unnecessary Frame wrapper** for detection

---

## Fix #1: Skip Frame Wrapper for Detection (HIGHEST IMPACT)

**File**: `src/video_processor.py:214-221`

**Before:**
```python
# Scale frame for detection if requested (Frame handles scaling)
if self.max_dimension and max(frame.height, frame.width) > self.max_dimension:
    scale = self.max_dimension / max(frame.height, frame.width)
    detection_frame = frame.scaled(scale)  # ❌ Creates Frame object + copy
else:
    detection_frame = frame

# Run detection on frame data
detections = self.detector.detect(detection_frame.data)
```

**After:**
```python
# Resize frame directly without Frame wrapper
if self.max_dimension and max(frame_data.shape[0], frame_data.shape[1]) > self.max_dimension:
    h, w = frame_data.shape[:2]
    scale = self.max_dimension / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    detection_data = cv2.resize(frame_data, (new_w, new_h))  # ✅ Direct resize
    scale_factor = scale
else:
    detection_data = frame_data  # ✅ No copy
    scale_factor = 1.0

# Run detection
detections = self.detector.detect(detection_data)
```

**Expected Speedup**: 2-3x (eliminates Frame object + metadata overhead)

---

## Fix #2: Remove Frame Object from Main Loop

**File**: `src/video_processor.py:158-166`

**Before:**
```python
frame = Frame(
    data=frame_data,
    frame_id=frame_count,
    timestamp=timestamp,
    source_id=str(self.source)
)

self.frame_num = frame_count + 1
should_detect = frame_count % self.skip_frames == 0
```

**After:**
```python
# Just use raw frame data - no Frame object needed
self.frame_num = frame_count + 1
should_detect = frame_count % self.skip_frames == 0

# Calculate timestamp only if needed
if self.video_fps and self.video_fps > 0:
    timestamp = (frame_count + 1) / self.video_fps
else:
    timestamp = (frame_count + 1) * (1.0 / 30.0)
```

**Expected Speedup**: 1.5-2x

---

## Fix #3: Update _detect_objects for Direct Data

**File**: `src/video_processor.py:200-228`

**Before:**
```python
def _detect_objects(self, frame: Frame):
    """Run object detection on frame."""
    start_time = time.time()

    if self.max_dimension and max(frame.height, frame.width) > self.max_dimension:
        scale = self.max_dimension / max(frame.height, frame.width)
        detection_frame = frame.scaled(scale)
    else:
        detection_frame = frame

    detections = self.detector.detect(detection_frame.data)

    if detection_frame.scale_factor != 1.0:
        detections = self._scale_detections(detections, detection_frame.scale_factor)

    elapsed = time.time() - start_time
    return detections, elapsed
```

**After:**
```python
def _detect_objects(self, frame_data: np.ndarray):
    """Run object detection on frame.

    Args:
        frame_data: Raw frame data (H, W, C) BGR numpy array

    Returns:
        Tuple of (detections, elapsed_time)
    """
    start_time = time.time()

    # Resize if needed
    if self.max_dimension:
        h, w = frame_data.shape[:2]
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            detection_data = cv2.resize(frame_data, (new_w, new_h))
            scale_factor = scale
        else:
            detection_data = frame_data
            scale_factor = 1.0
    else:
        detection_data = frame_data
        scale_factor = 1.0

    # Run detection
    detections = self.detector.detect(detection_data)

    # Scale detections back if needed
    if scale_factor != 1.0:
        detections = self._scale_detections(detections, scale_factor)

    elapsed = time.time() - start_time
    return detections, elapsed
```

---

## Fix #4: Update Main Loop to Pass Raw Data

**File**: `src/video_processor.py:168-176`

**Before:**
```python
if should_detect:
    # Run detection and render
    detections, elapsed = self._detect_objects(frame)
    self.fps_samples.append(1.0 / elapsed if elapsed > 0 else 0)

    display_frame = frame.data
    self._render_frame(display_frame, detections)
```

**After:**
```python
if should_detect:
    # Run detection and render
    detections, elapsed = self._detect_objects(frame_data)
    self.fps_samples.append(1.0 / elapsed if elapsed > 0 else 0)

    # Render directly on frame data
    display_frame = frame_data
    self._render_frame(display_frame, detections)
```

---

## Fix #5: Non-blocking Display (Optional)

**File**: `src/video_processor.py:340-342`

**Before:**
```python
def _handle_input(self, frame) -> bool:
    wait_time = 1 if self.continuous_mode else 0
    key = cv2.waitKey(wait_time) & 0xFF
```

**After:**
```python
def _handle_input(self, frame) -> bool:
    # Always non-blocking for performance
    key = cv2.waitKey(1) & 0xFF

    # Slow down if not continuous (sleep instead of blocking waitKey)
    if not self.continuous_mode:
        time.sleep(0.03)  # ~30 FPS max in step mode
```

---

## Combined Performance Impact

| Fix | Speedup | Difficulty |
|-----|---------|-----------|
| Skip Frame wrapper | 2-3x | Easy |
| Remove Frame objects | 1.5-2x | Easy |
| Non-blocking display | 1.2x | Easy |
| **Total** | **4-7x** | **30 min** |

---

## Expected Results After Fixes

**Current (with MPS):**
- Quick test: 43ms (23 FPS) ✓
- Main app: ~200-500ms (2-5 FPS) ✗

**After fixes:**
- Main app: ~50-80ms (12-20 FPS) ✓

---

## Implementation Order

1. ✅ Fix #3 - Update _detect_objects signature
2. ✅ Fix #4 - Update main loop to pass raw data
3. ✅ Fix #2 - Remove Frame object creation
4. (Optional) Fix #5 - Non-blocking display

**Test after each change** to measure impact.
