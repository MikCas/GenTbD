# Code Review - Performance Analysis

## Critical Issues

### 1. **Class Filtering Bug in ObjectDetector** (CRITICAL)
**File**: `src/detecting/detectors/object_detector.py:126-128`

```python
if self.classes is not None:
    class_mask = torch.isin(labels, torch.tensor(self.classes))
    # BUG: class_mask is created but never applied!
```

**Issue**: Class filtering is broken - the mask is created but not used to filter boxes/labels/scores.

**Fix Required**:
```python
if self.classes is not None:
    class_mask = torch.isin(labels, torch.tensor(self.classes))
    boxes = boxes[class_mask]
    labels = labels[class_mask]
    scores = scores[class_mask]
```

**Impact**: All detections pass through regardless of `--classes` flag.

---

### 2. **Unnecessary CPU Transfers in Postprocess** (PERFORMANCE)
**File**: `src/detecting/detectors/object_detector.py:107-109`

```python
boxes = output['boxes'].cpu()
labels = output['labels'].cpu()
scores = output['scores'].cpu()
```

**Issue**: Transfers tensors to CPU for filtering, then immediately uses them for tensor ops.

**Impact**:
- For MPS: ~5-10ms overhead per frame
- For CUDA: ~2-5ms overhead per frame

**Better Approach**:
- Keep tensors on device for filtering
- Only call `.cpu()` when converting to numpy/Python types at the end

---

### 3. **No Batch Processing Support** (PERFORMANCE)
**Current**: Processes one frame at a time
**Impact**:
- GPU sits idle during preprocessing
- Batch size of 4-8 could give 2-3x throughput on MPS/CUDA
- Especially beneficial for video processing

---

### 4. **Frame Skipping Creates Detection Lag** (FUNCTIONAL)
**File**: `src/video_processor.py:197-213`

**Issue**: When `skip_frames > 1`, cached detections don't account for object motion.

**Impact**: Bounding boxes lag behind fast-moving objects.

**Potential Fix**: Implement motion prediction or track-based interpolation.

---

## Performance Optimizations

### Already Implemented ✓
1. **Removed Frame wrapper overhead** - Direct numpy array processing
2. **MPS warmup** - Shader compilation handling
3. **Internal resize fix** - Prevents 800px upscaling
4. **Lazy evaluation** - Minimal property access
5. **Deque for FPS tracking** - Prevents unbounded growth

### Quick Wins (Not Yet Implemented)

#### 1. **Keep Tensors on Device** (5-10ms improvement)
```python
# Current: CPU filtering
boxes = output['boxes'].cpu()
scores = output['scores'].cpu()
mask = scores >= self.conf_threshold

# Better: GPU filtering
mask = output['scores'] >= self.conf_threshold
boxes = output['boxes'][mask].cpu()  # Only transfer filtered results
```

#### 2. **TorchScript Compilation** (1.2-1.5x speedup)
```python
self.model = torch.jit.script(self.model)
```

#### 3. **Tensor Pinning for CPU→GPU Transfer** (Minor improvement)
```python
input_tensor = input_tensor.pin_memory()
```

#### 4. **Half Precision (FP16) on MPS** (Up to 2x faster)
```python
if device == 'mps':
    self.model = self.model.half()
```

### Long-term Optimizations

1. **Batch Processing**: Process 4-8 frames at once
2. **Async Pipeline**: Overlap decode + preprocess + inference
3. **ONNX Runtime**: 2-3x speedup, better cross-platform
4. **Quantization**: INT8 for 2-4x speedup
5. **TensorRT**: 5-10x on NVIDIA GPUs

---

## Architecture Strengths

1. **Clean Separation**: Detector ABC enforces consistent interface
2. **Factory Pattern**: Easy to add new detectors
3. **Property Classes**: Immutable, well-designed (BoundingBox, Keypoints)
4. **Config Hierarchy**: YAML + CLI override works well
5. **Minimal State**: VideoProcessor doesn't accumulate unnecessary data

---

## Detector Comparison (Expected Performance)

| Detector | Model | Params | CPU FPS | MPS FPS | CUDA FPS | Use Case |
|----------|-------|--------|---------|---------|----------|----------|
| ObjectDetector | mobilenet | 5.5M | 5-6 | 20-30 | 60-100 | Fast object detection |
| ObjectDetector | resnet50 | 44M | 0.5-2 | 8-15 | 30-60 | Accurate object detection |
| ObjectDetector | retinanet | 34M | 1-3 | 10-20 | 40-80 | Balanced detection |
| KeypointDetector | resnet50 | 46M | 0.3-1 | 5-10 | 20-40 | Human pose estimation |
| ReIDDetector | osnet_x1_0 | 2.2M | 50-100 | 200-400 | 500-1000 | Person re-identification |

**Notes**:
- FPS assumes 640x360 resolution with internal resize disabled
- ReID works on crops, not full frames (much faster)
- MPS includes 30-60s first-run warmup

---

## Testing Gaps

1. **No performance regression tests** - Need pytest-benchmark integration
2. **No device comparison tests** - Should verify CPU/MPS/CUDA parity
3. **No integration tests** - End-to-end pipeline testing needed
4. **No stress tests** - Long videos, memory leaks?

---

## Recommendations (Priority Order)

### Immediate (< 30 min)
1. **Fix class filtering bug** - 3 lines of code
2. **Keep tensors on device during filtering** - 10 lines

### Short-term (1-2 hours)
3. **Create comprehensive benchmark suite** - Compare all detectors × devices
4. **Add TorchScript compilation option** - Config flag + 5 lines
5. **Document expected performance** - Update README with benchmarks

### Long-term (1-2 days)
6. **ONNX Runtime integration** - New detector type
7. **Batch processing support** - Refactor pipeline
8. **YOLOv8 detector** - Faster alternative to FasterRCNN

---

## Next Steps

1. Fix critical class filtering bug
2. Create comprehensive benchmarking script comparing:
   - All detector types (object, keypoint, reid)
   - All models (mobilenet, resnet50, retinanet)
   - All devices (cpu, mps, cuda if available)
   - Multiple resolutions (640, 480, 320)
3. Generate performance matrix for documentation
