# Current Parameters - Main Branch

**Generated:** 2025-11-14
**Branch:** main
**Source:** src/main.py

---

## All Parameters (Current Values)

### DETECTOR (Lines 38-49)
```python
model_path = 'models/yolov7_640x640.onnx'
confidence_threshold = 0.1
iou_threshold = 0.5
classes = [0]
```

### TRACKER (Lines 52-65)
```python
detection_threshold = 0.3
creation_threshold = 0.3
match_thresholds = [0.4, 0.2, 0.2]
activation_threshold = 10
deactivation_threshold = 15
```

### VIDEO PROCESSOR (Lines 68-76)
```python
video_file = 'TownCent.mp4'
video_path = os.path.join(os.getcwd(), 'data', video_file)
draw_mode = 'state'  # Options: 'state', 'id', 'none'
```

### LOGGER (Lines 8-21)
```python
log_level = logging.DEBUG
log_format = '%(asctime)s - %(levelname)s - %(message)s'
```

---

## Parameter Groupings

### Group 1: Detection Quality
- `confidence_threshold = 0.1` - Minimum detection confidence
- `iou_threshold = 0.5` - NMS overlap threshold
- `classes = [0]` - Detected classes (person only)

### Group 2: Track Association
- `detection_threshold = 0.3` - High/low confidence split
- `match_thresholds = [0.4, 0.2, 0.2]` - Cascaded matching thresholds

### Group 3: Track Lifecycle
- `creation_threshold = 0.3` - Create new track threshold
- `activation_threshold = 10` - Frames to activate track
- `deactivation_threshold = 15` - Frames to remove lost track

### Group 4: Input/Output
- `model_path = 'models/yolov7_640x640.onnx'` - Detector model
- `video_file = 'TownCent.mp4'` - Input video
- `draw_mode = 'state'` - Visualization mode

### Group 5: System
- `log_level = logging.DEBUG` - Logging verbosity

---

## Parameters Not in main.py (Hardcoded)

### Track.py
```python
TRAJECTORY_MAX_SIZE = 50  # Line 36
```

### KalmanFilter.py
```python
stdPosition = 1.0 / 20      # 0.05, Line 22
stdVelocity = 1.0 / 160     # 0.00625, Line 22
dt = 1.0                     # Line 22
```

### Detector_ONNX_YOLO7.py
```python
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']  # Line 42
padding_color = (114, 114, 114)  # Line 170
```

### Track.py (Draw colors)
```python
TrackState.NEW = (255, 0, 0)        # Blue, Line 256
TrackState.MATCHED = (0, 255, 0)    # Green, Line 257
TrackState.LOST = (0, 0, 255)       # Red, Line 258
TrackState.RESERVED = (255, 255, 0) # Cyan, Line 259
```

---

## Summary Statistics

- **Total Parameters in main.py:** 13
- **Total Hardcoded Parameters:** 11+
- **Total System Parameters:** 24+

**Components:**
- Detector: 4 parameters
- Tracker: 5 parameters
- Video Processor: 2 parameters
- Logger: 2 parameters
- Hidden: 11+ parameters

---

## Recommended Changes

1. ⚠️ **creation_threshold**: Change from `0.3` to `0.4` (inconsistency with default)
2. 💡 **log_level**: Change to `logging.WARNING` for performance testing
3. 💡 **Expose hardcoded parameters** for better configurability

See **PARAMETER_AUDIT.md** and **PARAMETERS_REFERENCE.md** for details.
