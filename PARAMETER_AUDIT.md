# GenTbD Parameter Audit

This document provides a comprehensive audit of all parameters used in the GenTbD tracking-by-detection system, organized by component.

---

## 1. DETECTOR PARAMETERS (YOLOv7ONNX)

### Current Parameters
| Parameter | Type | Default | Current (main.py) | Description | Location |
|-----------|------|---------|-------------------|-------------|----------|
| `model_path` | str | - | `'models/yolov7_640x640.onnx'` | Path to ONNX model file | main.py:39 |
| `confidence_threshold` | float | 0.1 | 0.1 | Minimum confidence for accepting detections | main.py:40 |
| `iou_threshold` | float | 0.5 | 0.5 | IoU threshold for NMS suppression | main.py:41 |
| `classes` | list[int] | - | `[0]` | List of class IDs to detect (0=person) | main.py:42 |

### Internal Parameters (not exposed)
| Parameter | Value | Description | Location |
|-----------|-------|-------------|----------|
| `padding_color` | `(114, 114, 114)` | Gray padding color for preprocessing | Detector_ONNX_YOLO7.py:170 |

### Potential Missing Parameters
- **`input_size`**: Currently hardcoded from model, could be explicit (640x640)
- **`max_detections`**: Limit on maximum number of detections per frame
- **`nms_method`**: Choice between different NMS algorithms (standard, soft-NMS, etc.)
- **`device`**: Explicit device selection (CPU, CoreML, CUDA, etc.) - currently hardcoded
- **`batch_size`**: For batch processing multiple frames

---

## 2. TRACKER PARAMETERS (SimpleTracker)

### Current Parameters
| Parameter | Type | Default | Current (main.py) | Description | Location |
|-----------|------|---------|-------------------|-------------|----------|
| `detection_threshold` | float | 0.3 | 0.3 | Threshold to partition high/low confidence detections | main.py:52 |
| `creation_threshold` | float | 0.4 | 0.3 | Minimum confidence score to create new track | main.py:53 |
| `match_thresholds` | list[float] | `[0.2]` | `[0.4, 0.2, 0.2]` | Cascaded matching thresholds for association stages | main.py:54 |
| `activation_threshold` | int | 5 | 10 | Number of consecutive matches before track activation | main.py:55 |
| `deactivation_threshold` | int | 10 | 15 | Number of consecutive misses before track deactivation | main.py:56 |

### Parameter Grouping Analysis
These parameters control the **track lifecycle**:
- `creation_threshold` - Controls NEW track creation
- `activation_threshold` - Controls NEW → MATCHED transition
- `deactivation_threshold` - Controls LOST → RESERVED transition

These parameters control the **association strategy**:
- `detection_threshold` - Partitions detections into high/low confidence
- `match_thresholds[0]` - Matches high-conf detections with MATCHED+LOST tracks
- `match_thresholds[1]` - Matches low-conf detections with unmatched MATCHED+LOST tracks
- `match_thresholds[2]` - Matches unmatched high-conf detections with NEW tracks

### Potential Issues
⚠️ **INCONSISTENCY**: `creation_threshold` (0.3) < default (0.4), but `detection_threshold` (0.3) = `creation_threshold`. This means:
- Detections with conf=0.3 are in the high-confidence group
- These can create new tracks (conf ≥ 0.3)
- Consider: Should `creation_threshold` be higher than `detection_threshold`?

### Potential Missing Parameters
- **`min_trajectory_length`**: Minimum trajectory length for valid track
- **`max_lost_time`**: Maximum time a track can be lost (currently only frame count)
- **`occlusion_handling`**: Flag to enable/disable occlusion reasoning
- **`reid_threshold`**: Threshold for re-identification (if using appearance features)
- **`spatial_distance_weight`**: Weight for spatial distance in matching cost

---

## 3. TRACK PARAMETERS

### Current Parameters
| Parameter | Type | Default | Description | Location |
|-----------|------|---------|-------------|----------|
| `TRAJECTORY_MAX_SIZE` | int | 50 | Maximum number of detections stored in trajectory | Track.py:36 |

### Potential Missing Parameters
- **`smooth_trajectory`**: Flag to enable trajectory smoothing
- **`interpolate_missing`**: Flag to interpolate missing detections
- **`min_box_area`**: Minimum bounding box area for valid detection
- **`max_box_area`**: Maximum bounding box area for valid detection

---

## 4. KALMAN FILTER PARAMETERS

### Current Parameters
| Parameter | Type | Default | Description | Location |
|-----------|------|---------|-------------|----------|
| `stdPosition` | float | 1.0/20 = 0.05 | Standard deviation for position noise | KalmanFilter.py:22 |
| `stdVelocity` | float | 1.0/160 = 0.00625 | Standard deviation for velocity noise | KalmanFilter.py:22 |
| `dt` | float | 1.0 | Time step for motion model | KalmanFilter.py:22 |

### Parameter Analysis
These parameters control the **motion model uncertainty**:
- `stdPosition` affects how much the filter trusts position measurements
- `stdVelocity` affects how much the filter expects velocity to change
- Lower values = more trust in the model, higher values = more trust in measurements

### Potential Missing Parameters
- **`process_noise_scale`**: Global scale factor for process noise
- **`measurement_noise_scale`**: Global scale factor for measurement noise
- **`initial_velocity`**: Initial velocity estimate (currently 0)
- **`motion_model`**: Choice of motion model (constant velocity, constant acceleration, etc.)

---

## 5. VIDEO PROCESSOR PARAMETERS

### Current Parameters
| Parameter | Type | Default | Current (main.py) | Description | Location |
|-----------|------|---------|-------------------|-------------|----------|
| `video_path` | str | - | `'data/TownCent.mp4'` | Path to input video file | main.py:68-69 |
| `draw_mode` | str | `'state'` | `'state'` | Visualization mode ('state', 'id', 'none') | main.py:70 |

### Potential Missing Parameters
- **`output_path`**: Path to save processed video
- **`display_fps`**: Whether to display FPS on frame
- **`skip_frames`**: Process every Nth frame only
- **`resize_display`**: Resize factor for display window
- **`save_output`**: Flag to save processed video
- **`roi`**: Region of interest for processing

---

## 6. LOGGING PARAMETERS

### Current Parameters
| Parameter | Type | Default | Description | Location |
|-----------|------|---------|-------------|----------|
| `log_level` | int | `logging.DEBUG` | Logging level | main.py:16 |

### Potential Missing Parameters
- **`log_file`**: Path to save log file (currently console only)
- **`log_format`**: Custom log format
- **`verbose`**: Simple verbose flag (True/False)

---

## PARAMETER GROUPINGS - RECOMMENDED STRUCTURE

### Suggested Parameter Organization

```python
# 1. DETECTION PARAMETERS GROUP
detection_params = {
    'model_path': 'models/yolov7_640x640.onnx',
    'confidence_threshold': 0.1,
    'iou_threshold': 0.5,
    'classes': [0],
    'device': 'auto',  # NEW
    'max_detections': 300,  # NEW
}

# 2. TRACKING PARAMETERS GROUP
tracking_params = {
    # Association parameters
    'detection_threshold': 0.3,
    'match_thresholds': [0.4, 0.2, 0.2],

    # Lifecycle parameters
    'creation_threshold': 0.4,  # SUGGESTION: Increase from 0.3
    'activation_threshold': 10,
    'deactivation_threshold': 15,

    # Track management
    'max_trajectory_size': 50,
    'min_trajectory_length': 3,  # NEW
}

# 3. MOTION MODEL PARAMETERS GROUP
motion_params = {
    'std_position': 1.0/20,
    'std_velocity': 1.0/160,
    'dt': 1.0,
}

# 4. VIDEO PROCESSING PARAMETERS GROUP
video_params = {
    'video_path': 'data/TownCent.mp4',
    'draw_mode': 'state',
    'save_output': False,  # NEW
    'output_path': None,  # NEW
    'skip_frames': 1,  # NEW
}

# 5. SYSTEM PARAMETERS GROUP
system_params = {
    'log_level': logging.DEBUG,
    'log_file': None,
    'verbose': True,
}
```

---

## CRITICAL FINDINGS

### 1. Parameter Inconsistencies
- ⚠️ `creation_threshold` (0.3) vs default (0.4) - may create too many low-confidence tracks
- ⚠️ `creation_threshold` = `detection_threshold` - questionable design choice

### 2. Missing Parameter Categories
- **Performance parameters**: No batch size, no threading options
- **Output parameters**: No video saving options
- **ROI parameters**: No region-of-interest filtering
- **Appearance features**: No Re-ID model parameters

### 3. Hardcoded Values That Should Be Parameters
- Padding color in preprocessing (Detector_ONNX_YOLO7.py:170)
- ONNX execution providers (Detector_ONNX_YOLO7.py:42)
- Draw colors for track states (Track.py:255-260)
- Kalman filter initialization covariance multipliers (KalmanFilter.py:80-81)

### 4. Parameter Validation Missing
- No validation for threshold ranges (should be [0, 1])
- No validation for list lengths (match_thresholds should be length 3 for ByteTrack)
- No mutual constraint checking (creation_threshold vs detection_threshold)

---

## RECOMMENDATIONS

### High Priority
1. **Create a config file system** (YAML/JSON) instead of hardcoding in main.py
2. **Add parameter validation** in constructors
3. **Group related parameters** into dataclasses or config objects
4. **Expose device selection** for ONNX runtime
5. **Fix creation_threshold** inconsistency

### Medium Priority
6. **Add output saving parameters** for processed videos
7. **Add ROI support** for processing specific image regions
8. **Add performance parameters** (batch size, threading)
9. **Add min/max area filters** for bounding boxes
10. **Expose Kalman filter initialization** parameters

### Low Priority
11. Add appearance-based Re-ID parameters (future feature)
12. Add multi-camera support parameters (future feature)
13. Add dataset evaluation parameters (future feature)
