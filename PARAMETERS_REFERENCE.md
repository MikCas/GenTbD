# GenTbD Parameters Quick Reference

**Branch:** main
**Last Updated:** 2025-11-14

This document provides a quick reference for all configurable parameters in the GenTbD system.

---

## How to Use This Guide

All parameters are currently configured in `src/main.py`. To modify system behavior:

1. Open `src/main.py`
2. Locate the parameter section (DETECTOR, TRACKER, or VIDEO SETUP)
3. Modify the values
4. Run `python src/main.py`

---

## Current Parameter Values (main.py)

### 🎯 DETECTOR Parameters (Lines 38-49)

| Parameter | Current Value | Type | Description | Valid Range |
|-----------|---------------|------|-------------|-------------|
| `model_path` | `'models/yolov7_640x640.onnx'` | str | Path to ONNX model file | Valid file path |
| `confidence_threshold` | `0.1` | float | Minimum confidence for detections | 0.0 - 1.0 |
| `iou_threshold` | `0.5` | float | IoU threshold for NMS | 0.0 - 1.0 |
| `classes` | `[0]` | list[int] | Class IDs to detect (0=person) | Valid COCO class IDs |

**Typical Adjustments:**
- **Higher precision:** `confidence_threshold = 0.3-0.5` (fewer false positives)
- **Higher recall:** `confidence_threshold = 0.1-0.2` (more detections, more false positives)
- **Less NMS suppression:** `iou_threshold = 0.6-0.8` (keep more overlapping boxes)
- **More NMS suppression:** `iou_threshold = 0.3-0.4` (remove more overlapping boxes)

---

### 🔍 TRACKER Parameters (Lines 52-65)

| Parameter | Current Value | Type | Description | Valid Range |
|-----------|---------------|------|-------------|-------------|
| `detection_threshold` | `0.3` | float | Split high/low confidence detections | 0.0 - 1.0 |
| `creation_threshold` | `0.3` | float | Min confidence to create new track | 0.0 - 1.0 |
| `match_thresholds` | `[0.4, 0.2, 0.2]` | list[float] | Cascaded matching thresholds | 0.0 - 1.0 each |
| `activation_threshold` | `10` | int | Frames before track activation | 1 - 100 |
| `deactivation_threshold` | `15` | int | Frames before track removal | 1 - 1000 |

**ByteTrack Association Stages:**
1. `match_thresholds[0] = 0.4`: Match high-conf detections with MATCHED+LOST tracks
2. `match_thresholds[1] = 0.2`: Match low-conf detections with remaining tracks
3. `match_thresholds[2] = 0.2`: Match remaining high-conf detections with NEW tracks

**Typical Adjustments:**

**For crowded scenes:**
```python
detection_threshold = 0.4      # Stricter split
creation_threshold = 0.5       # Only create from high-confidence
match_thresholds = [0.5, 0.3, 0.3]  # Stricter matching
activation_threshold = 5       # Activate faster
deactivation_threshold = 10    # Remove faster
```

**For sparse scenes with occlusions:**
```python
detection_threshold = 0.2      # More lenient split
creation_threshold = 0.3       # Allow lower confidence tracks
match_thresholds = [0.3, 0.1, 0.1]  # More lenient matching
activation_threshold = 15      # Wait longer to activate
deactivation_threshold = 30    # Keep tracks longer
```

**For fast-moving objects:**
```python
match_thresholds = [0.5, 0.3, 0.3]  # Stricter (objects move more)
deactivation_threshold = 10    # Remove faster
```

---

### 🎥 VIDEO PROCESSOR Parameters (Lines 68-76)

| Parameter | Current Value | Type | Description | Valid Options |
|-----------|---------------|------|-------------|---------------|
| `video_file` | `'TownCent.mp4'` | str | Video filename in data/ folder | Valid video file |
| `draw_mode` | `'state'` | str | Visualization mode | `'state'`, `'id'`, `'none'` |

**Draw Modes:**
- `'state'`: Color by track state (NEW=Blue, MATCHED=Green, LOST=Red)
- `'id'`: Unique color per track ID
- `'none'`: No visualization (fastest)

---

### 🔧 SYSTEM Parameters (Lines 8-21)

| Parameter | Current Value | Type | Description | Valid Options |
|-----------|---------------|------|-------------|---------------|
| `log_level` | `logging.DEBUG` | int | Logging verbosity | DEBUG, INFO, WARNING, ERROR |

**Log Levels:**
- `logging.DEBUG`: All messages (slowest, most verbose)
- `logging.INFO`: Informational messages
- `logging.WARNING`: Warnings and errors only
- `logging.ERROR`: Errors only (fastest)

**For performance testing:** Set to `logging.WARNING` or `logging.ERROR`

---

## Parameter Dependencies & Constraints

### ⚠️ Important Relationships

1. **creation_threshold vs detection_threshold**
   ```python
   # RECOMMENDED: creation_threshold >= detection_threshold
   creation_threshold = 0.4  # Only high-conf detections create tracks
   detection_threshold = 0.3  # Split at 0.3
   ```

   **Current setting has them equal (0.3)** - This means all high-confidence detections can create tracks. Consider increasing `creation_threshold` to 0.4 or higher.

2. **match_thresholds list length**
   - Must have exactly **3 values** for ByteTrack algorithm
   - Order matters: [high-conf matching, low-conf matching, new track matching]

3. **activation_threshold vs deactivation_threshold**
   ```python
   # RECOMMENDED: deactivation_threshold > activation_threshold
   activation_threshold = 10   # Activate after 10 matches
   deactivation_threshold = 15 # Remove after 15 misses
   ```

4. **confidence_threshold vs detection_threshold vs creation_threshold**
   ```python
   # Typical hierarchy:
   confidence_threshold = 0.1    # Detector accepts detections >= 0.1
   detection_threshold = 0.3     # High/low split at 0.3
   creation_threshold = 0.4      # New tracks need >= 0.4

   # Ensures: Only confident detections create tracks
   ```

---

## Hidden Parameters (Hardcoded in Source)

These parameters are not exposed in main.py but affect system behavior:

### Track Parameters
| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `TRAJECTORY_MAX_SIZE` | 50 | Track.py:36 | Max detections stored per track |

### Kalman Filter Parameters
| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `stdPosition` | 0.05 | KalmanFilter.py:22 | Position noise std dev |
| `stdVelocity` | 0.00625 | KalmanFilter.py:22 | Velocity noise std dev |
| `dt` | 1.0 | KalmanFilter.py:22 | Time step |

### ONNX Runtime Parameters
| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `providers` | `['CoreMLExecutionProvider', 'CPUExecutionProvider']` | Detector_ONNX_YOLO7.py:42 | Execution providers |

---

## Quick Tuning Guide

### Problem: Too many false positive tracks
**Solution:**
```python
confidence_threshold = 0.3     # ⬆️ Increase
creation_threshold = 0.5       # ⬆️ Increase
activation_threshold = 15      # ⬆️ Increase
```

### Problem: Missing true objects
**Solution:**
```python
confidence_threshold = 0.1     # ⬇️ Decrease
detection_threshold = 0.2      # ⬇️ Decrease
creation_threshold = 0.2       # ⬇️ Decrease
```

### Problem: Too many ID switches
**Solution:**
```python
match_thresholds = [0.3, 0.1, 0.1]  # ⬇️ More lenient
deactivation_threshold = 30         # ⬆️ Keep tracks longer
```

### Problem: Tracks stay too long after object leaves
**Solution:**
```python
deactivation_threshold = 10    # ⬇️ Remove faster
```

### Problem: Objects not tracked consistently
**Solution:**
```python
activation_threshold = 5       # ⬇️ Activate faster
match_thresholds = [0.4, 0.2, 0.2]  # ⬆️ Stricter matching
```

---

## Example Configurations

### Configuration 1: High Precision (Few false positives)
```python
# DETECTOR
confidence_threshold = 0.4
iou_threshold = 0.5
classes = [0]

# TRACKER
detection_threshold = 0.5
creation_threshold = 0.6
match_thresholds = [0.5, 0.3, 0.3]
activation_threshold = 15
deactivation_threshold = 10
```

### Configuration 2: High Recall (Catch all objects)
```python
# DETECTOR
confidence_threshold = 0.05
iou_threshold = 0.6
classes = [0]

# TRACKER
detection_threshold = 0.2
creation_threshold = 0.2
match_thresholds = [0.3, 0.1, 0.1]
activation_threshold = 5
deactivation_threshold = 30
```

### Configuration 3: Balanced (Default)
```python
# DETECTOR
confidence_threshold = 0.1
iou_threshold = 0.5
classes = [0]

# TRACKER
detection_threshold = 0.3
creation_threshold = 0.4  # ⚠️ Currently 0.3, recommend 0.4
match_thresholds = [0.4, 0.2, 0.2]
activation_threshold = 10
deactivation_threshold = 15
```

### Configuration 4: Fast Performance (For benchmarking)
```python
# DETECTOR
confidence_threshold = 0.3  # Higher = fewer detections = faster
iou_threshold = 0.5
classes = [0]

# TRACKER (same as balanced)
detection_threshold = 0.3
creation_threshold = 0.4
match_thresholds = [0.4, 0.2, 0.2]
activation_threshold = 10
deactivation_threshold = 15

# SYSTEM
log_level = logging.ERROR  # Minimal logging
draw_mode = 'none'  # No visualization
```

---

## Parameter Change Impact Matrix

| Parameter | ⬆️ Increase Effect | ⬇️ Decrease Effect |
|-----------|-------------------|-------------------|
| `confidence_threshold` | Fewer detections, higher precision | More detections, lower precision |
| `iou_threshold` | Keep more overlapping boxes | Remove more overlapping boxes |
| `detection_threshold` | Fewer "high-confidence" detections | More "high-confidence" detections |
| `creation_threshold` | Fewer new tracks created | More new tracks created |
| `match_thresholds` | Stricter matching, more ID switches | Looser matching, fewer ID switches |
| `activation_threshold` | Slower track activation | Faster track activation |
| `deactivation_threshold` | Tracks persist longer | Tracks removed faster |

---

## COCO Class IDs Reference

Common classes for the `classes` parameter:

```python
classes = [0]      # Person only (most common)
classes = [0, 1]   # Person + Bicycle
classes = [0, 2]   # Person + Car
classes = [2, 3, 5, 7]  # Vehicles: Car, Motorcycle, Bus, Truck
```

Full list:
```
0: person          1: bicycle        2: car            3: motorcycle
4: airplane        5: bus            6: train          7: truck
8: boat            9: traffic light  10: fire hydrant  11: stop sign
12: parking meter  13: bench         14: bird          15: cat
16: dog            17: horse         18: sheep         19: cow
20: elephant       21: bear          22: zebra         23: giraffe
24: backpack       25: umbrella      26: handbag       27: tie
28: suitcase       29: frisbee       30: skis          31: snowboard
... (80 classes total)
```

---

## Performance vs Accuracy Trade-offs

| Goal | Priority | Parameters to Adjust |
|------|----------|---------------------|
| **Maximum Speed** | Performance | ⬆️ `confidence_threshold`, ➡️ `log_level=ERROR`, ➡️ `draw_mode='none'` |
| **Maximum Accuracy** | Quality | ⬇️ All thresholds, ⬆️ `activation_threshold`, ⬆️ `deactivation_threshold` |
| **Crowded Scenes** | Quality | ⬆️ All thresholds, ⬇️ `activation_threshold`, ⬇️ `deactivation_threshold` |
| **Sparse Scenes** | Quality | ⬇️ Thresholds, ⬆️ `deactivation_threshold` |
| **Fast Motion** | Quality | ⬆️ `match_thresholds`, ⬇️ `deactivation_threshold` |
| **Occlusions** | Quality | ⬇️ `match_thresholds`, ⬆️ `deactivation_threshold` |

---

## Testing Your Configuration

After changing parameters, test with:

```bash
# Run with your new parameters
python src/main.py

# Observe:
# - Are all objects being detected?
# - Are there false positives (wrong detections)?
# - Do tracks stay consistent (same ID)?
# - Are tracks created/removed at the right time?
```

**Performance metrics to track:**
- FPS (frames per second)
- Number of active tracks
- Number of ID switches
- Detection recall (objects detected / total objects)
- Detection precision (correct detections / total detections)

---

## Next Steps

For more detailed information, see:
- **PARAMETER_AUDIT.md** - Complete parameter inventory with missing parameters
- **PERFORMANCE_ANALYSIS.md** - Performance bottlenecks and optimization guide
- **AUDIT_SUMMARY.md** - Executive summary and action plan

To implement a configuration file system (recommended):
- See PARAMETER_AUDIT.md Section: "PARAMETER GROUPINGS - RECOMMENDED STRUCTURE"
- Migrate from hardcoded main.py values to YAML/JSON config files
