# Add Interactive Video Processor with Enhanced UI Controls

## 🎯 Overview
This PR adds an enhanced `InteractiveVideoProcessor` that extends the base `VideoProcessor` with real-time parameter adjustment and improved user experience for development, debugging, and demos.

## ✨ Key Features

### Real-time Parameter Adjustment
- **Confidence Threshold** - Adjust detection threshold (0.0-1.0) with slider
- **Playback Speed** - Control speed from 0.1x to 5.0x
- **Skip Frames** - Process every Nth frame (1-30)
- **Visualization Mode** - Switch between 5 different display modes

### Enhanced Keyboard Controls
| Key | Action |
|-----|--------|
| `p` | Pause/Resume |
| `h` | Show/Hide help overlay |
| `d` | Toggle detections on/off |
| `v` | Cycle visualization modes |
| `+/-` | Adjust confidence threshold |
| `←/→` | Seek backward/forward 10 frames |
| `↑/↓` | Adjust playback speed |
| `0-9` | Jump to percentage (0%-90%) |
| `r` | Reset parameters to defaults |
| `c` | Toggle continuous/step mode |
| `SPACE` | Next frame (step mode) |
| `s` | Save current frame |
| `q` | Quit |

### Visualization Modes
1. **boxes** - Only bounding boxes (colored by class)
2. **labels** - Boxes + class labels
3. **confidence** - Boxes + confidence scores
4. **full** - Boxes + labels + confidence (default)
5. **minimal** - Clean video, no overlays

### Statistics Panel
- Total detections count
- Average detections per frame
- Processing time statistics
- Top 5 detected classes with percentages

## 📁 New Files

```
GenTbD/
├── src/
│   └── interactive_video_processor.py    [570 lines - Enhanced processor]
├── examples/
│   ├── demo_interactive.py               [180 lines - Demo script]
│   ├── compare_processors.py             [150 lines - Comparison tool]
│   └── README.md                         [Examples guide]
├── INTERACTIVE_README.md                 [Complete user guide]
```

## 🚀 Usage

### Basic Usage
```bash
# Interactive mode with all features
python examples/demo_interactive.py --video data/TownCent.mp4

# Webcam
python examples/demo_interactive.py --webcam

# Headless mode (backward compatible)
python examples/demo_interactive.py --video data/video.mp4 --no-interactive
```

### Programmatic Usage
```python
from src.interactive_video_processor import InteractiveVideoProcessor

# With interactive UI
processor = InteractiveVideoProcessor(
    source='data/video.mp4',
    detector=detector,
    interactive=True  # Enable trackbars and enhanced controls
)
processor.run()

# Headless (same as VideoProcessor)
processor = InteractiveVideoProcessor(
    source='data/video.mp4',
    detector=detector,
    interactive=False
)
processor.run()
```

## ✅ Backward Compatibility

The `InteractiveVideoProcessor` is a drop-in replacement for `VideoProcessor`:
- Same constructor signature (with optional `interactive` parameter)
- Same public API
- No breaking changes to existing code
- Can be disabled via `interactive=False` for headless operation

## 📊 Comparison: Before vs After

| Feature | VideoProcessor | InteractiveVideoProcessor |
|---------|----------------|---------------------------|
| Real-time parameter adjustment | ❌ | ✅ (trackbars) |
| Pause/resume | ❌ | ✅ |
| Video seeking | ❌ | ✅ |
| Playback speed control | ❌ | ✅ |
| Visualization modes | ❌ | ✅ (5 modes) |
| Statistics panel | ❌ | ✅ |
| Help overlay | ❌ | ✅ |
| Frame saving | ✅ | ✅ |
| Step/continuous mode | ✅ | ✅ |
| Headless mode | ✅ | ✅ |

## 🧪 Testing

### Manual Testing
```bash
# Test basic functionality
python examples/demo_interactive.py --video data/TownCent.mp4

# Test headless mode
python examples/demo_interactive.py --video data/TownCent.mp4 --no-interactive

# View comparison
python examples/compare_processors.py
```

### What to Test
- [x] Trackbars adjust parameters in real-time
- [x] Pause/resume functionality works
- [x] Video seeking with arrow keys
- [x] Playback speed adjustment
- [x] Visualization mode cycling
- [x] Help overlay displays correctly
- [x] Statistics panel updates in real-time
- [x] Headless mode works without trackbars
- [x] Backward compatible with VideoProcessor API

## 📚 Documentation

- **INTERACTIVE_README.md** - Complete user guide with all features, controls, and examples
- **examples/README.md** - Quick start guide for examples
- **Code comments** - Detailed docstrings in all new methods

## 🎯 Use Cases

### Development & Debugging
- Tune parameters in real-time without restarting
- Pause and inspect specific frames
- Toggle detections to compare results
- Monitor statistics for performance issues

### Demos & Presentations
- Show all information with full visualization mode
- Adjust parameters live to demonstrate effects
- Pause and seek to highlight moments
- Display help overlay to explain controls

### Parameter Tuning
- Find optimal confidence threshold
- Balance speed vs accuracy with skip_frames
- Test different visualization modes
- Reset and compare settings

## 🔮 Future Enhancements

Potential future additions (not in this PR):
- Parameter presets (save/load favorite configurations)
- Gradio web UI for browser-based demos
- Export statistics to CSV
- Detection trails/trajectories visualization
- Heatmap mode for detection density

## 📝 Checklist

- [x] Code follows project style guidelines
- [x] Comprehensive documentation added
- [x] Examples and demo scripts provided
- [x] Backward compatible with existing VideoProcessor
- [x] No new external dependencies (uses existing cv2)
- [x] Help overlay for user guidance
- [x] Clean git history with descriptive commit message

## 🙏 Review Notes

This PR significantly improves the developer experience and makes the framework more accessible to non-technical users while maintaining full backward compatibility. The implementation uses only existing dependencies (cv2 trackbars) for zero overhead.

Suggested review focus:
1. API compatibility with VideoProcessor
2. Code quality and documentation
3. User experience of interactive controls
4. Performance impact (should be minimal when `interactive=False`)
