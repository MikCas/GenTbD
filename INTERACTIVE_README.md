# Interactive Video Processor Guide

This guide explains the new **InteractiveVideoProcessor** and how to use its enhanced features.

## 🎯 Overview

The `InteractiveVideoProcessor` extends the basic `VideoProcessor` with:

- **Real-time parameter adjustment** via trackbars
- **Enhanced keyboard controls** (pause, seek, speed control)
- **Multiple visualization modes** (boxes, labels, confidence, full, minimal)
- **Statistics tracking** (detections per class, processing time)
- **Help overlay** (press 'h' for shortcuts)
- **Backward compatible** (works as drop-in replacement)

---

## 🚀 Quick Start

### Basic Usage

```python
from src.interactive_video_processor import InteractiveVideoProcessor
from src.detecting import DetectorFactory
from src.config import Config

# Setup detector
config = Config.from_default()
detector = DetectorFactory.create(config)

# Create interactive processor
processor = InteractiveVideoProcessor(
    source='data/video.mp4',
    detector=detector,
    interactive=True  # Enable interactive features
)

# Run
processor.run()
```

### Command Line

```bash
# With interactive controls
python examples/demo_interactive.py --video data/TownCent.mp4

# Webcam with interactive controls
python examples/demo_interactive.py --webcam

# Disable interactive mode (use basic controls)
python examples/demo_interactive.py --video data/TownCent.mp4 --no-interactive
```

---

## ⌨️ Keyboard Shortcuts

### Navigation & Playback
| Key | Action |
|-----|--------|
| `p` | Pause/Resume |
| `←` | Seek backward 10 frames |
| `→` | Seek forward 10 frames |
| `↑` | Increase playback speed |
| `↓` | Decrease playback speed |
| `0-9` | Jump to 0%, 10%, ..., 90% of video |

### Visualization
| Key | Action |
|-----|--------|
| `d` | Toggle detections on/off |
| `v` | Cycle through visualization modes |
| `h` | Show/hide help overlay |

### Parameters
| Key | Action |
|-----|--------|
| `+` or `=` | Increase confidence threshold |
| `-` or `_` | Decrease confidence threshold |
| `r` | Reset all parameters to defaults |

### Other
| Key | Action |
|-----|--------|
| `c` | Toggle continuous/step mode |
| `SPACE` | Next frame (in step mode) |
| `s` | Save current frame as image |
| `q` | Quit |

---

## 🎨 Visualization Modes

Press `v` to cycle through modes, or use the **Viz Mode** trackbar:

1. **boxes** - Only bounding boxes (colored by class)
2. **labels** - Bounding boxes + class labels
3. **confidence** - Bounding boxes + confidence scores
4. **full** - Everything (boxes + labels + confidence)
5. **minimal** - No overlays (clean video)

---

## 🎛️ Trackbar Controls

The **Controls** window provides real-time adjustment:

### Confidence (0-100)
- Adjusts detection threshold (0.0 - 1.0)
- Higher = fewer false positives, may miss detections
- Lower = more detections, may include false positives
- **Tip**: Start at 50 (0.5) and adjust based on your needs

### Speed (1-50)
- Controls playback speed (0.1x - 5.0x)
- 10 = 1.0x (normal speed)
- 5 = 0.5x (slow motion)
- 20 = 2.0x (double speed)
- **Tip**: Use slower speeds for detailed analysis

### Skip Frames (1-30)
- Process every Nth frame
- 1 = process all frames (slowest, most accurate)
- 5 = process every 5th frame (5x faster, may miss objects)
- **Tip**: Use higher values for faster preview, lower for final output

### Viz Mode (0-4)
- Select visualization mode (see above)
- 0 = boxes, 1 = labels, 2 = confidence, 3 = full, 4 = minimal

---

## 📊 Statistics Panel

Located on the right side of the video, shows:

- **Total Detections**: Cumulative count across all frames
- **Avg/Frame**: Average detections per frame
- **Avg Time**: Average processing time per frame (milliseconds)
- **Top Classes**: Most detected object classes with percentages

---

## 🎯 Use Cases

### Development & Debugging
- Use **interactive mode** to tune parameters in real-time
- Pause at specific frames to inspect detections
- Toggle detections on/off to compare
- Check statistics to identify issues

### Demos & Presentations
- Use **full visualization mode** to show all information
- Adjust confidence live to demonstrate threshold effects
- Pause and seek to highlight specific moments
- Show help overlay to explain controls

### Parameter Tuning
- Start with default parameters
- Adjust confidence until false positives are minimized
- Find optimal skip_frames for speed/accuracy balance
- Reset and try different settings

### Batch Processing
- Use `interactive=False` or `--no-interactive` flag
- Same as basic VideoProcessor
- No trackbars or interactive features
- Optimal for scripts and automation

---

## 🔄 Switching Between Processors

### Option 1: Import Different Class

```python
# Basic (original)
from src.video_processor import VideoProcessor
processor = VideoProcessor(source, detector)

# Interactive (enhanced)
from src.interactive_video_processor import InteractiveVideoProcessor
processor = InteractiveVideoProcessor(source, detector, interactive=True)
```

### Option 2: Use Flag

```python
from src.interactive_video_processor import InteractiveVideoProcessor

# Interactive mode
processor = InteractiveVideoProcessor(source, detector, interactive=True)

# Headless mode (same as basic)
processor = InteractiveVideoProcessor(source, detector, interactive=False)
```

### Option 3: Command Line

```bash
# Basic processor
python -m src.main --video data/video.mp4

# Interactive processor
python examples/demo_interactive.py --video data/video.mp4

# Interactive processor in headless mode
python examples/demo_interactive.py --video data/video.mp4 --no-interactive
```

---

## 🛠️ Advanced Usage

### Custom Initial Parameters

```python
processor = InteractiveVideoProcessor(
    source='data/video.mp4',
    detector=detector,
    interactive=True,
    max_dimension=640,      # Resize for speed
    skip_frames=3,          # Process every 3rd frame
    save_output=True,       # Save processed video
    output_path='output.mp4'
)
```

### Access Statistics During Processing

```python
# After processing
print(f"Total detections: {processor.total_detections}")
print(f"Detections per class: {processor.class_counts}")
print(f"Average processing time: {sum(processor.frame_times) / len(processor.frame_times)}")
```

### Programmatic Parameter Changes

```python
# Change confidence threshold
processor.current_conf_threshold = 0.7
processor.detector.conf_threshold = 0.7

# Change visualization mode
from src.interactive_video_processor import VisualizationMode
processor.visualization_mode = VisualizationMode.MINIMAL

# Change playback speed
processor.playback_speed = 2.0
```

---

## 📝 Tips & Best Practices

### For Development
1. Start with **full visualization mode** to see everything
2. Use **pause** to inspect specific frames
3. Adjust **confidence** while paused to see immediate effect
4. Use **statistics panel** to track performance

### For Demos
1. Use **full or labels mode** for clarity
2. Keep **playback speed at 1.0x** for natural viewing
3. Show **help overlay** to explain controls
4. Use **seek controls** to jump to interesting moments

### For Performance Tuning
1. Start with **skip_frames=1** to see all detections
2. Gradually increase **skip_frames** until quality drops
3. Adjust **max_dimension** to balance speed/accuracy
4. Monitor **Avg Time** in statistics panel

### For Production
1. Use `interactive=False` to disable UI overhead
2. Set all parameters via config or CLI
3. Use **save_output=True** for batch processing
4. Monitor logs instead of visual output

---

## 🐛 Troubleshooting

### Trackbar window not showing
- Ensure `interactive=True` is set
- Check that cv2 GUI backend is available
- Try running with `--verbose` to see initialization logs

### Video seeking not working
- Seeking only works with video files, not webcams/streams
- Some video codecs don't support seeking
- Try converting video to a different format (e.g., MP4 with H.264)

### Slow performance
- Increase `skip_frames` to process fewer frames
- Set `max_dimension` to reduce frame size (e.g., 640)
- Use faster model (mobilenet instead of resnet50)
- Use GPU if available (`--device cuda` or `--device mps`)

### Detections not showing
- Press `d` to toggle detections on
- Check `visualization_mode` (press `v` to cycle)
- Verify confidence threshold isn't too high (lower it with `-`)
- Check that classes are not filtered out

---

## 🔗 Related Files

- `src/interactive_video_processor.py` - Enhanced processor implementation
- `src/video_processor.py` - Original basic processor
- `examples/demo_interactive.py` - Demo script with full setup
- `examples/compare_processors.py` - Side-by-side comparison

---

## 📚 See Also

- [Main README](README.md) - Project overview
- [Configuration Guide](config/README.md) - Config file documentation
- [Detector Documentation](src/detecting/README.md) - Detector details

---

**Need help?** Press `h` during playback to see keyboard shortcuts!
