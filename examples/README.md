# Examples Directory

This directory contains example scripts demonstrating various features of GenTbD.

## 📁 Files

### `demo_interactive.py`
Interactive video processor demo with enhanced UI controls.

**Features:**
- Real-time parameter adjustment via trackbars
- Enhanced keyboard controls (pause, seek, speed)
- Multiple visualization modes
- Statistics tracking

**Usage:**
```bash
# Basic usage
python examples/demo_interactive.py --video data/TownCent.mp4

# With webcam
python examples/demo_interactive.py --webcam

# Custom settings
python examples/demo_interactive.py \
    --video data/video.mp4 \
    --model mobilenet \
    --conf 0.7 \
    --device cpu \
    --max-dimension 640 \
    --save-output

# Disable interactive mode (headless)
python examples/demo_interactive.py --video data/video.mp4 --no-interactive
```

### `compare_processors.py`
Side-by-side comparison of basic vs interactive processors.

**Usage:**
```bash
# Show comparison table
python examples/compare_processors.py

# Run basic processor
python examples/compare_processors.py --run-basic --video data/video.mp4

# Run interactive processor
python examples/compare_processors.py --run-interactive --video data/video.mp4
```

## 🎯 Quick Examples

### 1. Interactive Development Mode
Perfect for tuning parameters and debugging:
```bash
python examples/demo_interactive.py \
    --video data/TownCent.mp4 \
    --conf 0.5 \
    --verbose
```
Then press `h` to see all controls.

### 2. Fast Preview Mode
Process every 5th frame for quick preview:
```bash
python examples/demo_interactive.py \
    --video data/TownCent.mp4 \
    --skip-frames 5 \
    --max-dimension 640
```

### 3. Webcam Demo
Live detection from webcam:
```bash
python examples/demo_interactive.py \
    --webcam \
    --model mobilenet \
    --conf 0.7 \
    --classes 1  # People only
```

### 4. Headless Processing
Batch processing without interactive UI:
```bash
python examples/demo_interactive.py \
    --video data/TownCent.mp4 \
    --no-interactive \
    --save-output \
    --output results/output.mp4
```

## 📊 Feature Comparison

| Feature | Basic Processor | Interactive Processor |
|---------|----------------|----------------------|
| Parameter adjustment | ❌ | ✅ (trackbars) |
| Pause/resume | ❌ | ✅ |
| Video seeking | ❌ | ✅ |
| Playback speed control | ❌ | ✅ |
| Visualization modes | ❌ | ✅ (5 modes) |
| Statistics panel | ❌ | ✅ |
| Help overlay | ❌ | ✅ |

## 🎮 Interactive Controls

### Keyboard Shortcuts
- `h` - Show/hide help overlay
- `p` - Pause/resume
- `d` - Toggle detections
- `v` - Cycle visualization modes
- `+/-` - Adjust confidence
- `←/→` - Seek frames
- `↑/↓` - Adjust speed
- `0-9` - Jump to position
- `r` - Reset parameters
- `q` - Quit

### Trackbar Controls
- **Confidence** - Detection threshold (0.0 - 1.0)
- **Speed** - Playback speed (0.1x - 5.0x)
- **Skip Frames** - Process every Nth frame
- **Viz Mode** - Visualization mode selector

## 📚 More Information

See [INTERACTIVE_README.md](../INTERACTIVE_README.md) for detailed documentation.

## 💡 Tips

1. **Start interactive**: Use `demo_interactive.py` for development
2. **Tune parameters**: Adjust trackbars in real-time
3. **Compare results**: Use different visualization modes
4. **Go headless**: Add `--no-interactive` for production
5. **Save presets**: Note good parameter combinations in config files
