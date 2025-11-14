"""Comparison script to demonstrate the difference between basic and interactive processors.

This script shows side-by-side comparison of:
1. Basic VideoProcessor (original)
2. InteractiveVideoProcessor (enhanced)

Usage:
    python examples/compare_processors.py --video data/TownCent.mp4
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("\n" + "="*80)
print("VIDEO PROCESSOR COMPARISON")
print("="*80)

print("\nFeature Comparison:")
print("-" * 80)

features = [
    ("Feature", "Basic VideoProcessor", "Interactive VideoProcessor"),
    ("-" * 30, "-" * 20, "-" * 25),
    ("Real-time conf adjustment", "❌ No", "✅ Yes (trackbar)"),
    ("Playback speed control", "❌ No", "✅ Yes (0.1x - 5.0x)"),
    ("Pause/Resume", "❌ No", "✅ Yes (p key)"),
    ("Video seeking", "❌ No", "✅ Yes (arrow keys)"),
    ("Jump to position", "❌ No", "✅ Yes (0-9 keys)"),
    ("Visualization modes", "❌ No", "✅ Yes (5 modes)"),
    ("Toggle detections", "❌ No", "✅ Yes (d key)"),
    ("Statistics panel", "❌ No", "✅ Yes (auto)"),
    ("Help overlay", "❌ No", "✅ Yes (h key)"),
    ("Parameter reset", "❌ No", "✅ Yes (r key)"),
    ("Class-based colors", "❌ No", "✅ Yes (auto)"),
    ("Frame saving", "✅ Yes", "✅ Yes"),
    ("Step/continuous mode", "✅ Yes", "✅ Yes"),
    ("Video output", "✅ Yes", "✅ Yes"),
    ("Headless mode", "✅ Yes", "✅ Yes (--no-interactive)"),
]

for feature in features:
    print(f"{feature[0]:<30} {feature[1]:<20} {feature[2]:<25}")

print("-" * 80)

print("\nKeyboard Controls Comparison:")
print("-" * 80)
print(f"{'Key':<15} {'Basic':<30} {'Interactive':<30}")
print("-" * 80)

controls = [
    ("c", "Toggle step/continuous", "Toggle step/continuous"),
    ("SPACE", "Next frame (step mode)", "Next frame (step mode)"),
    ("s", "Save frame", "Save frame"),
    ("q", "Quit", "Quit"),
    ("p", "-", "Pause/Resume"),
    ("h", "-", "Show/hide help"),
    ("d", "-", "Toggle detections"),
    ("v", "-", "Cycle viz modes"),
    ("+/-", "-", "Adjust confidence"),
    ("←/→", "-", "Seek -/+ 10 frames"),
    ("↑/↓", "-", "Adjust speed"),
    ("0-9", "-", "Jump to percentage"),
    ("r", "-", "Reset parameters"),
]

for key, basic, interactive in controls:
    print(f"{key:<15} {basic:<30} {interactive:<30}")

print("-" * 80)

print("\nVisualization Modes (Interactive only):")
print("-" * 80)
viz_modes = [
    "1. 'boxes'      - Only bounding boxes",
    "2. 'labels'     - Boxes + class labels",
    "3. 'confidence' - Boxes + confidence scores",
    "4. 'full'       - Boxes + labels + confidence",
    "5. 'minimal'    - No overlays (clean video)",
]
for mode in viz_modes:
    print(f"  {mode}")

print("-" * 80)

print("\nTrackbar Controls (Interactive only):")
print("-" * 80)
trackbars = [
    "• Confidence:    Adjust detection threshold (0.0 - 1.0)",
    "• Speed:         Control playback speed (0.1x - 5.0x)",
    "• Skip Frames:   Process every Nth frame (1 - 30)",
    "• Viz Mode:      Select visualization mode (0 - 4)",
]
for trackbar in trackbars:
    print(f"  {trackbar}")

print("-" * 80)

print("\nHow to Use:")
print("-" * 80)
print("1. Basic VideoProcessor:")
print("   python -m src.main --video data/TownCent.mp4")
print()
print("2. Interactive VideoProcessor:")
print("   python examples/demo_interactive.py --video data/TownCent.mp4")
print()
print("3. Interactive (headless mode, same as basic):")
print("   python examples/demo_interactive.py --video data/TownCent.mp4 --no-interactive")

print("="*80)
print("\nRecommendation:")
print("  • Use BASIC for: batch processing, scripting, production pipelines")
print("  • Use INTERACTIVE for: development, debugging, demos, parameter tuning")
print("="*80 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare basic and interactive video processors'
    )
    parser.add_argument('--run-basic', action='store_true',
                       help='Run the basic VideoProcessor')
    parser.add_argument('--run-interactive', action='store_true',
                       help='Run the InteractiveVideoProcessor')
    parser.add_argument('--video', type=str, default='data/TownCent.mp4',
                       help='Video file to process')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.run_basic or args.run_interactive:
        from src.video_processor import VideoProcessor
        from src.interactive_video_processor import InteractiveVideoProcessor
        from src.detecting import DetectorFactory
        from src.config import Config
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Setup detector
        config = Config({
            'detection': {
                'type': 'object',
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        })
        detector = DetectorFactory.create(config, logger)

        if args.run_basic:
            print("\nRunning BASIC VideoProcessor...")
            processor = VideoProcessor(
                source=args.video,
                detector=detector
            )
            processor.run()

        if args.run_interactive:
            print("\nRunning INTERACTIVE VideoProcessor...")
            processor = InteractiveVideoProcessor(
                source=args.video,
                detector=detector,
                interactive=True
            )
            processor.run()
