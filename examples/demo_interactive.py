"""Demo script showing the enhanced interactive video processor.

This script demonstrates the new interactive features:
- Real-time parameter adjustment with trackbars
- Enhanced keyboard controls
- Multiple visualization modes
- Statistics tracking
- Video seeking and playback control

Usage:
    # With default video
    python examples/demo_interactive.py

    # With custom video
    python examples/demo_interactive.py --video data/your_video.mp4

    # With webcam
    python examples/demo_interactive.py --webcam

    # Headless mode (no interactive UI)
    python examples/demo_interactive.py --no-interactive
"""

import sys
import os
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.interactive_video_processor import InteractiveVideoProcessor
from src.detecting import DetectorFactory
from src.config import Config


def setup_logging(verbose=False):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interactive Video Processor Demo',
        epilog='Press "h" during playback to see all keyboard shortcuts'
    )

    # Video source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--video', type=str, default='data/TownCent.mp4',
                             help='Path to video file')
    source_group.add_argument('--webcam', type=int, nargs='?', const=0,
                             help='Use webcam (optionally specify index)')

    # Detection settings
    parser.add_argument('--model', default='mobilenet',
                       choices=['mobilenet', 'resnet50', 'retinanet'],
                       help='Detection model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Initial confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter by class IDs (e.g., --classes 1 for people)')

    # Processing settings
    parser.add_argument('--max-dimension', type=int, default=None,
                       help='Resize frames to max dimension before detection')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame')

    # Output settings
    parser.add_argument('--save-output', action='store_true',
                       help='Save processed video')
    parser.add_argument('--output', default=None,
                       help='Output video path')

    # UI settings
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive UI (use basic cv2 controls)')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    return parser.parse_args()


def main():
    """Main entry point for interactive demo."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Determine video source
    if args.webcam is not None:
        source = args.webcam
        logger.info(f"Using webcam (index: {source})")
    else:
        source = args.video
        if not os.path.exists(source):
            logger.error(f"Video file not found: {source}")
            return
        logger.info(f"Using video file: {source}")

    # Create minimal config for detector factory
    config_dict = {
        'detection': {
            'type': 'object',
            'model': args.model,
            'device': args.device,
            'conf_threshold': args.conf,
            'classes': args.classes
        }
    }
    config = Config(config_dict)

    # Load detector
    logger.info(f"Loading {args.model} detector on {args.device}...")
    try:
        detector = DetectorFactory.create(config, logger)
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return

    # Create interactive video processor
    interactive = not args.no_interactive

    if interactive:
        logger.info("Starting INTERACTIVE mode - Press 'h' for help")
    else:
        logger.info("Starting BASIC mode")

    processor = InteractiveVideoProcessor(
        source=source,
        detector=detector,
        save_output=args.save_output,
        output_path=args.output,
        max_dimension=args.max_dimension,
        skip_frames=args.skip_frames,
        interactive=interactive
    )

    # Print quick help
    if interactive:
        print("\n" + "="*60)
        print("INTERACTIVE CONTROLS ENABLED")
        print("="*60)
        print("Quick shortcuts:")
        print("  h - Show full help overlay")
        print("  p - Pause/Resume")
        print("  d - Toggle detections on/off")
        print("  v - Cycle visualization modes")
        print("  +/- - Adjust confidence threshold")
        print("  ←/→ - Seek backward/forward")
        print("  q - Quit")
        print("="*60 + "\n")

    # Run processor
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)


if __name__ == '__main__':
    main()
