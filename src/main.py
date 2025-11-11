"""Video detection processor - main entry point.

Usage:
    python -m src.main --video data/TownCent.mp4
    python -m src.main --video data/TownCent.mp4 --conf 0.7 --save-output

Controls:
    c      - Toggle continuous/step mode
    SPACE  - Next frame (in step mode)
    s      - Save current frame as image
    q      - Quit
"""

import argparse
import logging
from video_processor import VideoProcessor
from detecting.detectors import ObjectDetector

logger = logging.getLogger(__name__)

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video detection processor',
        epilog='Controls: c=toggle mode | SPACE=next frame | s=save frame | q=quit'
    )
    # Video processor arguments
    parser.add_argument('--video', default='data/TownCent.mp4',
                       help='Path to input video')
    parser.add_argument('--max-dimension', type=int, default=None,
                       help='Resize frames to max dimension before detection (e.g., 640 for speed)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (1=all frames, 5=every 5th frame)')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', default=None,
                       help='Output video path (default: output_YYYYMMDD_HHMMSS.mp4)')

    # Detector arguments
    parser.add_argument('--model', default='mobilenet',
                       help='Detection model: resnet50 (accurate), mobilenet (fast), retinanet')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='cpu',
                       help='Device for detection: cpu, mps, or cuda')

    # Logger arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    return parser.parse_args()


def setup_logging(verbose):
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

def main():
    """Main entry point for video processing with detection."""
    args = setup_arguments()
    setup_logging(args.verbose)

    # Setup detector
    logger.info(f"Loading {args.model} detector...")
    try:
        detector = ObjectDetector(
            model=args.model,
            device=args.device,
            conf_threshold=args.conf
        )
        logger.info(f"Detector loaded on device: {args.device}")
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return

    # Setup and run video processor
    try:
        processor = VideoProcessor(
            video_path=args.video,
            detector=detector,
            save_output=args.save_output,
            output_path=args.output,
            max_dimension=args.max_dimension,
            skip_frames=args.skip_frames
        )
        processor.run()
    except ValueError as e:
        logger.error(f"{e}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == '__main__':
    main()
