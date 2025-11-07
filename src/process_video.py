"""Video detection processor - main entry point.

Usage:
    python src/process_video.py --video data/TownCent.mp4
    python src/process_video.py --video data/TownCent.mp4 --conf 0.7 --save-output

Controls:
    c      - Toggle continuous/step mode
    SPACE  - Next frame (in step mode)
    s      - Save current frame as image
    q      - Quit
"""

import argparse
import logging
from .video_processor import VideoProcessor
from .detecting.detectors import ObjectDetector

logger = logging.getLogger(__name__)


def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video detection processor',
        epilog='Controls: c=toggle mode | SPACE=next frame | s=save frame | q=quit'
    )
    parser.add_argument('--video', default='data/TownCent.mp4',
                       help='Path to input video')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='cpu',
                       help='Device for detection: cpu, mps, or cuda')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter by class IDs (e.g., --classes 1 for people only)')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', default=None,
                       help='Output video path (default: output_YYYYMMDD_HHMMSS.mp4)')
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

    # Load detector
    logger.info("Loading FasterRCNN detector...")
    try:
        detector = ObjectDetector.from_fasterrcnn_resnet50(
            device=args.device,
            conf_threshold=args.conf,
            classes=args.classes
        )
        logger.info(f"Detector loaded on device: {args.device}")
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return

    # Create and run processor
    try:
        processor = VideoProcessor(
            video_path=args.video,
            detector=detector,
            save_output=args.save_output,
            output_path=args.output
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
