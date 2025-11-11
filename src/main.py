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
from .video_processor import VideoProcessor
from .detecting.detectors import ObjectDetector, KeypointDetector
from .config import Config

logger = logging.getLogger(__name__)

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video detection processor',
        epilog='Controls: c=toggle mode | SPACE=next frame | s=save frame | q=quit'
    )
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (default: config/default.yaml)')

    # Video source arguments (mutually exclusive: video file or webcam)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--video', type=str, default=None,
                             help='Path to input video file')
    source_group.add_argument('--webcam', type=int, nargs='?', const=0, default=None,
                             help='Use webcam as input (optionally specify camera index, default: 0)')

    # Video processor arguments
    parser.add_argument('--max-dimension', type=int, default=None,
                       help='Resize frames to max dimension before detection (e.g., 640 for speed)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (1=all frames, 5=every 5th frame)')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', default=None,
                       help='Output video path (default: output_YYYYMMDD_HHMMSS.mp4)')

    # Detector arguments
    parser.add_argument('--detector-type', type=str, default=None, choices=['object', 'keypoint'],
                       help='Detector type: object (default) or keypoint (human pose)')
    parser.add_argument('--model', default=None,
                       help='Detection model: resnet50, mobilenet, retinanet (object); resnet50 (keypoint)')
    parser.add_argument('--conf', type=float, default=None,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default=None,
                       help='Device for detection: cpu, mps, or cuda')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter by class IDs (e.g., --classes 1 for people only) - object detector only')

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

    # Load configuration (config file + command-line args)
    if args.config:
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = Config.from_default()
        logger.info("Using default configuration")

    # Merge command-line arguments (they take precedence)
    config.merge_args(args)

    # Setup logging
    verbose = config.get('logging.verbose', False)
    setup_logging(verbose)

    # Get configuration values
    source = config.get('video.source', 'data/TownCent.mp4')
    detector_type = config.get('detection.type', 'object')
    device = config.get('detection.device', 'cpu')
    conf_threshold = config.get('detection.conf_threshold', 0.5)
    max_dimension = config.get('video.max_dimension', None)
    skip_frames = config.get('video.skip_frames', 1)
    save_output = config.get('video.save_output', False)
    output_path = config.get('video.output_path', None)

    # Log source
    if isinstance(source, int):
        logger.info(f"Using webcam (camera index: {source})")
    else:
        logger.info(f"Using video source: {source}")

    # Setup detector based on type
    logger.info(f"Loading {detector_type} detector...")
    try:
        if detector_type == 'keypoint':
            # Keypoint detector (human pose)
            model = config.get('detection.model', 'resnet50')
            keypoint_threshold = config.get('detection.keypoint.keypoint_threshold', 0.5)

            detector = KeypointDetector(
                model=model,
                device=device,
                conf_threshold=conf_threshold,
                keypoint_threshold=keypoint_threshold
            )
            logger.info(f"Keypoint detector loaded on device: {device}")
        else:
            # Object detector
            model = config.get('detection.model', 'mobilenet')
            classes = config.get('detection.classes', None)

            detector = ObjectDetector(
                model=model,
                device=device,
                conf_threshold=conf_threshold,
                classes=classes
            )
            if classes:
                logger.info(f"Object detector loaded on device: {device}, filtering classes: {classes}")
            else:
                logger.info(f"Object detector loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return

    # Setup and run video processor
    try:
        processor = VideoProcessor(
            source=source,
            detector=detector,
            save_output=save_output,
            output_path=output_path,
            max_dimension=max_dimension,
            skip_frames=skip_frames
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
