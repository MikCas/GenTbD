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
from .detecting import DetectorFactory, get_registry
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
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', default=None,
                       help='Output video path (default: output_YYYYMMDD_HHMMSS.mp4)')
    
    # Information
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')

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

    # Handle informational flags
    if args.list_models:
        registry = get_registry()
        print("\nAvailable Models:")
        print("-" * 60)
        print(f"{'Name':<30} | {'Backend':<15} | {'Params':<10}")
        print("-" * 60)
        for name, spec in sorted(registry._models.items()):
            params = f"{spec.params_million:.1f}M" if spec.params_million > 0 else "?"
            print(f"{name:<30} | {spec.backend.value:<15} | {params:<10}")
        
        print("\nAliases:")
        for alias, target in sorted(registry._aliases.items()):
            print(f"  {alias:<20} -> {target}")
        print("-" * 60)
        return

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
    detector_type = config.get('detection.type', 'object_detection')
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

    # Setup detector using factory
    logger.info(f"Loading {detector_type} detector...")
    try:
        detector = DetectorFactory.create(config, logger)
    except Exception as e:
        logger.error(f"Failed to load detector: {e}", exc_info=verbose)
        return

    # TODO: ReID integration pending VideoProcessor refactoring
    # ReID feature extraction will be added when tracking system is implemented
    # See ARCHIVE/ for tracking code that will use ReID embeddings

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
        logger.error(f"{e}", exc_info=verbose)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
