import cv2
import time
import argparse
import logging
import os
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Constants
FPS_WINDOW = 30
TEXT_COLOR = (255, 0, 0)  # BGR blue
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def setup_video(video_path, save_output, output_path=None):
    """Setup video capture and writer. Handles errors and returns video resources."""
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    props = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    logger.info(f"Loaded video: {props['width']}x{props['height']}, "
                f"{props['fps']:.1f} FPS, {props['total_frames']} frames")

    out = None
    if save_output:
        if not output_path:
            output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, props['fps'],
                             (props['width'], props['height']))
        if not out.isOpened():
            raise ValueError(f"Could not open video writer: {output_path}")
        logger.info(f"Saving to: {output_path}")

    return cap, out, props

def process_frame(frame, detector):
    """Run detection on frame. Returns (detections, elapsed_time)."""
    start_time = time.time()

    detections = []
    if detector:
        detections = detector.detect(frame)
        detector.draw_detections(frame, detections)

    elapsed = time.time() - start_time
    return detections, elapsed

def draw_frame(frame, fps, frame_num, total_frames, mode, detection_count):
    """Draw FPS, frame count, mode and detection count on output frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), TEXT_FONT, 1, TEXT_COLOR, 2)
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 70), TEXT_FONT, 1, TEXT_COLOR, 2)
    cv2.putText(frame, f"Mode: {mode}", (10, 110), TEXT_FONT, 1, TEXT_COLOR, 2)
    if detection_count > 0:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 150), TEXT_FONT, 1, TEXT_COLOR, 2)

def handle_key(key, continuous_mode, frame, frame_num):
    """Process keyboard input. Returns (should_quit, new_continuous_mode)."""
    if key == ord('q'):
        return True, continuous_mode
    elif key == ord('c'):
        continuous_mode = not continuous_mode
        return False, continuous_mode
    elif key == ord('s'):
        filename = f"frame_{frame_num:05d}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Saved {filename}")
        return False, continuous_mode
    return False, continuous_mode

def cleanup(cap, out, frame_count, fps_list):
    """Release resources and print summary."""
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        logger.info(f"Processed {frame_count} frames @ {avg_fps:.1f} FPS average")

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video detection processor',
        epilog='Controls: c=toggle mode | SPACE=next frame | s=save frame | q=quit'
    )
    parser.add_argument('--video', default='data/TownCent.mp4',
                        help='Path to input video')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Detection confidence threshold')
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
    """Main video processing loop with detection and visualization."""
    args = setup_arguments()
    setup_logging(args.verbose)

    try:
        cap, out, props = setup_video(args.video, args.save_output, args.output)
    except ValueError as e:
        logger.error(f"{e}")
        return

    detector = None
    # TODO: Uncomment when detector is ready
    # from detecting.detectors import Detector
    # detector = Detector(confidence_threshold=args.conf)

    continuous_mode = False
    frame_num = 0
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        detections, elapsed = process_frame(frame, detector)
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

        avg_fps = sum(fps_list[-FPS_WINDOW:]) / len(fps_list[-FPS_WINDOW:])
        mode = "CONTINUOUS" if continuous_mode else "STEP"
        draw_frame(frame, avg_fps, frame_num, props['total_frames'], mode, len(detections))

        cv2.imshow('Video Tracking', frame)
        if out:
            out.write(frame)

        key = cv2.waitKey(1 if continuous_mode else 0) & 0xFF
        should_quit, continuous_mode = handle_key(key, continuous_mode, frame, frame_num)
        if should_quit:
            break

    cleanup(cap, out, frame_num, fps_list)

if __name__ == '__main__':
    main()
