import logging
import cv2
import sys

class VideoProcessor:
    def __init__(self, detector, tracker, video_path=None, is_stream=False, continuous_mode = False, logger=None):
        
        # INITIALISE LOGGER, OTHERWISE USE DEFAULT LOGGER
        self._logger = logger if logger else logging.getLogger(__name__)
        
        # SYSTEM PROPERTIES
        self._current_frame = 1                                          # Current frame number
        self._is_stream = is_stream                                      # Flag to determine if video is a stream or not
        self._continuous_mode = continuous_mode                          # Flag to determine if video is processed in continuous mode
        self._detector = detector
        self._tracker = tracker

        # VIDEO CAPTURE OBJECTS - VIDEO OR STREAM 
        self._cap = self.initialise_video_source(video_path)

        # VIDEO PROPERTIES
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)                      # Frames per second  
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # Frame width 
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # Frame height  
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video

        # SET IMAGE SIZE FOR DETECTOR
        detector.imgsz = (self._height, self._width)

    def initialise_video_source(self, video_path):
        if self._is_stream:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self._logger.error(f"----ERROR COULD NOT OPEN VIDEO CAPTURE: {video_path}")
            raise ValueError("Video source could not be opened.")
        return cap
    
    # CLEAN UP RESOURCES
    def __del__(self):
        self._cap.release()
    
    def increment_frame(self):
        self._current_frame += 1
    
    def get_current_frame(self):
        return self._current_frame

    # PROCESS A SINGLE FRAME - PERFORM DETECTION AND TRACKING
    def process_frame(self, frame):
        detections = self._detector.inference(frame)
        # self._detector.display_detections(detections, frame)

        self._tracker.update(self._current_frame, detections)
        self._tracker.display_tracks(frame, mode='id')

        cv2.imshow('Frame', frame)
        return

    # PROCESS VIDEO - FRAME BY FRAME
    def process_video(self):
        
        while self._current_frame < self._frame_count:
            self._logger.info(f"----PROCESSING FRAME: {self._current_frame}")
            result, frame = self._cap.read()

            if not result:
                self._logger.error("----ERROR READING FRAME")
                break

            self.process_frame(frame)

            # CONTINUOUS MODE - DOES NOT WORK YET
            # if self._continuous_mode:
            #     key = cv2.waitKey(1) & 0xFF  # Continue playing with short delay
            # else:
            #     key = cv2.waitKey(0) & 0xFF  # Wait for key press in discrete mode

            # CONTINUE OR EXIT VIDEO 
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                self.increment_frame()
                self._logger.info("----CONTINUING TO NEXT FRAME")
                self._logger.info("------------------------------------------------------------")
            # elif key == ord('c'):  # Toggle between continuous and discrete mode
            #     self._continuous_mode = not self._continuous_mode
            #     mode = "continuous" if self._continuous_mode else "discrete"
            #     self._logger.info(f"----TOGGLED TO {mode.upper()} MODE")
            elif key == ord('q'):
                self._logger.info("----EXITING VIDEO PROCESSING")
                break

    # PROCESS STREAM - FRAME BY FRAME
    def process_stream(self):
        while True:
            self._logger.info(f"----PROCESSING FRAME: {self._current_frame}")
            result, frame = self._cap.read()

            if not result:
                self._logger.error("----ERROR READING FRAME")
                break

            self.process_frame(frame)
            
            # EXIT STREAM
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._logger.info("----EXITING STREAM PROCESSING")
                break

            self.increment_frame()
        return
    
    # PROCESS VIDEO OR STREAM
    def process(self):
        if self._is_stream:
            self.process_stream()
        else:
            self.process_video()

    


