import cv2
import logging

class VideoProcessor:
    def __init__(self, detector, tracker, video_path=None, is_stream=False, continuous_mode=False, display_tracks_mode='state', logger=None):
        
        # INITIALISE LOGGER, OTHERWISE USE DEFAULT LOGGER
        self._logger = logger if logger else logging.getLogger(__name__)
        
        # SYSTEM PROPERTIES
        self._current_frame = 1                                          # CURRENT FRAME NUMBER
        self._is_stream = is_stream                                      # FLAG TO DETERMINE IF VIDEO IS A STREAM
        self._continuous_mode = continuous_mode                          # FLAG TO DETERMINE IF VIDEO PROCESSING IS IN CONTINUOUS MODE
        self._display_tracks_mode = display_tracks_mode                  # MODE TO DISPLAY TRACKS - 'state' OR 'id'
        self._detector = detector
        self._tracker = tracker

        # VIDEO CAPTURE OBJECTS - VIDEO OR STREAM 
        self._cap = self.initialise_video_source(video_path)

        # VIDEO PROPERTIES
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)                      # FRAMES PER SECOND
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # FRAME WIDTH
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # FRAME HEIGHT
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) # TOTAL NUMBER OF FRAMES (IF VIDEO)

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

        detections = self._detector.detect(frame)
        # self._detector.display_detections(detections, frame)

        self._tracker.update(self._current_frame, detections)
        self._tracker.display_tracks(frame, mode=self._display_tracks_mode)

        cv2.imshow('Frame', frame)
        return

    # PROCESS VIDEO - FRAME BY FRAME
    def process_video(self):

        # cv2.namedWindow("TRACKING", cv2.WINDOW_NORMAL)
        
        while self._current_frame < self._frame_count:
            self._logger.info(f"----PROCESSING FRAME: {self._current_frame}")
            result, frame = self._cap.read()

            if not result:
                self._logger.error("----ERROR READING FRAME")
                break

            self.process_frame(frame)

            # cv2.imshow("Video Processing", frame)

            # HANDLE KEY EVENTS
            key = cv2.waitKey(1) & 0xFF if self._continuous_mode else cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):                                                 # 'a' - MANUAL ADVANCE
                self._continuous_mode = False
                self.increment_frame()
            elif key == ord('c'):                                               # 'c' - TOGGLE CONTINUOUS MODE
                self._continuous_mode = not self._continuous_mode
                mode = "CONTINUOUS" if self._continuous_mode else "STEPPED"
                self._logger.info(f"----TOGGLED TO {mode} MODE")
            elif key == ord('q'):                                               # 'q' - QUIT PROCESSING
                self._logger.info("----EXITING VIDEO PROCESSING")
                break
            elif self._continuous_mode:                                         # AUTO-ADVANCE IN CONTINUOUS MODE
                self.increment_frame()

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

    


