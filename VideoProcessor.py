import cv2
import logging
import sys

class VideoProcessor:
    def __init__(self, detector, tracker, videoPath=None, isStream=False):
        
        self._currentFrame = 1                                         # Current frame number
        self._isStream = isStream                                      # Flag to determine if video is a stream or not

        self._detector = detector
        self._tracker = tracker

        # Video Properties  
        if self._isStream:
            self._cap = cv2.VideoCapture(0)                            # Camera capture object
        else:
            self._cap = cv2.VideoCapture(videoPath)                    # Video capture object
        if not self._cap.isOpened():
            sys.exit(1)

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)                     # Frames per second  
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Frame width 
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # Frame height  
        self._frameCount = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video

        detector.setImgSize((self._height, self._width))

    def __del__(self):
        self._cap.release()
    
    def incrementFrame(self):
        self._currentFrame += 1
    
    def getCurrentFrame(self):
        return self._currentFrame

    def processVideo(self):
        while self._currentFrame < self._frameCount:
            print("---PROCESSING FRAME " + str(self._currentFrame))
            result, frame = self._cap.read()

            if not result:
                # ERROR READING FRAME
                break

            # PROCESS FRAME HERE
            detections = self._detector.inference(frame)
            # for detection in detections:
            #     detection.display(frame)
            
            self._tracker.update(self._currentFrame, detections)
            self._tracker.displayTracks(frame)

            cv2.imshow('Frame', frame)

            # CONTINUE OR EXIT
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                self.incrementFrame()
                print("----------------------------------------------------------------------")
                continue
            if key == ord('q'):
                break
        return

    def processStream(self):
        while True:
            print("---FRAME " + str(self._currentFrame))
            result, frame = self._cap.read()

            if not result:
                # ERROR READING FRAME
                break

            # PROCESS FRAME HERE
            detections = self._detector.inference(frame)
            for detection in detections:
                detection.display(frame)
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.incrementFrame()
        return
    
    def process(self):
        if self._isStream:
            self.processStream()
        else:
            self.processVideo()


