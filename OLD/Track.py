from enum import Enum
import numpy as np

from properties.BoundingBox import BoundingBox
from Detection import Detection 
from Trajectory import Trajectory
from KalmanFilter import KalmanFilter

class TrackState(Enum):
    NEW = 0     
    MATCHED = 1 
    LOST = 2
    REMOVED = 3

class Track:
    MaxLostCount = 10      # Maximum number of consecutive frames the track can be lost
    TrajectoryMaxSize = 5  # Maximum number of detections stored in the trajectory

    def __init__(self, trackId, trackState=TrackState.REMOVED):
        self._id = trackId                                          # Unique track ID
        self._trackState = trackState                               # Track state
        self._lostCounter = 0                                       # Number of consecutive frames the track has been lost
        self._trajectory = Trajectory(max=Track.TrajectoryMaxSize)  # Trajectory acts like a LIFO queue
        self._kalmanFilter = KalmanFilter()                         # Kalman filter instance for tracking object state

    # Getters
    def getId(self):
        return self._id

    def getTrackState(self):
        return self._trackState

    def getLostCount(self):
        return self._lostCounter

    def getTrajectory(self):
        return self._trajectory
    
    def resetLostCounter(self):
        self._lostCounter = 0
    
    def incrementLostCounter(self):
        self._lostCounter += 1
    
    # Trajectory
    def updateTrajectory(self, frameCount, detection):
        self._trajectory[frameCount] = detection

    # Kalman Filter
    def initialiseKalmanFilter(self, detection):
        self._kalmanFilter.initialise(detection.getBoundingBox())
        self._kalmanFilter.predict()

    def updateKalmanFilter(self, detection):
        self._kalmanFilter.update(detection.getBoundingBox())
        self._kalmanFilter.predict()
    
    def getPredictedState(self):
        prediction = self._kalmanFilter.getState()
        width = prediction[2]
        height = prediction[3]

        predictedState = BoundingBox.fromCorners(prediction[0], prediction[1], prediction[0] + width, prediction[1] + height)
        return predictedState

    # Track Management
    # Activate a removed track from the inactive tracks
    def activate(self, frameCount, detection):
        self._trackState = TrackState.NEW
        self.initialiseKalmanFilter(detection)
        self.updateTrajectory(frameCount, detection)

    def updateMatched(self, frameCount, detection):
        if self._trackState == TrackState.NEW:
            # NEW -> MATCHED
            self._trackState = TrackState.MATCHED
            self.updateKalmanFilter(detection)
            self.updateTrajectory(frameCount, detection)

        elif self._trackState == TrackState.MATCHED:
            # MATCHED -> MATCHED
            self.updateKalmanFilter(detection)
            self.updateTrajectory(frameCount, detection)
            
        elif self._trackState == TrackState.LOST:
            # LOST -> MATCHED
            self._trackState = TrackState.MATCHED
            self.resetLostCounter()
            self.updateKalmanFilter(detection)
            self.updateTrajectory(frameCount, detection)

    def updateUnmatched(self):
        if self._trackState == TrackState.NEW:
            # NEW -> REMOVED
            self._trackState = TrackState.REMOVED
            self.resetLostCounter()
            
        elif self._trackState == TrackState.MATCHED:
            # MATCHED -> LOST
            self._trackState = TrackState.LOST
            self.incrementLostCounter()

        elif self._trackState == TrackState.LOST:
            # LOST -> LOST (LOST COUNTER < MAX)
            if self._lostCounter < Track.MaxLostCount:
                self.incrementLostCounter()
            # LOST -> REMOVED (LOST COUNTER > MAX)
            else:
                self._trackState = TrackState.REMOVED
                self.resetLostCounter()

    def display(self, frame):
        self._trajectory.getMostRecentDetection().display(frame, self.getId())

    def __repr__(self):
        # return f"Track(ID({self.getId()}), STATE({self.getTrackState()}), LOSTCOUNT({self.getLostCount()}), TRAJ_LEN({len(self.getTrajectory())})"
        return f"({self.getId()}, {self.getTrackState()}, {self.getLostCount()}, {len(self.getTrajectory())})"
