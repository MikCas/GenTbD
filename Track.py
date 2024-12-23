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

    _MAXLOSTCOUNT = 10      # Maximum number of consecutive frames the track can be lost
    _TRAJECTORYMAXSIZE = 5  # Maximum number of detections stored in the trajectory

    def __init__(self, trackId, trackState=TrackState.REMOVED):
        self._id = trackId                                          # Unique track ID

        # TRACK STATE
        self._trackState = trackState                               # Track state
        self._lostCounter = 0                                       # Number of consecutive frames the track has been lost

        # TRACK HISTORY
        self._trajectory = Trajectory(max=Track._TRAJECTORYMAXSIZE)  # Trajectory acts like a LIFO queue
        self._kalmanFilter = KalmanFilter()                         # Kalman filter instance for tracking object state

    def getId(self):
        return self._id
    
    # TRACK STATE
    def getTrackState(self):
        return self._trackState

    def getLostCount(self):
        return self._lostCounter
    
    def resetLostCount(self):
        self._lostCounter = 0
    
    def incrementLostCount(self):
        self._lostCounter += 1

    # Activate a track - REMOVED -> NEW
    def activate(self, frameCount, detection):
        self._trackState = TrackState.NEW
        self.initialiseKalmanFilter(detection)
        self.updateTrajectory(frameCount, detection)

    # Update track state in the case of a match 
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
            self.resetLostCount()
            self.updateKalmanFilter(detection)
            self.updateTrajectory(frameCount, detection)

    # Update track state in the case of no match
    def updateUnmatched(self):
        if self._trackState == TrackState.NEW:
            # NEW -> REMOVED
            self._trackState = TrackState.REMOVED
            self.resetLostCount()
            
        elif self._trackState == TrackState.MATCHED:
            # MATCHED -> LOST
            self._trackState = TrackState.LOST
            self.incrementLostCount()

        elif self._trackState == TrackState.LOST:
            # LOST -> LOST (LOST COUNTER < MAX)
            if self._lostCounter < Track._MAXLOSTCOUNT:
                self.incrementLostCount()
            # LOST -> REMOVED (LOST COUNTER > MAX)
            else:
                self._trackState = TrackState.REMOVED
                self.resetLostCount()

    # TRACK HISTORY
    def getTrajectory(self):
        return self._trajectory

    def getMostRecentDetection(self):
        return self._trajectory.getMostRecentDetection()
    
    def updateTrajectory(self, frameCount, detection):
        self._trajectory[frameCount] = detection

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
    
    # TRACK COST
    # Calculate the similarity between the detection and the predicted state of the track/the most recent detection in the trajctory. Then combine the similarity scores (also using NIPP features) to calculate the cost.
    # TODO: Update the cost by incorporating different similarity features
    # TODO: Can also use statistics on the trajectory to calculate cost
    # TODO: Can also use other NIPP features to calculate cost
    def calculateCost(self, detection):
        predictedState = self.getPredictedState()
        similarity = detection.calculateSimilarity(predictedState)
        cost = 1 - similarity
        return cost
    
    def display(self, frame):
        self._trajectory.getMostRecentDetection().display(frame, self.getId())

    def __repr__(self):
        return f"Track(ID={self.getId()}, State={self.getTrackState()}, LostCount={self.getLostCount()}, TrajectoryLength={len(self.getTrajectory())})"

    # Helper method to make it more readable when printing
    def pretty_print(self):
        return f"Track ID: {self._id}\nState: {self._trackState.name}\nLost Count: {self._lostCounter}\nTrajectory Length: {len(self._trajectory)}"