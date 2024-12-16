from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from pprint import pprint

from Track import Track, TrackState

class Tracker:

    def __init__(self, maxTracks=100, detectionThreshold=0.3, newTrackThreshold=0.3, matchThreshold=0.1):

        # TRACKER PARAMETERS
        self._frameCount = 0                             # Current frame count

        # TODO: CREATE AN ADAPTIVE TRACKER SUCH THAT THE NUMBER OF TRACKS INCREASES DEPENDING ON THE NUMBER OF AVAILABLE TRACKS - GENERATE NEW TRACKS IF NO REMOVED TRACKS ARE AVAILABLE
        # IN THIS CASE - REMVOED TRACKS IS INITIALLY EMPTY, EVERYTHING IS INITIALLY EMPTY AND TRACKS ARE ADDED AS MORE NEW DETECTIONS ARE FOUND
        self._maxTracks = maxTracks                      # Maximum number of tracks that can be maintained

        # THRESHOLDS 
        self._detectionThreshold = detectionThreshold    # Cascaded LA threshold for detection
        self._newTrackThreshold = newTrackThreshold      # Activation condition for new tracks

        # TRACKS - List for each track state
        self._newTracks = []                             
        self._matchedTracks = []                         
        self._lostTracks = []                           
        self._removedTracks = deque()                        

        self.initialise()
    
    #GETTERS
    def getNewTracks(self):
        return self._newTracks
    
    def getMatchedTracks(self):
        return self._matchedTracks

    def getLostTracks(self):
        return self._lostTracks
    
    def getRemovedTracks(self):
        return self._removedTracks
    
    def getNumRemovedTracks(self):
        return len(self._removedTracks)
    
    def initialise(self):
        self._frameCount = 0

        self._newTracks.clear()
        self._matchedTracks.clear()
        self._lostTracks.clear()
        self._removedTracks.clear()

        for i in range(self._maxTracks):
            self._removedTracks.append(Track(i + 1))

    # LINEAR ASSIGNMENT
    # Generate cost matrix for tracks(rows) and detections(columns)
    def calculateCostMatrix(self, tracks, detections):
        numTracks = len(tracks)
        numDetections = len(detections)
        n = max(numTracks, numDetections)

        costMatrix = np.ones((n, n))

        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                costMatrix[i, j] = track.calculateCost(detection)
        
        return costMatrix
    
    # Check if the cost is less than the match threshold
    def matchThresholdPassed(self, cost, matchThreshold):
        return cost <= matchThreshold
    
    def handleMatch(self, matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections):
        if(matchPassed):
            matchedTracksDetections.append([trackIndex, detectionIndex])
        else:
            unmatchedTracks.append(trackIndex)
            unmatchedDetections.append(detectionIndex)
    
    def linearAssignment(self, tracks, detections, matchThreshold=0.3):
        
        numTracks = len(tracks) 
        numDetections = len(detections)

        if(numTracks == 0 or numDetections == 0):
            return [], [], []
        
        costMatrix = self.calculateCostMatrix(tracks, detections)
        rowIndices, colIndices = linear_sum_assignment(costMatrix)

        matchedTracksDetections = []
        unmatchedTracks = []
        unmatchedDetections = []

        if(numTracks == numDetections):
            for trackIndex in range(numTracks):
                detectionIndex = colIndices[trackIndex]

                matchPassed = self.matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
                self.handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
        
        # If number of tracks is less than number of detections, the cost matrix will generate fake tracks  
        # Since the rowIndices are ordered, all tracks greater than the number of tracks are fake, and so the corresponding detections are unmatched
        elif numTracks < numDetections:
            for trackIndex in range(numTracks):
                detectionIndex = colIndices[trackIndex]
                matchPassed = self.matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
                self.handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
            
            for trackIndex in range(numTracks, numDetections):
                detectionIndex = colIndices[trackIndex]
                unmatchedDetections.append(detectionIndex)

        # If number of tracks is greater than number of detections, the cost matrix will generate fake detections
        # Since the colIndices are not ordered, go through all of the tracks and check if they match with a valid detection (i.e. colIndeces[trackIndex] < numDetections)
        else:
            for trackIndex in range(numTracks):
                detectionIndex = colIndices[trackIndex]
                if detectionIndex < numDetections:
                    matchPassed = self.matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
                    self.handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
                else: 
                    unmatchedTracks.append(trackIndex)

        return matchedTracksDetections, unmatchedTracks, unmatchedDetections

    # TRACK MANAGEMENT
    def moveItem(self, index, fromArray, toArray):
        item = fromArray[index]
        fromArray[index] = None  # Set the value in the original array to None
        toArray.append(item)     # Add to the new array

    # ACTIVATE TRACK (REMOVED -> NEW)
    # TODO: ACTIVATETRACKCONDITION AND ACTIVATETRACK FUNCTIONS CAN BE MERGED
    # TODO: THIS CAN BE UPDATED DEPENDING ON WHAT NIPP TO USE
    def activateTrackCondition(self, detection):
        return detection.getConfidenceScore() >= self._newTrackThreshold
    
    # DEACTIVATE TRACK (LOST -> REMOVED)
    # TODO: THIS CAN BE UPDATED DEPENDING ON WHAT NIPP TO USE
    def deactivateTrackCondition(self, track):
        # return track.getConsecutiveInvisibleCount() >= 10
        return True
    
    # TODO: ADD A CHECK HERE TO MAKE SURE THERE ARE STILL TRACKS LEFT IN THE REMOVEDTRACKS
    def activateTrack(self, detection, tracks):
        activatedTrack = self._removedTracks.popleft()
        activatedTrack.activate(self._frameCount, detection)
        tracks.append(activatedTrack)
    
    def processMatchedTracks(self, matches, tracks, detections, matchableTracks):
        for trackIndex, detectionIndex in matches:
            track = tracks[trackIndex]
            detection = detections[detectionIndex]
            track.updateMatched(self._frameCount, detection)
            self.moveItem(trackIndex, tracks, matchableTracks)

    def processUnmatchedTracks(self, trackIndexes, tracks, matchableTracks, removedTracks):
        for trackIndex in trackIndexes:
            track = tracks[trackIndex]
            track.updateUnmatched()
            if(track.getTrackState() == TrackState.LOST):
                self.moveItem(trackIndex, tracks, matchableTracks)
            elif(track.getTrackState() == TrackState.REMOVED):
                self.moveItem(trackIndex, tracks, removedTracks)

    def processNewTracks(self, detectionIndexes, detections, activatedTracks):
        for detectionIndex in detectionIndexes:
            detection = detections[detectionIndex]
            if detection.getConfidenceScore() >= self._newTrackThreshold:
                self.activateTrack(detection, activatedTracks)


    # CASCADED LINEAR ASSIGNMENT
    def partitionDetections(self, detections):
        highDetections = []
        lowDetections = []

        for detection in detections:
            if detection.getConfidenceScore() >= self._detectionThreshold:
                highDetections.append(detection)
            else:
                lowDetections.append(detection)

        return highDetections, lowDetections

    def update(self, frameCount, detections):
        self._frameCount = frameCount

        if(frameCount == 1):
            self.processInitialFrame(detections)
        else:
            self.processSubsequentFrame(detections)

    def processInitialFrame(self, detections):
        print("Processing initial frame")
        for detection in detections:
            if self.activateTrackCondition(detection):
                self.activateTrack(detection, self._newTracks)

    def processSubsequentFrame(self, detections):
        print("Processing subsequent frame")

        bufferActivatedTracks, bufferMatchableTracks = [], []

        # Partition detections into high and low confidence detections
        detections1, detections2 = self.partitionDetections(detections)

        # if(self._frameCount == 2):
            # matchedIndexes, unmatchedTracksIndexes, unmatchedDetectionsIndexes = self.linearAssignment(self._activatedTracks, detections1)

            # self.processMatchedTracks(matchedIndexes, self._activatedTracks, detections1, bufferMatchableTracks)
            # self.processUnmatchedTracks(unmatchedTracksIndexes, self._activatedTracks, bufferMatchableTracks, self._removedTracks)
            # self.processNewTracks(unmatchedDetectionsIndexes, detections1, bufferActivatedTracks)

            # self._activatedTracks = bufferActivatedTracks
            # self._matchableTracks = bufferMatchableTracks

        # else:
            # # Stage 1: Match detections1 (High Scoring Detection) with matchable tracks (MATCHED/LOST Tracks)
            # matched1Indexes, unmatchedTracks1Indexes, unmatchedDetections1Indexes = self.linearAssignment(self._matchableTracks, detections1)

            # # ALSO FILTER THE NSTATE OF THE TRACK TO ONLY HAVE MATCHED TRACKS - SO LOST TRACKS ARE LOST AFTER FIRST ASSIGNMENT
            # unmatchedTracks1 = []
            # for trackIndex in unmatchedTracks1Indexes:  
            #     self.moveItem(trackIndex, self._matchableTracks, unmatchedTracks1)
                
            # unmatchedDetections1 = []
            # for detectionIndex in unmatchedDetections1Indexes:
            #     self.moveItem(detectionIndex, detections1, unmatchedDetections1)

            # print("UNMATCHED DETECTIONS 1")
            # pprint(unmatchedDetections1)

            # # Stage 2: Match unmatchedTracks1 with detections2 (Low Scoring Detection)
            # matched2Indexes, unmatchedTracks2Indexes, unmatchedDetections2Indexes = self.linearAssignment(unmatchedTracks1, detections2)

            # # Stage 3: Match activated tracks (NEW Tracks) with remaining unmatched detections from Stage 1 (Unmatched high scoring detections)
            # matched3Indexes, unmatchedTracks3Indexes, unmatchedDetections3Indexes = self.linearAssignment(self._activatedTracks, unmatchedDetections1)

            # # Track Management
            # # Process matched tracks
            # self.processMatchedTracks(matched1Indexes, self._matchableTracks, detections1, bufferMatchableTracks)
            # self.processMatchedTracks(matched2Indexes, unmatchedTracks1, detections2, bufferMatchableTracks)
            # self.processMatchedTracks(matched3Indexes, self._activatedTracks, unmatchedDetections1, bufferMatchableTracks)

            # # Process unmatched tracks
            # self.processUnmatchedTracks(unmatchedTracks2Indexes, unmatchedTracks1, bufferMatchableTracks, self._removedTracks)
            # self.processUnmatchedTracks(unmatchedTracks3Indexes, self._activatedTracks, bufferMatchableTracks, self._removedTracks)

            # # Process new tracks
            # # If the unmatched detections from the high scoring detections are not empty, create new tracks
            # # NEED TO SEE IF IT MAKES SENSE TO POPULATE THE RESULTS FROM LINEAR ASSIGNMENT WITH THE INDECES OF EVERYTHING IF THE INPUT IS EMPTY
            # if(len(unmatchedDetections3Indexes) == 0 and len(unmatchedDetections1) > 0):
            #     unmatchedDetections3Indexes = list(range(len(unmatchedDetections1)))
            # self.processNewTracks(unmatchedDetections3Indexes, unmatchedDetections1, bufferActivatedTracks)

            # self._activatedTracks = bufferActivatedTracks
            # self._matchableTracks = bufferMatchableTracks

            # pprint(self._matchableTracks)
            # pprint(self._activatedTracks)

    def display(self, frame):
        for track in self._matchableTracks:
            if track.getTrackState() == TrackState.MATCHED:
                track.display(frame)
        
        for track in self._activatedTracks:
            track.display(frame)


    
   