from collections import deque
from pprint import pprint

from Track import Track, TrackState
from Assignment import linearAssignment, calculateCostMatrix, matchThresholdPassed, handleMatch

class Tracker:

    # __slots__ = ['id', 'state', 'detections']

    def __init__(self, maxTracks=100, detectionThreshold=0.3, newTrackThreshold=0.3, matchThreshold=0.1):

        # TRACKER PARAMETERS
        self._frameCount = 0                             # Current frame count

        # TODO: CREATE AN ADAPTIVE TRACKER SUCH THAT THE NUMBER OF TRACKS INCREASES DEPENDING ON THE NUMBER OF AVAILABLE TRACKS - GENERATE NEW TRACKS IF NO RESERVED TRACKS ARE AVAILABLE
        # IN THIS CASE - REMVOED TRACKS IS INITIALLY EMPTY, EVERYTHING IS INITIALLY EMPTY AND TRACKS ARE ADDED AS MORE NEW DETECTIONS ARE FOUND
        self._maxTracks = maxTracks                      # Maximum number of tracks that can be maintained

        # THRESHOLDS 
        self._detectionThreshold = detectionThreshold    # Cascaded LA threshold for detection
        self._newTrackThreshold = newTrackThreshold      # Activation condition for new tracks
        
        # MAYBE HAVE ONE CENTRAL TRACK INSTEAD  
        # ACTIVE AND INACTIVE ALSO MAKE SENSE IN THIS CASE
        # TRACKS - List for each track state
        self._newTracks = []                             
        self._matchedTracks = []                         
        self._lostTracks = []                           
        self._reservedTracks = deque()                        

        self.initialise()

    def initialise(self):
        self._frameCount = 0

        self._newTracks.clear()
        self._matchedTracks.clear()
        self._lostTracks.clear()
        self._reservedTracks.clear()

        for i in range(self._maxTracks):
            self._reservedTracks.append(Track(i + 1))
    
    #GETTERS
    def getNewTracks(self):
        return self._newTracks
    
    def getMatchedTracks(self):
        return self._matchedTracks

    def getLostTracks(self):
        return self._lostTracks
    
    def getReservedTracks(self):
        return self._reservedTracks
    
    def getNumTracks(tracks):
        return len(tracks)
    
    # TRACK MANAGEMENT
    # ACTIVATION
    def activationCondition(self, detection):
        return detection.getConfidenceScore() > self._newTrackThreshold
    
    def activateTrack(self, detection, tracks):
        if(self.activationCondition(detection)):
            activatedTrack = self._reservedTracks.popleft()
            activatedTrack.activate(self._frameCount, detection)
            tracks.append(activatedTrack)

    # Naive assignment of detections to track
    # All detections which pass the activation condition are assigned to a new track
    def activateTracks(self, detections):
        print("---ACTIVATING TRACKS")

        for detection in detections:
            self.activateTrack(detection, self._newTracks)
    
    # PARTITIONING

    def handleMatches(self, tracks, detections, matchedIndexes, matchedTracks):
        for trackIndex, detectionIndex in matchedIndexes:
            track = tracks[trackIndex]
            detection = detections[detectionIndex]
            track.updateMatched(self._frameCount, detection)
            matchedTracks.append(track)
    
    def handleUnmatchedTracks(self, tracks, unmatchedTracksIndexes, unmatchedTracks, reservedTracks):
        for trackIndex in unmatchedTracksIndexes:
            track = tracks[trackIndex]
            track.updateUnmatched()
            if(track.getTrackState() == TrackState.LOST):
                unmatchedTracks.append(track)
            elif(track.getTrackState() == TrackState.RESERVED):
                reservedTracks.append(track)

    def handleUnmatchedDetectons(self, detections, unmatchedDetectionsIndexes, unmatchedDetections):
        for detectionIndex in unmatchedDetectionsIndexes:
            detection = detections[detectionIndex]
            unmatchedDetections.append(detection)

    def partition(self, tracks, detections, matchedIndexes, unmatchedTracksIndexes, unmatchedDetectionsIndexes, matchedTracks, unmatchedTracks, unmatchedDetections):
        
        self.handleMatches(tracks, detections, matchedIndexes, matchedTracks) # Handled since matched tracks are finished in the tracking procedure 

        unmatchedTracks = [tracks[i] for i in unmatchedTracksIndexes] # Not handled since unmatched tracks are not finished in the tracking procedure (they can be matched again)
        unmatchedDetections = [detections[i] for i in unmatchedDetectionsIndexes]

        return unmatchedTracks, unmatchedDetections
    
    def assignment(self, tracks, detectionsList, matchTresholds):
        print("---ASSIGNMENT") 

        matchedTracks, unmatchedDetectionsList = [], [] 
        numStages = len(detectionsList)

        currUnmatchedTracks = tracks

        for i, detections in enumerate(detectionsList):
            matchTreshold = matchTresholds[i]
        
            matchedIndexes, unmatchedTracksIndexes, unmatchedDetectionsIndexes = linearAssignment(currUnmatchedTracks, detections, matchThreshold=matchTreshold)
            unmatchedTracks, unmatchedDetections = self.partition(currUnmatchedTracks, detections, matchedIndexes, unmatchedTracksIndexes, unmatchedDetectionsIndexes, matchedTracks, [], [])

            currUnmatchedTracks = unmatchedTracks
            unmatchedDetectionsList.append(unmatchedDetections)
        
        return matchedTracks, currUnmatchedTracks, unmatchedDetectionsList

    def tracking(self, tracks, detections):
        print("---TRACKING")

        # Partition detections into high and low confidence detections
        highDetections, lowDetections = [], []
        for detection in detections:
            if detection.getConfidenceScore() >= self._detectionThreshold:
                highDetections.append(detection)
            else:
                lowDetections.append(detection)
        
        detectionsList = [highDetections, lowDetections]
        matchTresholds = [0.3, 0.1]

        matchedTracks, unmatchedTracks, unmatchedDetectionsList = self.assignment(tracks, detectionsList, matchTresholds)

        # Handle unmatched detections

    # TRACKING 
    def update(self, frameCount, detections):
        self._frameCount = frameCount

        print("---TRACKING")

        numMatched = self.getNumTracks(self._matchedTracks)
        numLost = self.getNumTracks(self._lostTracks)
        numNew = self.getNumTracks(self._newTracks)

        # If there are no matched or lost tracks, handle any new tracks
        if(numMatched == 0 and numLost == 0):
            # If there are no new tracks, activate new tracks
            if(numNew == 0):
                self.activateTracks(detections)
            else:
                # self.simpleTracking(self._newTracks, detections)
                print("SIMPLE TRACKING")
        else: 
            print("NORMAL TRACKING")
            # self.tracking(detections)

    # DISPLAY 
    def displayTracks(self, frame):
        for track in self._newTracks:
            track.display(frame)

        for track in self._matchedTracks:
            track.display(frame)

        for track in self._lostTracks:
            track.display(frame)
        
    def printTracks(self, tracks):
        for track in tracks:
            print(track)
            # print(track.getMostRecentDetection().getConfidenceScore())



    
    # def processUnmatchedTracks(self, unmatchedTracksIndexes, tracks, lostTracks, reservedTracks):
    #     for trackIndex in unmatchedTracksIndexes:
    #         track = tracks[trackIndex]
    #         track.updateUnmatched()
    #         if(track.getTrackState() == TrackState.LOST):
    #             self.moveElement(trackIndex, tracks, lostTracks)
    #         elif(track.getTrackState() == TrackState.RESERVED):
    #             self.moveElement(trackIndex, tracks, reservedTracks)

    # # Process unmatched detections
    # def processNewTracks(self, detectionIndexes, detections, newTracks):
    #     for detectionIndex in detectionIndexes: 
    #         detection = detections[detectionIndex]
    #         self.activateTrack(detection, newTracks)
    
    # # TRACKING PROCEDURE
    # def partitionDetections(self, detections):
    #     highDetections = []
    #     lowDetections = []

    #     for detection in detections:
    #         if detection.getConfidenceScore() >= self._detectionThreshold:
    #             highDetections.append(detection)
    #         else:
    #             lowDetections.append(detection)

    #     return highDetections, lowDetections

    
    # def simpleTracking(self, tracks, detections):
    #     print("---SIMPLE TRACKING") 
    #     newTracksBuffer, matchedTracksBuffer, lostTracksBuffer = [], [], []

    #     matchedIndexes, unmatchedTracksIndexes, unmatchedDetectionsIndexes = linearAssignment(tracks, detections, matchThreshold=0.3)

    #     self.processMatched(matchedIndexes, self._newTracks, detections, matchedTracksBuffer)
    #     self.processUnmatchedTracks(unmatchedTracksIndexes, self._newTracks, lostTracksBuffer, self._reservedTracks)
    #     self.processNewTracks(unmatchedDetectionsIndexes, detections, newTracksBuffer)

    #     self._newTracks = newTracksBuffer
    #     self._matchedTracks = matchedTracksBuffer
    #     self._lostTracks = lostTracksBuffer

    # def tracking(self, detections):
    #     print("---TRACKING") 

    #     newTracksBuffer, matchedTracksBuffer, lostTracksBuffer = [], [], []
    #     matchableTracks = self._matchedTracks + self._lostTracks

    #     # THIS CAN CHANGE - IN THE CASE THAT WE DO NOT WANT TO PARTITION DETECTIONS IN TERMS OF CONFIDENCE SCORE
    #     detections1, detections2 = self.partitionDetections(detections)

    #    # Stage 1: Match detections1 (High Scoring Detection) with matchable tracks (MATCHED/LOST Tracks)
    #     matched1Indexes, unmatchedTracks1Indexes, unmatchedDetections1Indexes = linearAssignment(matchableTracks, detections1)

    #     # HERE CREATE A PARTITION SET FOR MATCHABLE TRACKS - SO THAT UNMATCHEDTRACKS1 = MATCHABLETRACKS.PARTITION1 OR SOMETHIGN LIKE THAT 
    #     # ALSO FILTER THE NSTATE OF THE TRACK TO ONLY HAVE MATCHED TRACKS - SO LOST TRACKS ARE LOST AFTER FIRST ASSIGNMENT
    #     unmatchedTracks1 = []
    #     for trackIndex in unmatchedTracks1Indexes:  
    #         self.moveElement(trackIndex, matchableTracks, unmatchedTracks1)
            
    #     unmatchedDetections1 = []
    #     for detectionIndex in unmatchedDetections1Indexes:
    #         self.moveElement(detectionIndex, detections1, unmatchedDetections1)

    #     # Stage 2: Match unmatchedTracks1 with detections2 (Low Scoring Detection)
    #     matched2Indexes, unmatchedTracks2Indexes, unmatchedDetections2Indexes = linearAssignment(unmatchedTracks1, detections2)

    #     # Stage 3: Match activated tracks (NEW Tracks) with remaining unmatched detections from Stage 1 (Unmatched high scoring detections)
    #     matched3Indexes, unmatchedTracks3Indexes, unmatchedDetections3Indexes = linearAssignment(self._newTracks, unmatchedDetections1)

    #     # Track Management
    #     # Process matched tracks
    #     self.processMatched(matched1Indexes, matchableTracks, detections1, matchedTracksBuffer)
    #     self.processMatched(matched2Indexes, unmatchedTracks1, detections2, matchedTracksBuffer)
    #     self.processMatched(matched3Indexes, self._newTracks, unmatchedDetections1, matchedTracksBuffer)

    #     # Process unmatched tracks
    #     self.processUnmatchedTracks(unmatchedTracks2Indexes, unmatchedTracks1, lostTracksBuffer, self._reservedTracks)
    #     self.processUnmatchedTracks(unmatchedTracks3Indexes, self._newTracks, lostTracksBuffer, self._reservedTracks)

    #     # Process new tracks
    #     # If the unmatched detections from the high scoring detections are not empty, create new tracks
    #     # NEED TO SEE IF IT MAKES SENSE TO POPULATE THE RESULTS FROM LINEAR ASSIGNMENT WITH THE INDECES OF EVERYTHING IF THE INPUT IS EMPTY
    #     if(len(unmatchedDetections3Indexes) == 0 and len(unmatchedDetections1) > 0):
    #         unmatchedDetections3Indexes = list(range(len(unmatchedDetections1)))
    #     self.processNewTracks(unmatchedDetections3Indexes, unmatchedDetections1, newTracksBuffer)

    #     self._newTracks = newTracksBuffer
    #     self._matchedTracks = matchedTracksBuffer
    #     self._lostTracks = lostTracksBuffer

    
   