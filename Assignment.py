import numpy as np
from scipy.optimize import linear_sum_assignment

import Track, Detection 

# LINEAR ASSIGNMENT
# Generate cost matrix for tracks(rows) and detections(columns)
def calculateCostMatrix(tracks, detections):
    numTracks = len(tracks)
    numDetections = len(detections)
    n = max(numTracks, numDetections)

    costMatrix = np.ones((n, n))

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            costMatrix[i, j] = track.calculateCost(detection)
    
    return costMatrix

# Check if the cost is less than the match threshold
def matchThresholdPassed(cost, matchThreshold):
    return cost <= matchThreshold

def handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections):
    if(matchPassed):
        matchedTracksDetections.append([trackIndex, detectionIndex])
    else:
        unmatchedTracks.append(trackIndex)
        unmatchedDetections.append(detectionIndex)

def linearAssignment(tracks, detections, matchThreshold=0.3):
    
    numTracks = len(tracks) 
    numDetections = len(detections)

    if(numTracks == 0 or numDetections == 0):
        return [], [], []
    
    costMatrix = calculateCostMatrix(tracks, detections)
    rowIndices, colIndices = linear_sum_assignment(costMatrix)

    matchedTracksDetections = []
    unmatchedTracks = []
    unmatchedDetections = []

    if(numTracks == numDetections):
        for trackIndex in range(numTracks):
            detectionIndex = colIndices[trackIndex]

            matchPassed = matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
            handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
    
    # If number of tracks is less than number of detections, the cost matrix will generate fake tracks  
    # Since the rowIndices are ordered, all tracks greater than the number of tracks are fake, and so the corresponding detections are unmatched
    elif numTracks < numDetections:
        for trackIndex in range(numTracks):
            detectionIndex = colIndices[trackIndex]
            matchPassed = matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
            handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
        
        for trackIndex in range(numTracks, numDetections):
            detectionIndex = colIndices[trackIndex]
            unmatchedDetections.append(detectionIndex)

    # If number of tracks is greater than number of detections, the cost matrix will generate fake detections
    # Since the colIndices are not ordered, go through all of the tracks and check if they match with a valid detection (i.e. colIndeces[trackIndex] < numDetections)
    else:
        for trackIndex in range(numTracks):
            detectionIndex = colIndices[trackIndex]
            if detectionIndex < numDetections:
                matchPassed = matchThresholdPassed(costMatrix[trackIndex, detectionIndex], matchThreshold)
                handleMatch(matchPassed, trackIndex, detectionIndex, matchedTracksDetections, unmatchedTracks, unmatchedDetections)
            else: 
                unmatchedTracks.append(trackIndex)

    return matchedTracksDetections, unmatchedTracks, unmatchedDetections