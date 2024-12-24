import numpy as np
from scipy.optimize import linear_sum_assignment

# GENERATE COST MATRIX FOR TRACKS(ROWS) AND DETECTIONS(COLUMNS)
def calculate_cost_matrix(tracks, detections):
    num_tracks = len(tracks)
    num_detections = len(detections)
    n = max(num_tracks, num_detections)

    cost_matrix = np.ones((n, n))

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            cost_matrix[i, j] = track.calculate_cost(detection)
    
    return cost_matrix

# CHECK IF THE COST PASSES THE MATCH THRESHOLD
def match_threshold_passed(cost, match_threshold):
    return cost <= match_threshold

# HANDLE MATCHED TRACKS AND DETECTIONS
# TODO: MAYBE HERE IT MAKES SENSE TO UPDATE TRACKS WITH DETECTIONS AND ONLY UPDATE THE TRACKS SINCE THE DETECTIONS WON'E BE NEEDED ANYMORE AFTER MATCHING THE TRACKS ANYWAYS
def handle_match(match_passed, track_index, detection_index, matched_tracks_detections, unmatched_tracks, unmatched_detections):
    if match_passed:
        matched_tracks_detections.append([track_index, detection_index])
    else:
        unmatched_tracks.append(track_index)
        unmatched_detections.append(detection_index)

# PERFORM LINEAR ASSIGNMENT
def linear_assignment(tracks, detections, match_threshold=0.3):
    num_tracks = len(tracks) 
    num_detections = len(detections)

    if num_tracks == 0 or num_detections == 0:
        return [], [], []
    
    cost_matrix = calculate_cost_matrix(tracks, detections)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_tracks_detections = []
    unmatched_tracks = []
    unmatched_detections = []

    if num_tracks == num_detections:
        for track_index in range(num_tracks):
            detection_index = col_indices[track_index]

            match_passed = match_threshold_passed(cost_matrix[track_index, detection_index], match_threshold)
            handle_match(match_passed, track_index, detection_index, matched_tracks_detections, unmatched_tracks, unmatched_detections)
    
    # If number of tracks is less than number of detections, the cost matrix will generate fake tracks  
    # Since the row_indices are ordered, all tracks greater than the number of tracks are fake, and so the corresponding detections are unmatched
    elif num_tracks < num_detections:
        for track_index in range(num_tracks):
            detection_index = col_indices[track_index]
            match_passed = match_threshold_passed(cost_matrix[track_index, detection_index], match_threshold)
            handle_match(match_passed, track_index, detection_index, matched_tracks_detections, unmatched_tracks, unmatched_detections)
        
        for track_index in range(num_tracks, num_detections):
            detection_index = col_indices[track_index]
            unmatched_detections.append(detection_index)

    # If number of tracks is greater than number of detections, the cost matrix will generate fake detections
    # Since the col_indices are not ordered, go through all of the tracks and check if they match with a valid detection (i.e. col_indices[track_index] < num_detections)
    else:
        for track_index in range(num_tracks):
            detection_index = col_indices[track_index]
            if detection_index < num_detections:
                match_passed = match_threshold_passed(cost_matrix[track_index, detection_index], match_threshold)
                handle_match(match_passed, track_index, detection_index, matched_tracks_detections, unmatched_tracks, unmatched_detections)
            else: 
                unmatched_tracks.append(track_index)

    return matched_tracks_detections, unmatched_tracks, unmatched_detections

