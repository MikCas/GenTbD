import numpy as np
from scipy.optimize import linear_sum_assignment

# CONDITION TO ACCEPT MATCH GIVEN COST 
def match_condition(cost, match_threshold):
    return cost <= match_threshold

# GENERATE COST MATRIX 
def calculate_cost_matrix(x_list, y_list):
    len_x_list = len(x_list)
    len_y_list = len(y_list)

    cost_matrix = np.ones((len_x_list, len_y_list))

    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            cost_matrix[i, j] = x.calculate_cost(y)
    
    return cost_matrix

# PERFORM LINEAR ASSIGNMENT
def linear_assignment(x_list, y_list, match_threshold=0.3):

    matched_x = []
    matched_y = []
    unmatched_x = []
    unmatched_y = []

    num_x_list = len(x_list) 
    num_y_list = len(y_list)

    if num_x_list == 0 or num_y_list == 0:
        return [], [], x_list, y_list
    
    cost_matrix = calculate_cost_matrix(x_list, y_list)
    matched_x_indexes, matched_y_indexes = linear_sum_assignment(cost_matrix)

    # TODO: POTENTIAL OPTIMISATION HERE BY GOING THROUGH THE LARGEST LIST FIRST AND PARTITIONING, THEN THE SMALLEST LIST
    # MATCHED_X AND MATCHED_Y ARE EQUAL IN SIZE
    for x_index, y_index in zip(matched_x_indexes, matched_y_indexes):
        x = x_list[x_index]
        y = y_list[y_index]

        if match_condition(cost_matrix[x_index][y_index], match_threshold):
            matched_x.append(x)
            matched_y.append(y)
        else:
            unmatched_x.append(x)
            unmatched_y.append(y)
        
    # UNMATCHED_X 
    for x_index in range(num_x_list):
        if x_index not in matched_x_indexes:
            unmatched_x.append(x_list[x_index])

    # UNMATCHED_Y
    for y_index in range(num_y_list):
        if y_index not in matched_y_indexes:
            unmatched_y.append(y_list[y_index])

    # unmatched_x.extend([x_list[i] for i in unmatched_tracks_indexes] if unmatched_tracks_indexes else [])
    # unmatched_detections = [detections[i] for i in unmatched_detections_indexes] if unmatched_detections_indexes else []

    return matched_x, matched_y, unmatched_x, unmatched_y
