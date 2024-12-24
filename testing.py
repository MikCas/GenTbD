# Assuming we have these imports based on your Track class definition
from Detection import Detection
from properties.BoundingBox import BoundingBox
from KalmanFilter import KalmanFilter
from Trajectory import Trajectory
from Track import Track, TrackState


def test_track():
    # Step 1: Create a dummy detection (with dummy BoundingBox)
    bounding_box = BoundingBox(0, 0, 100, 200)  # Assuming BoundingBox(x1, y1, width, height)
    detection = Detection(0, bounding_box, 0.5)  # Assuming Detection takes BoundingBox
    
    # Step 2: Create a Track
    track = Track(track_id=1)
    print(track.get_trajectory())
    print(f"Track Created: {track}\n")
    
    # Step 3: Activate the Track with the first detection (Frame 1)
    track.activate(frame_count=1, detection=detection)
    print(track.get_trajectory())
    print(f"Track Activated: {track}\n")
    
    
    # Step 4: Add a second detection at Frame 2 (simulating a new frame)
    detection_2 = Detection(0, BoundingBox(10, 20, 110, 210), 0.8)  # New detection
    track.update_matched(frame_count=2, detection=detection_2)
    print(f"Track after frame 2 (Matched): {track}\n")
    
    # Step 5: Simulate a case where the track is lost (update unmatched)
    track.update_unmatched()  # Track will move to LOST state
    print(f"Track after being unmatched: {track}\n")
    
    # Step 6: Simulate the track being matched again after being lost
    track.update_matched(frame_count=3, detection=detection_2)  # Matching again
    print(f"Track after being matched again: {track}\n")
    
    # Step 7: Get and display the most recent detection
    most_recent_detection = track.get_most_recent_detection()
    print(f"Most recent detection: {most_recent_detection}\n")
    
    # Step 8: Get and display the predicted state from Kalman filter
    predicted_state = track.get_predicted_state()
    print(f"Predicted state (BoundingBox): {predicted_state}\n")
    
    # Step 9: Calculate cost for the current detection
    cost = track.calculate_cost(detection_2)
    print(f"Cost calculated: {cost}\n")

# Run the test
if __name__ == "__main__":
    test_track()
