# GenTbD
Generalized Tracking-by-Detection in Python 

This is a tracking approach based on a **Detection** System which generates detections given some frame, and propogates these detections to a **Tracking** system which processes these detections in a temporal manner - with the aim of preserving the identity of these detections. 

Features of genTbD:
1. The detection is generalised in the sense that the system can expect ay attribute which would help in better tracking, as long as the cost function for the attribute is also provided. Currently there is only BoundingBox (Position) attributes - TBA: Appearance (using a re-id system) and Keypoints.
2. A more streamlined tracking procedure, explicitly outlining the main features of a tracking procedure in a more intuitive way. Based on a defined track lifecycle.
3. Different ways of incorporating multiple features - TBA: Weighted Sum and Gating Thresholds
4. Easily changable tracking mechanism.

## System Components
The system consists of three core components:

1. **Video Processor**: Manages video input and processes each frame.
2. **Detector**: Performs object detection within frames.
3. **Tracker**: Updates and manages tracks based on detected objects.

### 1. Video Processor
The `VideoProcessor` class handles video input, processes frames sequentially, and applies the detection and tracking system to each frame. It can work with video files or live streams. The video processor executed frame-by-frame, to continue to the next frame press `a`, to quit the program press `q`. 

- `process()`: Runs the corresponding process method - START VIDEO PROCESSING
- `process_video()`: Processes video file frame-by-frame.
- `process_stream()`: Processes live streams.
- `process_frame(frame)`: Runs detection/tracking algorithms on a single frame.

#### Example:
```python
from video_processor import VideoProcessor
from detector import Detector
from tracker import Tracker

detector = Detector()
tracker = Tracker()

# Initialize VideoProcessor with the detector, tracker, and video path
video_processor = VideoProcessor(detector, tracker, video_path="video.mp4", is_stream=False)
video_processor.process()  # Start processing the video
```

### 2. Detector/Detections

A `Detection` is a set of properties, with functionality to display the detection. Detections can also be compared using the `calculate_similarity(self, other_bounding_box)` function. 
<!-- 
Identity-preserving features include Position, Appearance and Keypoints - they are able to determine the identity of an individual between temporal frames to some extent. Some are better than others but the main cost is computational complexity and ability to determine identity. All Identity-Preserving Features have a cost which is computed for linear assignment, such as IoU, cosine distance. 

Non-Identity-preserving features include confidence score - they may be fused with Identity-preserving features and also help in tracing procedure but alone are not able to give an indicaton of identity.  -->

Currently there are `Class ID`, `Confidence Score`, and `Boudning Box` properties - and no explicit distinction is made between Identity-Preserving Properties (IPP) and Non-Identity-Preserving Properties NIPP properties. 

- `calculate_similarity(self, other_bounding_box)`: Calculate the similarity between current boudning box and an input bounding box
- `display(self, frame, label, color)`: Output the boudning box of a detection, while also specifying the colour to show uniqueness. 

Note: `Boudning Box` is a IPP since it is used to calculate the similarity between objects.

The `Detector` takes a frame as input, and outputs a list of detections. This will be later implemented as an abstract class in order to use different models, and detect different objects depending on the user. 

- `inference(self, frame)`: Takes a frame as input and outputs detections within the frame
- `inference(self, cls, xyxy, conf)`: Obtain the Detection Properties from the inference results to generate a Detection Object

### 3.1 Track
The track is the **Preservation of Identity**. It is defined as a state machine and chnages state depending on whether it matched (Preserved Identity) or not matched (Not Preserved Identity) at a given frame. 

There are a number of reasons a track could not match in the subsequent frame, to accomodate for these the followign state automata is used.

1. **New** - A Track which has potential to Match - but we are not sure if the Track itself is consistent due to potential issues with the Detector. This is a standby state until we make sure that the Track is tracking a correct detection. A track that becomes NEW is said to be **activated**. 
2. **Matched** - A Track which is consistently Matching, will have potential to match again.
3. **Lost** - A Track which has matched consistently, but has not matched in the current or recent frames - a Lost Track is given a time frame for which it can match again and if it does not match it is removed. This state takes into account false positives and missed Detections.
4. **Reserved** - Not available to Match, when there is a potential for a new track to occur we get a Removed Track. A track which has become Reserved is said to be **deactivated**.

The attributes of a track:
- `id`: The unique identifier for each track
- `track_state`: The current state of the track (NEW, MATCHED, LOST, RESERVED)
- `lost_counter`: Used for `deactivation condition` - keeps track of how many frames a track has been lost (_MAX_LOST_COUNT) 
- `trajectory`: A (LIFO) queue that holds the recent detections for the track, up to a specified maximum number (_TRAJECTORY_MAX_SIZE) 
- `kalman_filter`: A Kalman filter that helps smooth predictions for the object’s state (bounding box) across frames.

#### Components of a Track
-`Track Lifecycle` - Activates -> matched consecutively -> becomes lost -> potentially rematches -> Deactivates 
-`Kalman Filter` 
-`Trajectory` - The trajectory is an ordered dictionary that contains the _TRAJECTORY_MAX_SIZE previously matched detections to the current track.
<!-- The trajecotry is an ordered dictionary whcih has detections mapped by frame number - indicating the frame when a track matched - with the latest added element being the last matched detection at the corresponding frame.   -->
-`Calculating Cost` - The cost combines the different similarity metrics of a detection using different techniques to obtain a cost which determines how dissimilar the recently tracked object (recent element in the trajectory) and an object are. 

### 3.2 Tracker
A Tracker in this context is a Track Manager which performs **Association** between tracks and detections, and also acts as a state manager updating the states of tracks after association. 

The tracker consists of 4 sets each corresponding to a different track state - New, Matched, Lost and Reserved Tracks.

Parameters of the Tracker:
- `Maximum Reserved Tracks` - Total number of tracks
- `Detection Threshold` - Confidence Score based value which determines whether or not a detection is high-scoring or not.
- `Activation Threshold` - Activation threshold which determines whether or not a detection is activated based on confidence score
- `Matching Threshold` - Threshold used in association to make sure that only matches with a score lower than the matching threshold are considered a valid match. 

Association is defined as an operation that takes Tracks, Detections and a matching threshold and outputs two disjoint subsets for Tracks - Tm and Tu, and Detections - Dm and Du each corresponding to the matched and unmatched elements between the sets. So Tm and Dm are matched while Tu and Du are unmatched with no correlation between them. 

Association(T, D,m) = Tm, Tu, Dm, Du

**After an association procedure, the matched tracks and detections are left as Matched. The unmatched Tracks are Lost/Removed depending on the deactivation condition or if the tracks are NEW. Unmatched Detections are Activated for any NEW Tracks**

###### 3.2.1 Cost 
Matching is a Linear Assignment - where we compare IPP of most recent detection in trajectory with newly detected objects.

These IPPs can be the IoU distance position, cosine distance for appearance. Also Predicted position to take into account any noise in the system using a Kalman Filter. 

We can fuse these different similarity scores in a multitude of ways. Assuming all similarity metric values are normalised ([0-1]) 
- Weigthed Sums
- Predictors (Like Kalman Filters)
- Gating Thresholds
- Matching Thresholds

##### 3.2.2 Generalised Tracking
Using some attributes of tracks/detections which correspond to matching priority, we can partition the set and perform a cascaded association. 

Some significatn attributes are `Confidence Score` for detections, `Track Age` for tracks. By prioritising association to the highest confidence detections first we are saying that we want them to match first as they are the most likely to match. 

#### BYTETRACK CASCADED ASSIGNMENT
1. High Scoring Detections are matched with the Matched and Lost Tracks from the previous frame
2. From the remaining Unmatched Tracks - Lost Tracks are classified as Lost again, while the remaining Matched Tracks are given another chance to match with the low-scoring detections. Again the remaining unmatched tracks are now classified as Lost.
3. The unmatched Low scoring detections are not considered anymore.
4. The unmatched high scoring detections are matched with the New Tracks from the previous frame to see if they have the potential to match again. The New Tracks which do not match are removed
5. The Remaining unmatched high-scoring detections from the second round are then determined to see if they can be new tracks.

#### VP TODO:
- Have a view method which runs continuously - have a key which toggles between continuous and discreet viewing modes - PRESS 't' to toggle continuous mode

#### DETECTOR TODO:
- Create a distinction between IPP and NIPP properties (as an abstract class), such that IPP properties have some data structure and a corresponding similarity metric. 
- IPPs to add 
    - `Appearance`, n-dimensional vector with cosine similarity
    - `Keypoint`, nx3 dimensional matrix - where (x, y, c) correspond to a position and confidence score of a keypoint. Using the similarity matrix defined in the Alpha Pose.
- Modify the display() method in the Detection class as something is wrong. 
- Create a Detector abstract class to use different detectors

#### TRACK TODO:
- Update the trajectory to have a (start, end) timeline for each consecutive matching. 
- Delete the trajectory of a track when deactivated
- Customisable Activation/Deactivation Conditions


#### TRACKER TODO:
- Using NIPP specifically for partitioning in cascaded assignment
- Remove Reserved Queue and implement a simpler no pre-specified amount of trackers to track with. So generate trackers as we go and remove them when they are done. 
-Integrate different cost combination techniques.




