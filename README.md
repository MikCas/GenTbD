# GenTbD
Generalized Tracking-by-Detection in Python 

This is a tracking approach based on a **Detection** System which generates detections given some frame, and propogates these detections to a **Tracking** system which processes these detections in a temporal manner - with the aim of preserving the identity of these detections. 

Features of genTbD:
1. The detection is generalised in the sense that the system can expect ay attribute which would help in better tracking, as long as the cost function for the attribute is also provided. Currently there is only BoundingBox (Position) attributes - TBA: Appearance (using a re-id system) and Keypoints.
2. A more streamlined tracking procedure, explicitly outlining the main features of a tracking procedure in a more intuitive way. Based on a defined track lifecycle.
3. Different ways of incorporating multiple features - TBA: Weighted Sum and Gating Thresholds
4. Easily changable tracking mechanism.

# Components
## Detection
A class which contains the properties of a detection. Good to emphasize the distinciton between **Identity-preserving features** and non-identity **preserving features**. A detection is composed of both and can use either other aid in the tracking process. 

Identity-preserving features include Position, Appearance and Keypoints - they are able to determine the identity of an individual between temporal frames to some extent. Some are better than others but the main cost is computational complexity and ability to determine identity. All Identity-Preserving Features have a cost which is computed for linear assignment, such as IoU, cosine distance. 

Non-Identity-preserving features include confidence score - they may be fused with Identity-preserving features and also help in tracing procedure but alone are not able to give an indicaton of identity. 

## Detector
Currently based on UltraLytics, the only defining feature a detector should have is to be able to output a set of detections. 

## Track
The Track is the preservation of identity, it is defined as a state machine and changes state depending on whether or not it has matched (Preserved Identity) or not (Not Preserved Identity). A track could fail to match due to a variety of reasons - so the state machine takes this into account by having the track be active longer. 

The following are the 4 states of a Track:
1. **Removed** - Not available to Match, when there is a potential for a new track to occur we get a Removed Track
2. **New** - A Track which has potential to Match - but we are not sure if the Track itself is consistent due to potential issues with the Detector. This is a standby state until we make sure that the Track is tracking a correct detection. A track that becomes NEW is said to be activated. 
3. **Matched** - A Track which is consistently Matching, will have potential to match again.
4. **Lost** - A Track which has matched consistently, but has not matched in the current or recent frames - a Lost Track is given a time frame for which it can match again and if it does not match it is removed. This state takes into account and missed Detections.

PROVIDE THE STATE DIAGRAM HERE

The two interesting aspects of this State Diagram is the confident detection system - which determines when a detection is good enough to be a New Track - i.e. it has not matched with the current Tracks but also seems to be a relevant detection (high confidence score). ACTIVATION

Another interesting aspect is the Lost state - we use a counter to determine whether or not a Track has been lost for too long , then it is removed. Issues with this could be if there is an identity switch - then the track is lost forever, however this method is useful for missed detections. This is why the Confident Detection and The Lost mechanism should work together to counteract any identity switches based on a new track. DEACTIVATION

An identity switch based on a matching issue occurs due to the matching algorithm and thus the cost metrics and Tracker.

## Trajectory
Given tracking is a temporal problem, we use a trajectory to hold the previous matched detections of the last n frames. The trajectory has a maximum number of elements and when exceeded removes the last element and adds the new element so as to not overflow with uneccesary aged information. The trajecotry is an ordered dictionary whcih has detections mapped by frame number - indicating the frame when a track matched - with the latest added element being the last matched detection at the corresponding frame.  

## Kalman Filter
Used by a Track to update the position information for prediction future states. 

## Tracker
A Tracker in this context is a Track Manager, which performs association between previously matching tracks and newly obtained detections, then updates the internal information of the Tracker. 

REWORD EVERYTHING HERE AND GO THROUGH IT AGAIN:
The tracker makes use of the Non-Identity Preserving features of Tracks and Detections to have a better idea of who is most likely to match and use the Identity-Preserving features when Matching. 

**The main function of the Tracker is to match the Tracks to Detections at each frame.** 

The Tracker has a number of parameters:

- The maximum number of Tracks
- The Detection Threshold - What is considered a high/low confidence threshold
- The new Track Threshold - What is considered a a good enough confidence score for a new Track.

Moreover, the tracker also has two main lists of Tracks - Active Tracks and Inactive Tracks. The active tracks are those track which have potential to match, while the inactive tracks are all the removed tracks. So the active tracks are the New, Matched and the Lost Tracks. It depends on the user on the best strategy to match these tracks. 

The general procedure is to split tracks or detections based on some property which determines that a track is consistently matching or has a potential to be a good match (i.e.e track age or confidence score) These are non-identity preserving features or properties. 

Then a cascaded linear assignment is performed absed on these properties. 

Some heuristics based on ByteTrack with regards to Matching:

- High Scoring Detections are matched with the Matched and Lost Tracks from the previous frame
- From the remaining Unmatched Tracks - Lost Tracks are classified as Lost again, while the remaining Matched Tracks are given another chance to match with the low-scoring detections. Again the remaining unmatched tracks are now classified as Lost.
- The unmatched Low scoring detections are not considered anymore.
- The unmatched high scoring detections are matched with the New Tracks from the previous frame to see if they have the potential to match again. The New Tracks which do not match are removed
- The Remaining unmatched high-scoring detections from the second round are then determined to see if they can be new tracks.

This cascaded matching approach provides relaiable results and there is a lot of improvement in terms of playing aorund with. This system makes it easy to change different aspect of Tracking to try out different approcahes based on the different properties one might have.

## Cost 
When matching we are performing a linear assignment - We compare the identity-preserving properties of the latest detections in the trajectory of the tracks capable of matching, and new detections. 

These properties could be the IoU distance between bounding boxes, or the cosine distance for Re-ID. In the case of position we can also use priejcted position using the Kalman Filter. We can also fuse scores usign non-identity preserving features such as comibing costs with the confidence score of a detection. There is a lot of variation here which can be explored. 

Combining costs is also another area of interest. We can scale all the different sub-costs to end up ahving a range from 0 to 1 and combine these costs together using weighted sums or gating thresholds. 

Another important notion is to have a matching threshold in a tracker. Just linear assignment is not enough as some tracks and detections may be matched together even though they have a high cost - so a match threshold can be used to better control this. 





