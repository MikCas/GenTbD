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
