from .AbstractProperty import AbstractProperty

class Keypoints(AbstractProperty):
    """
    A class representing a set of keypoints in 2D space.
    Each keypoint is represented by its (x, y) coordinates.
    """

    ### ATTRIBUTES
    __slots__ = ['_keypoints']

    @property
    def num_keypoints(self): return len(self._keypoints)
    @property
    def keypoints(self): return self._keypointss