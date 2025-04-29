from typing import List, Optional, Tuple

class Partition:
    """
    This class is used to store and manage matched and unmatched elements between two sets (X and Y). 

    X U Y = MatchedX U MatchedY U UnmatchedX U UnmatchedY

    Attributes:
        matched (List[Tuple], optional): A list of matched pairs between X and Y, Defaults to an empty list.
        unmatched_x (List, optional): A list of unmatched elements from X, Defaults to an empty list.
        unmatched_y (List, optional): A list of unmatched elements from Y, Defaults to an empty list.
    """

    ### ATTRIBUTES
    @property
    def matched(self): return self._matched
    @property
    def unmatched_x(self): return self._unmatched_x
    @property
    def unmatched_y(self): return self._unmatched_y

    ### SETUP 
    def __init__(self, 
                matched: Optional[List[Tuple]] = None,
                unmatched_x: Optional[List] = None, 
                unmatched_y: Optional[List] = None):
        self._matched = matched or []
        self._unmatched_x = unmatched_x or []
        self._unmatched_y = unmatched_y or []

    ### UTILITIES
    def is_empty(self) -> bool:
        """
        Check if the partition is empty.

        Returns:
            bool: True if there are no matched or unmatched elements, False otherwise.
        """
        return len(self._matched) == 0 and len(self._unmatched_x) == 0 and len(self._unmatched_y) == 0
    def num_matched(self) -> int: return len(self._matched)
    def num_unmatched_x(self) -> int: return len(self._unmatched_x)
    def num_unmatched_y(self) -> int: return len(self._unmatched_y)
    def clear(self) -> None:
        self.matched.clear()
        self.unmatched_x.clear()
        self.unmatched_y.clear()
    def __str__(self) -> str:
        return (
            f"PARTITION[M: {self.num_matched()}, Ux: {self.num_unmatched_x()}, Uy: {self.num_unmatched_y()}]"
        )