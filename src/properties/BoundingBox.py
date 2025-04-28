from .AbstractProperty import AbstractProperty

class BoundingBox(AbstractProperty):
    """
    A class representing a bounding box in 2D space.
    Note that x increases from left to right (->) and y increases from top to bottom (v).

    The bounding box can be represented in different formats:
    - xyxy: (x_min, y_min, x_max, y_max)
    - xywh: (x_min, y_min, width, height)
    - cxcywh: (center_x, center_y, width, height)
    """

    ### ATTRIBUTES
    __slots__ = ['_x_min', '_y_min', '_x_max', '_y_max']

    @property
    def width(self): return abs(self._x_max - self._x_min)
    @property
    def height(self): return abs(self._y_max - self._y_min)
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return (self._x_min + self._x_max) / 2, (self._y_min + self._y_max) / 2

    ### SETUP METHODS
    def __init__(self, x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0):
        """
        Initialize a BoundingBox instance.

        Args:
            x_min (float): Minimum x-coordinate (left).
            y_min (float): Minimum y-coordinate (top).
            x_max (float): Maximum x-coordinate (right).
            y_max (float): Maximum y-coordinate (bottom).

        Raises:
            ValueError: If any of the coordinates are negative or if x_max < x_min or y_max < y_min.  
        """
        if x_max < x_min:
            raise ValueError("x_max must be greater than or equal to x_min.")
        if y_max < y_min:
            raise ValueError("y_max must be greater than or equal to y_min.")
        
        self._x_min = float(x_min)
        self._y_min = float(y_min)
        self._x_max = float(x_max)
        self._y_max = float(y_max)
   
    @classmethod 
    def from_corners(cls, x_min, y_min, x_max, y_max):
        """
        Create a BoundingBox instance from corner coordinates.
        Args:
            x_min (float): Minimum x-coordinate (left).
            y_min (float): Minimum y-coordinate (top).
            x_max (float): Maximum x-coordinate (right).
            y_max (float): Maximum y-coordinate (bottom).
        Returns:
            BoundingBox: A new instance of BoundingBox.
        """
        return cls(float(x_min), float(y_min), float(x_max), float(y_max))

    @classmethod 
    def from_center(cls, center_x, center_y, width, height):
        """
        Create a BoundingBox instance from center coordinates and dimensions.
        Args:
            center_x (float): X-coordinate of the center.
            center_y (float): Y-coordinate of the center.
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
        Returns:
            BoundingBox: A new instance of BoundingBox.
        """
        x_min = float(center_x - width / 2)
        y_min = float(center_y - height / 2)
        x_max = float(center_x + width / 2)
        y_max = float(center_y + height / 2)
        return cls(x_min, y_min, x_max, y_max)

    ### SIMILARITY
    def intersect(self, other):
        """
        Calculate the intersection of two bounding boxes.
        Args:
            other (BoundingBox): Another bounding box to intersect with.
        Returns:
            BoundingBox: A new BoundingBox instance representing the intersection.
            None: If there is no intersection.
        """
        if not isinstance(other, BoundingBox):
            raise TypeError("The other object must be an instance of BoundingBox.") 
        x_min, y_min = max(self._x_min, other._x_min), max(self._y_min, other._y_min)
        x_max, y_max = min(self._x_max, other._x_max), min(self._y_max, other._y_max)

        # IF VALID INTERSECTION
        if x_min < x_max and y_min < y_max:
            return BoundingBox(x_min, y_min, x_max, y_max)
        
        return None

    def similarity(self, other):
        """
        Calculate the similarity between this bounding box and another bounding box.
        Args:
            other (BoundingBox): Another bounding box to calculate similarity with.
        Returns:    
            float: The IoU value between 0 and 1.
        """
        intersection = self.intersect(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area

    ### OUTPUT
    def xyxy(self): return self._x_min, self._y_min, self._x_max, self._y_max
    def xywh(self): return self._x_min, self._y_min, self.width, self.height
    def cxcywh(self):
        center_x, center_y = self.center
        return center_x, center_y, self.width, self.height

    def to_dict(self) -> dict:
        return {
            "x_min": self._x_min,
            "y_min": self._y_min,
            "x_max": self._x_max,
            "y_max": self._y_max
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            x_min=data.get("x_min", 0.0),
            y_min=data.get("y_min", 0.0),
            x_max=data.get("x_max", 0.0),
            y_max=data.get("y_max", 0.0)
        )
    
    def __repr__(self, format='corners'):
        if format == 'corners':
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"
        elif format == 'center':
            return f"BB(cxcywh=[{self.center[0]}, {self.center[1]}, {self.width}, {self.height}])"
        else:
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"