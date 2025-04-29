from .Property import Property

class BoundingBox(Property):
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

    ### SETUP 
    def __init__(self, 
                 x_min:float = 0.0,
                 y_min:float = 0.0,
                 x_max:float = 0.0,
                 y_max:float = 0.0):
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
    def from_corners(cls: int, 
                     x_min: float, 
                     y_min: float, 
                     x_max: float, 
                     y_max: float):
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
    def from_center(cls: int, 
                    center_x: float, 
                    center_y: float, 
                    width: float, 
                    height: float):
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

    ### FUNCTIONS
    @staticmethod
    def iou(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Args:
            box1 (BoundingBox): First bounding box.
            box2 (BoundingBox): Second bounding box.
        Returns:
            float: The IoU value between 0 and 1.
        Raises:
            TypeError: If either box1 or box2 is not an instance of BoundingBox.
        """
        if not isinstance(box1, BoundingBox) or not isinstance(box2, BoundingBox):
            raise TypeError("Both arguments must be instances of BoundingBox. Got {type(box1)} and {type(box2)}")
                
        # Calculate intersection
        x_min = max(box1._x_min, box2._x_min)
        y_min = max(box1._y_min, box2._y_min)
        x_max = min(box1._x_max, box2._x_max)
        y_max = min(box1._y_max, box2._y_max)

        # If there is no intersection, return 0.0
        if x_min >= x_max or y_min >= y_max:
            return 0.0

        # Calculate areas
        intersection_area = (x_max - x_min) * (y_max - y_min)
        union_area = box1.area + box2.area - intersection_area

        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def similarity(self, other: 'BoundingBox') -> float:
        """
        Calculate the similarity between this bounding box and another bounding box.

        Args:
            other (BoundingBox): Another bounding box to calculate similarity with.

        Returns:
            float: The IoU value between 0 and 1.
        """
        return BoundingBox.iou(self, other)

    ### REPRESENTATION
    def xyxy(self): return self._x_min, self._y_min, self._x_max, self._y_max
    def xywh(self): return self._x_min, self._y_min, self.width, self.height
    def cxcywh(self): return self.center, self.width, self.height
    
    def __repr__(self, format: str ='corners'):
        if format == 'corners':
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"
        elif format == 'center':
            return f"BB(cxcywh=[{self.center[0]}, {self.center[1]}, {self.width}, {self.height}])"
        else:
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"