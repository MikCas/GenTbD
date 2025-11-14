# det['key'] - Strict access, raises error if missing
# det.get('key', default) - Safe access, returns default if missing

class Detection:
    """Dictionary-like container for detection properties.

    Stores detection information such as bounding boxes, class IDs, confidences,
    and other attributes in a flexible key-value structure.

    Usage:
        det = Detection({'bbox': BoundingBox(...), 'class_id': 1, 'confidence': 0.95})
        bbox = det['bbox']  # Strict access
        score = det.get('confidence', 0.0)  # Safe access with default
    """

    def __init__(self, properties=None):
        self._props = properties if properties is not None else {}

    def __setitem__(self, key, value):
        self._props[key] = value

    def __getitem__(self, key):
        return self._props[key]

    def __contains__(self, key):
        return key in self._props

    def get(self, key, default=None):
        return self._props.get(key, default)

    def keys(self):
        return self._props.keys()

    def __repr__(self):
        return f"Detection({self._props})"