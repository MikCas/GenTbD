# det['key'] - Strict access, raises error if missing
# det.get('key', default) - Safe access, returns default if missing

class Detection:

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