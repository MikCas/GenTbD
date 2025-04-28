from abc import ABC, abstractmethod

class AbstractProperty:
    """
    Abstract class for properties.
    """
    @abstractmethod
    def similarity(self, other) -> float:
        """
        Computes the similarity between this property and another property.
        """
        pass

    @abstractmethod 
    def to_dict(self) -> dict:
        """
        Converts the property to a dictionary representation.
        """
        pass

    @abstractmethod
    def from_dict(self, data:dict) -> None:
        """
        Populates the property from a dictionary representation.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Returns a string representation of the property.
        """ 
        pass

