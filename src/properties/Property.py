from abc import ABC, abstractmethod

class Property:
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
    def __repr__(self):
        """
        Returns a string representation of the property.
        """ 
        pass

