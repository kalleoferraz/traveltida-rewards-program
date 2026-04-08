

class Config:
    """A configuration class that converts dictionaries into object attributes with nested support."""
    
    def __init__(self, dictionary):
        """
        Initialize Config with a dictionary.
        
        Args:
            dictionary: A dictionary to convert into object attributes.
        """
        self.from_dict(dictionary)
    
    def from_dict(self, dictionary):
        """
        Populate object attributes from a dictionary, converting nested dicts to Config objects.
        
        Args:
            dictionary: A dictionary to convert into object attributes.
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
    
    def to_dict(self):
        """
        Convert the Config object back to a dictionary, recursively converting nested Config objects.
        
        Returns:
            A dictionary representation of the Config object.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result

def clip_outliers(serie, std_multiplier=3):
    """
    Remove outliers from a series by clipping values beyond a standard deviation threshold.
    
    Args:
        serie: A pandas Series to filter.
        std_multiplier: Number of standard deviations to use as the threshold (default: 3).
    
    Returns:
        A filtered Series with outliers removed.
    """
    return serie[serie < serie.mean() + std_multiplier * serie.std()]