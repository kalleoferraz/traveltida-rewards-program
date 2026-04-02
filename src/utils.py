

class Config:
    def __init__(self, dictionary):
        self.dict2obj(dictionary)
    
    def dict2obj(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def clip_outliers(serie, std_multiplier=3):
  return serie[serie < serie.mean() + std_multiplier * serie.std()]