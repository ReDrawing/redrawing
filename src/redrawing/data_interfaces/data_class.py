import json
from json import JSONEncoder
import time

import numpy as np

#https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


def test(obj):
    if isinstance(obj, np.integer):
            return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    return obj.__dict__

class Data:
    def __init__(self):
        pass

    def toJSON(self):
        self.__time = time.time()
        return json.dumps(self, default=lambda o: test(o))
    
    def toMessage(self):
        return self.toJSON().encode("utf-8")