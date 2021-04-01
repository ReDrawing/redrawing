from abc import ABC
import json
from json import JSONEncoder
import time

import numpy as np



def encoder(obj):
    '''!
        Encode a Data class to JSON

        Implements encoding of numpy data types. Other data types are returned as dicts

        Parameters:
            obj (Object): object to be encoded
        
        Returns
            @returns encoded object
    '''

    if isinstance(obj, np.integer):
            return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    return obj.__dict__

class Data(ABC):
    '''!
        Base class for data messages classes
    '''

    def __init__(self):
        pass

    def toJSON(self):
        '''!
            Convert object to JSON
        '''

        self.__time = time.time()
        return json.dumps(self, default=lambda o: encoder(o))
    
    def toMessage(self):
        '''!
            Converct object to JSON encoded ready to send message
        '''

        return self.toJSON().encode("utf-8")