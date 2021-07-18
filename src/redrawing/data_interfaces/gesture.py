from redrawing.data_interfaces.data_class import Data

class Gesture(Data):
    '''!
        Stores a gesture detection.
    '''
    gesture_name = ["FIVE",
                    "FIST",
                    "OK",
                    "PEACE",
                    "ONE",
                    "TWO",
                    "TRHEE",
                    "FOUR",
                    "UNKNOWN"]
    

    def __init__(self, gesture="UNKNOWN"):
        '''!
            Gesture constructor.
            @param gesture: The gesture name.
        '''
        
        super().__init__()
        
        self._gesture = gesture

    @property
    def gesture(self):
        '''!
            Gesture name.
            @return: The gesture name.
        '''
        return self._gesture

    @gesture.setter
    def gesture(self, gesture):
        '''!
            Gesture name setter.
            @param gesture: The gesture name.
        '''
        self._gesture = gesture
        