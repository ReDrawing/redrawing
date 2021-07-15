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
        self.gesture = gesture
        super().__init__()
        
        self._gesture = gesture