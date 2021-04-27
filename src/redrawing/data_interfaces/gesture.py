from redrawing.data_interfaces.data_class import Data

class Gesture(Data):
    gesture_name = ["FIVE",
                    "FIST",
                    "OK",
                    "PEACE",
                    "ONE",
                    "TWO",
                    "TRHEE",
                    "FOUR",]
    

    def __init__(self, gesture=None):
        super().__init__()
        
        self._gesture = gesture