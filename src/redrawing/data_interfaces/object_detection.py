import numpy as np

from redrawing.data_interfaces.data_class import Data

class ObjectDetection(Data):
    '''!
        Stores data from a object detection model
    '''

    def __init__(self, object_type="", bounding_box=np.zeros((2,2), dtype=np.float), frame_id="UNKNOW"):
        '''!
            Constructor.

            Parameters:
                @param object_type: The type of the detected object.
                @param bounding_box: Bounding box of the object.
                @param frame_id: ID of the frame where the object was detected.
        '''

        super().__init__()
        
        self.bounding_box = np.zeros((2,2), dtype=np.float)
        self.frame_id = frame_id