import numpy as np

from redrawing.data_interfaces.data_class import Data

class ObjectDetection(Data):
    '''!
        Stores data from a object detection model
    '''

    def __init__(self, bounding_box=np.zeros((2,2), dtype=np.float), frame_id="UNKNOW"):
        self.bounding_box = np.zeros((2,2), dtype=np.float)
        self.frame_id = frame_id