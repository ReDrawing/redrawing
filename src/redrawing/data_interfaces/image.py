import numpy as np

from redrawing.data_interfaces.data_class import Data

class Image(Data):
    def __init__(self, frame_id = "UNKNOWN"):
        self.__frame_id = frame_id
        self.__image = np.asarray([],dtype=np.uint8)
    
    def set_image(self, image):
        if isinstance(image, np.ndarray):
            self.__image = image.astype(np.uint8)
    
    def get_image(self):
        return self.__image

    def get_frame_id(self):
        return self.__frame_id