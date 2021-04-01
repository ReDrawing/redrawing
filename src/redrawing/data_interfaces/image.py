import numpy as np

from redrawing.data_interfaces.data_class import Data

class Image(Data):
    '''!
        Data class for image messages
    '''

    def __init__(self, frame_id = "UNKNOWN"):
        '''!
            Image constructor

            Parameters:
                @param frame_id (string): the frame where the camera that captured the image is
        '''

        self.__frame_id = frame_id
        self.__image = np.asarray([],dtype=np.uint8)
    
    def set_image(self, image):
        '''!
            Set the image data

            Parameters:
                @param image: the image data. Must be a numpy.array or a list

        '''

        if isinstance(image, np.ndarray):
            self.__image = image.astype(np.uint8)
        else:
            self.__image = image.array(image, dtype=np.uint8)
    
    def get_image(self):
        '''!
            Gets the image data

            Returns
                @return image (numpy.array): the image data
        '''

        return self.__image

    def get_frame_id(self):
        '''!
            Gets the frame id

            Returns
                @return frame_id (string): the frame where the camera that captured the image is
        '''
        return self.__frame_id