import numpy as np

from redrawing.data_interfaces.data_class import Data

class Depth_Map(Data):
    '''!
        Data class for depth maps

        Depth data must be in meters
    '''

    def __init__(self, frame_id = "UNKNOWN", depth = np.zeros((1,1,3),dtype=np.float64)):
        '''!
            Depth_Map constructor

            Parameters:
                @param frame_id (string): the frame where the depth map is measured
                @param depth (np.array, float64): depth map in meters
        '''

        super().__init__()
        
        self._frame_id = frame_id
        self._depth = depth

    @property
    def depth(self):
        '''!
            Gets the image data

            Returns
                @return image (numpy.array): the image data
        '''
        return self._depth
    
    @depth.setter
    def depth(self, depth):
        '''!
            Set the image data

            Parameters:
                @param image: the image data. Must be a numpy.array or a list

        '''

        if isinstance(depth, np.ndarray):
            self._depth = depth.astype(np.float64)
        else:
            self._depth = depth.array(depth, dtype=np.float64)

    @property
    def frame_id(self):
        '''!
            Gets the frame id

            Returns
                @return frame_id (string): the frame where the camera that captured the image is
        '''

        return self._frame_id