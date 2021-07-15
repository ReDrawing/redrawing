import numpy as np

from redrawing.data_interfaces.data_class import Data

def is_rotation(R):
    '''!
        See if a matrix is a rotation matrix.

        Parameters:
            @param R: The matrix to be checked.

        Returns:
            @return: True if the matrix is a rotation matrix, False otherwise.
    '''
    if np.linalg.norm((R.T @ R) - np.eye(3)) >0.001:
        return False
    if (np.linalg.det(R) -1) > 0.001:
        return False
    
    return True

class Frame_TF(Data):
    '''!
        This class represents a transform between two frames.
    '''


    def __init__(self, frame_origin="UNKNOWN", frame_destiny="UNKNOWN", R=np.eye(3,dtype=np.float64), t = np.zeros(3,dtype=np.float64)):
        '''!
            Constructor of the class Frame_TF.

            Parameters:
                @param frame_origin: The name of the origin frame.
                @param frame_destiny: The name of the destiny frame.
                @param R: The rotation matrix.
                @param t: The translation vector.
        '''

        super().__init__()
        
        if not is_rotation(R):
            raise Exception("R must be a rotation matrix")


        self._R = R
        self._t = t
        self._frame_origin = frame_origin
        self._frame_destity = frame_destiny 

    

    @property
    def R(self):
        '''!
            Getter of the rotation matrix.
        '''
        return self.R

    @R.setter
    def R(self, value):
        '''!
            Setter of the rotation matrix.

            Parameters:
                @value: The new rotation matrix, must be a 3x3 special orthogonal matrix.
        '''

        if isinstance(value, list):
            value = np.array(list,dtype=np.float64)
        
        if not is_rotation(value):
            raise Exception("R must be a rotation matrix")

        self._R = value
    