import numpy as np

from redrawing.data_interfaces.data_class import Data

def is_rotation(R):
    if np.linalg.norm((R.T @ R) - np.eye(3)) >0.001:
        return False
    if (np.linalg.det(R) -1) > 0.001:
        return False
    
    return True

class Frame_TF(Data):


    def __init__(self, frame_origin="UNKNOWN", frame_destiny="UNKNOWN", R=np.eye(3,dtype=np.float64), t = np.zeros(3,dtype=np.float64)):
        super().__init__()
        
        if not is_rotation(R):
            raise Exception("R must be a rotation matrix")


        self._R = R
        self._t = t
        self._frame_origin = frame_origin
        self._frame_destity = frame_destiny 

    

    @property
    def R(self):
        return R

    @R.setter
    def R(self, value):
        if isinstance(value, list):
            value = np.array(list,dtype=np.float64)
        
        if not is_rotation(value):
            raise Exception("R must be a rotation matrix")

        self._R = value
    