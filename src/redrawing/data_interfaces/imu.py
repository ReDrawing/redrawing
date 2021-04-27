import numpy as np
from scipy.spatial.transform import Rotation as R

from redrawing.data_interfaces.data_class import Data


class IMU(Data):

    def __init__(self, frame_id = "UNKNOW"):
        super().__init__()

        self._frame_id = frame_id

        self._accel = np.zeros(3, dtype=np.float64)
        self._gyro = np.zeros(3, dtype=np.float64)
        self._mag = np.zeros(3, dtype=np.float64)
        self._time = 0.0

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        setted = False

        if isinstance(value, float):
            self._time = value
            setted = True
        elif isinstance(value, int):
            self._time = float(value)
            setted = True

        if not setted:
            raise ValueError("'time' must be a float or integer")

    @property
    def accel(self):
        return self._accel

    @accel.setter
    def accel(self, value):
        setted = False
        
        if isinstance(value, list):
            if len(value) == 3:
                self._accel = np.array(value,dtype =np.float64)
                setted = True
        elif isinstance(value, np.ndarray):
            if value.shape == (3,):
                self._accel = value.astype(np.float64) 
                setted = True

        if not setted:
            raise ValueError("O valor não é nem uma lista 3x1 nem um ndarray (3,)")

    @property
    def gyro(self):
        return self._gyro

    @gyro.setter
    def gyro(self, value):
        setted = False
        
        if isinstance(value, list):
            if len(value) == 3:
                self._gyro = np.array(value,dtype =np.float64)
                setted = True
        elif isinstance(value, np.ndarray):
            if value.shape == (3,):
                self._gyro = value.astype(np.float64)
                setted = True

        if not setted:
            raise ValueError("O valor não é nem uma lista 3x1 nem um ndarray (3,)")

    @property
    def mag(self):
        return self._mag

    @mag.setter
    def mag(self, value):
        setted = False
        
        if isinstance(value, list):
            if len(value) == 3:
                self._mag = np.array(value,dtype =np.float64)
                setted = True
        elif isinstance(value, np.ndarray):
            if value.shape == (3,):
                self._mag = value.astype(np.float64)
                setted = True

        if not setted:
            raise ValueError("O valor não é nem uma lista 3x1 nem um ndarray (3,)")

class Orientation(Data):

    def __init__(self, frame_id = "UNKNOW"):
        super().__init__()
        
        self._frame_id = frame_id

        self._orientation = np.zeros(4, dtype=np.float64)
        self._orientation[3] = 1.0
    
    @property
    def orientation(self):
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        setted = False

        if isinstance(value, R):
            self._orientation = value.as_quat()
            setted = True
        elif isinstance(value, list):
            if len(value) == 4:
                self._orientation = np.array(value, dtype = np.float64)
        elif isinstance(value, np.ndarray):
            if value.shape == (4,):
                self._orientation = value.astype(np.float64)
    
        if not setted:
                raise ValueError("O valor não é nem uma lista 4x1 nem um ndarray (4,) nem um scipy.Rotation")