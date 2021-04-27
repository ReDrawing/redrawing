from redrawing.data_interfaces.data_class import Data
import numpy as np

class BodyPose(Data):
    '''!
        Data class for body poses messages

        @todo Reorganizar lista de keypoints para evitar repetição. Verificar como lidar com diferentes modelos
    '''

    #List of available keypoints types
    keypoints_names = [
    "HEAD"          ,
    "NOSE"          ,
    "EYE_R"         ,
    "EYE_L"         ,
    "EAR_R"         ,
    "EAR_L"         ,
    "NECK"          ,
    "SHOULDER_R"    ,
    "SHOULDER_L"    ,
    "ELBOW_R"       ,
    "ELBOW_L"       ,
    "WRIST_R"       ,
    "WRIST_L"       ,
    "HAND_R"        ,
    "HAND_L"        ,
    "HAND_THUMB_L"  ,
    "HAND_THUMB_R"  ,
    "SPINE_SHOLDER" ,
    "SPINE_MID"     ,
    "SPINE_BASE"    ,
    "HIP_R"         ,
    "HIP_L"         ,
    "KNEE_R"        ,
    "KNEE_L"        ,
    "ANKLE_R"       ,
    "ANKLE_L"       ,
    "FOOT_L"        ,
    "FOOT_R"        ,

    "THUMB_CMC_L",
    "THUMB_MCP_L",
    "THUMB_IP_L",
    "THUMB_TIP_L",
    "INDEX_FINGER_MCP_L",
    "INDEX_FINGER_PIP_L",
    "INDEX_FINGER_DIP_L",
    "INDEX_FINGER_TIP_L",
    "MIDDLE_FINGER_MCP_L",
    "MIDDLE_FINGER_PIP_L", 
    "MIDDLE_FINGER_DIP_L",
    "MIDDLE_FINGER_TIP_L",
    "RING_FINGER_MCP_L",
    "RING_FINGER_PIP_L",
    "RING_FINGER_DIP_L",
    "RING_FINGER_TIP_L",
    "PINKY_MCP_L",
    "PINKY_PIP_L",
    "PINKY_DIP_L",
    "PINKY_TIP_L",

    "THUMB_CMC_R",
    "THUMB_MCP_R",
    "THUMB_IP_R",
    "THUMB_TIP_R",
    "INDEX_FINGER_MCP_R",
    "INDEX_FINGER_PIP_R",
    "INDEX_FINGER_DIP_R",
    "INDEX_FINGER_TIP_R",
    "MIDDLE_FINGER_MCP_R",
    "MIDDLE_FINGER_PIP_R", 
    "MIDDLE_FINGER_DIP_R",
    "MIDDLE_FINGER_TIP_R",
    "RING_FINGER_MCP_R",
    "RING_FINGER_PIP_R",
    "RING_FINGER_DIP_R",
    "RING_FINGER_TIP_R",
    "PINKY_MCP_R",
    "PINKY_PIP_R",
    "PINKY_DIP_R",
    "PINKY_TIP_R",
    ]

    def __init__(self, pixel_space=False, frame_id="UNKOWN"):
        '''!
            BodyPose constructor

            Parameters:
                @param pixels_space (boolean): True if the keypoints are in camera pixel (2D) space, false if its in 3D metric space
                @param frame_id (string): The name of the coordinate system where keypoints are
        '''
        object.__setattr__(self,"_keypoints",{})
        self._pixel_space : bool = pixel_space
        self._keypoints_names : list = BodyPose.keypoints_names
        self._frame_id : string = frame_id

        pass
    
    
    def add_keypoint(self, name, x, y, z=1.0):
        '''!
            Define the pose of a keypoint

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
                @param x (float): keypoint x position
                @param y (float): keypoint y position
                @param z (float): keypoint z position. Default is 1.0 for pixel space keypoints

        '''

        if name not in self._keypoints_names:
            raise AttributeError("BodyPose has no keypoint "+str(name))


        self._keypoints[name] = [float(x),float(y),float(z)]

        pass

    def __getattr__(self, name):

        if name in {'__getstate__', '__setstate__'}:
            return object.__getattr__(self, name)

        try:
            return self._keypoints[name]
        except KeyError:
            raise AttributeError("BodyPose has no keypoint or attribute "+str(name))

    def __setattr__(self, name, value):
        if not name in BodyPose.keypoints_names:
            return super().__setattr__(name, value)
        else:
            self._keypoints[name] = value

    def get_keypoint(self, name):
        '''!
            Returns the keypoint pose

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
            
            Returns:
                @return keypoint (list of float): keypoints position
        '''

        return self._keypoints[name]

    def del_keypoint(self, name):
        '''!
            Delete a keypoint pose

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
        '''


        del self.keypoints[name]
