from redrawing.data_interfaces.data_class import Data
import numpy as np

class BodyPose(Data):
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
    "FOOT_R"        ,]

    def __init__(self, pixel_space=False, frame_id="UNKOWN"):
        self.__pixel_space = pixel_space
        self.__keypoints = {}
        self.__keypoints_names = BodyPose.keypoints_names
        self.__frame_id = frame_id

        pass

    def set_keypoint(self, name, x, y, z=1.0):
        self.__keypoints[name] = [float(x),float(y),float(z)]

        pass

    def get_keypoint(self, name):
        return self.__keypoints[name]
