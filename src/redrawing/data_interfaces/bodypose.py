from redrawing.data_interfaces.data_class import Data
import numpy as np

class BodyPose(Data):
    keypoint_dict = {
    "HEAD"          : 1,
    "NOSE"          : 2,
    "EYE_R"         : 3,
    "EYE_L"         : 4,
    "EAR_R"         : 5,
    "EAR_L"         : 6,
    "NECK"          : 7,
    "SHOULDER_R"    : 8,
    "SHOULDER_L"    : 9,
    "ELBOW_R"       : 10,
    "ELBOW_L"       : 11,
    "WRIST_R"       : 12,
    "WRIST_L"       : 13,
    "HAND_R"        : 14,
    "HAND_L"        : 15,
    "HAND_THUMB_L"  : 16,
    "HAND_THUMB_R"  : 17,
    "SPINE_SHOLDER" : 18,
    "SPINE_MID"     : 19,
    "SPINE_BASE"    : 20,
    "HIP_R"         : 21,
    "HIP_L"         : 22,
    "KNEE_R"        : 23,
    "KNEE_L"        : 24,
    "ANKLE_R"       : 25,
    "ANKLE_L"       : 26,
    "FOOT_L"        : 28,
    "FOOT_R"        : 29,
    }
    

    def __init__(self, pixel_space=False, frame_id="UNKOWN"):
        self.__pixel_space = pixel_space
        self.__keypoints = np.zeros((29,3), dtype=np.float64)
        self.__keypoint_dict = BodyPose.keypoint_dict
        self.__frame_id = frame_id

        pass

    def set_keypoint(self, index, x, y, z=1.0):
        self.__keypoints[index] = [float(x),float(y),float(z)]

        pass

    def get_keypoint(self, index):
        return self.__keypoints[index]
