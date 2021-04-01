from redrawing.data_interfaces.data_class import Data
import numpy as np

class BodyPose(Data):
    '''!
        Data class for body poses messages
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
    "FOOT_R"        ,]

    def __init__(self, pixel_space=False, frame_id="UNKOWN"):
        '''!
            BodyPose constructor

            Parameters:
                @param pixels_space (boolean): True if the keypoints are in camera pixel (2D) space, false if its in 3D metric space
                @param frame_id (string): The name of the coordinate system where keypoints are
        '''

        self.__pixel_space = pixel_space
        self.__keypoints = {}
        self.__keypoints_names = BodyPose.keypoints_names
        self.__frame_id = frame_id

        pass

    def set_keypoint(self, name, x, y, z=1.0):
        '''!
            Define the pose of a keypoint

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
                @param x (float): keypoint x position
                @param y (float): keypoint y position
                @param z (float): keypoint z position. Default is 1.0 for pixel space keypoints

        '''

        if name not in self.__keypoints_names:
            raise Exception


        self.__keypoints[name] = [float(x),float(y),float(z)]

        pass

    def get_keypoint(self, name):
        '''!
            Returns the keypoint pose

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
            
            Returns:
                @return keypoint (list of float): keypoints position
        '''

        return self.__keypoints[name]
