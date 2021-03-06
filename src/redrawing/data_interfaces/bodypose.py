from redrawing.data_interfaces.data_class import Data
import numpy as np
import time as tm
    

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
    "EYE_L_INNER"   ,
    "EYE_R_INNER"   ,
    "EYE_L_OUTER"   ,
    "EYE_R_OUTER"   ,

    "EAR_R"         ,
    "EAR_L"         ,

    "MOUTH_R",
    "MOUTH_L",

    "NECK"          ,
    "SHOULDER_R"    ,
    "SHOULDER_L"    ,
    "SPINE_SHOULDER" ,
    "SPINE_MID"     ,
    "SPINE_BASE"    ,

    "ELBOW_R"       ,
    "ELBOW_L"       ,

    "WRIST_R"       ,
    "WRIST_L"       ,
    "HAND_R"        ,
    "HAND_L"        ,
    
    "HAND_THUMB_L"  ,
    "HAND_THUMB_R"  ,

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

    

    "FOOT_R_INDEX",
    "FOOT_L_INDEX"
    ]

    

    def __init__(self, pixel_space=False, frame_id="UNKOWN", user_id = "UNKOWN", time=-1):
        '''!
            BodyPose constructor

            Parameters:
                @param pixels_space (boolean): True if the keypoints are in camera pixel (2D) space, false if its in 3D metric space
                @param frame_id (string): The name of the coordinate system where keypoints are
                @param user_id (string): The id of the pose corresponding user
                @param time (float): The time of the pose detection
        '''
        if time == -1:
            time = tm.time()

        super().__init__(time=time)

        self._keypoints = {name : np.ones(3)*np.inf for name in BodyPose.keypoints_names}
        self._covariance = {name: np.zeros((3,3)) for name in BodyPose.keypoints_names}
        self._pixel_space  = pixel_space
        self._user_id  = user_id
        self._keypoints_names  = BodyPose.keypoints_names
        self._frame_id  = frame_id

        pass

    @property
    def frame_id(self):
        '''!
            Frame id getter.
        '''
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        '''!
            Frame id setter.
        '''
        if isinstance(value, str):
            self._frame_id = value
    
    @property
    def user_id(self):
        '''!
            User id getter.
        '''
        return self._user_id

    @user_id.setter
    def user_id(self, value):
        '''!
            User id setter.
        '''
        if isinstance(value, str):
            self._user_id = value

    @property
    def time(self):
        '''!
            Time getter.
        '''
        return self._time
    
    def add_keypoint(self, name, x, y, z=1.0, covariance=np.zeros((3,3))):
        '''!
            Define the pose of a keypoint

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
                @param x (float): keypoint x position
                @param y (float): keypoint y position
                @param z (float): keypoint z position. Default is 1.0 for pixel space keypoints
                @param covariance (3x3 matrix): position covariance matrix. Default is identity matrix.
        '''

        if name not in self._keypoints_names:
            raise AttributeError("BodyPose has no keypoint "+str(name))

        if isinstance(covariance, list):
            covariance = np.array(covariance)

        if covariance.shape != (3,3):
            raise Exception("Covariance must be a 3x3 matrix")

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            self._keypoints[name] = np.array([np.inf,np.inf,np.inf])
        else:
            self._keypoints[name] = np.array([float(x),float(y),float(z)])


        self._covariance[name] = covariance
        

    def add_keypoint_array(self, name, array, covariance=np.zeros((3,3))):
        '''!
            Define the pose of a keypoint.

            Parameters:
                @param name (string): the name of the keypoints. Must be in keypoints_names list
                @param array (list of float): list of keypoint x,y,z positions
                @param covariance (3x3 matrix): position covariance matrix. Default is identity matrix.
        '''
        if name not in self._keypoints_names:
            raise AttributeError("BodyPose has no keypoint "+str(name))
        
        if isinstance(covariance, list):
            covariance = np.array(covariance)
        if covariance.shape != (3,3):
            raise Exception("Covariance must be a 3x3 matrix")

        if isinstance(array, np.ndarray):
            self._keypoints[name] = array
        elif isinstance(array,list):
            self._keypoints[name] = np.array(array,dtype=np.float64)
        
        self._covariance[name] = covariance

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

    def get_covariance(self, name):
        '''!
            Return the covariance of a keypoint
        '''
        return self._covariance[name]

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

    def apply_transformation(self, R, t, new_frame_id):
        '''!
            Apply a transformation to the pose.

            Parameters:
                @param R (3x3 matrix): rotation matrix
                @param t (3x1 matrix): translation vector
                @param new_frame_id (string): new frame id
        '''
        for name in self._keypoints:
            if self._keypoints[name] is None:
                continue

            self._keypoints[name] = (R@self._keypoints[name])+t 

        self._frame_id = new_frame_id

    @staticmethod
    def distance(bodypose1, bodypose2):
        '''!
            Compute the distance between two BodyPose objects.
            The distance is computed as the Euclidean distance between the keypoints,
            and only for keypoints that are defined in both bodyposes.
        '''
        dist = 0.0
        count = 0


        for name in BodyPose.keypoints_names:
            kp1 = bodypose1.get_keypoint(name)
            kp2 = bodypose2.get_keypoint(name)
            if (not np.isinf(kp1[0])) and  (not np.isinf(kp2[0])):
                dist += np.linalg.norm(kp2-kp1)
                count += 1

        if count == 0:
            dist = float('inf')

        return dist

class BodyVel(BodyPose):
    '''!
        This class stores the velocity of a body pose.
    '''

    def __init__(self, pixel_space=False, frame_id='UNKOWN', user_id='UNKOWN', time=tm.time()):
        '''!
            BodyVel constructor

            Parameters:
                @param pixels_space (boolean): True if the keypoints are in camera pixel (2D) space, false if its in 3D metric space
                @param frame_id (string): The name of the coordinate system where keypoints are
                @param user_id (string): The id of the body corresponding user
                @param time (float): The time of the velocity detection
        '''

        super().__init__(pixel_space=pixel_space, frame_id=frame_id, user_id=user_id, time=time)

    @classmethod
    def from_bodyposes(cls, bodypose, bodypose_last):
        '''!
            Create a BodyVel object from two BodyPose objects.
            @todo Calcular covariância a partir da covariância das posições
        '''
        
        body_vel = BodyVel(bodypose._pixel_space, bodypose._frame_id, bodypose.user_id, bodypose._time)

        deltaT = bodypose.time - bodypose_last.time

        for name in BodyPose.keypoints_names:
            kp1 = bodypose.get_keypoint(name)
            kp2 = bodypose_last.get_keypoint(name)

            if (not np.isinf(kp1[0])) and  (not np.isinf(kp2[0])):
                vel = kp2-kp1
                vel /= deltaT

                body_vel.add_keypoint_array(name, vel)
                

        return body_vel


class BodyAccel(BodyPose):
    '''!
        This class stores the acceleration of a body pose.
    '''

    def __init__(self, pixel_space=False, frame_id='UNKOWN', user_id='UNKOWN', time=tm.time()):
        '''!
            BodyAccel constructor

            Parameters:
                @param pixels_space (boolean): True if the keypoints are in camera pixel (2D) space, false if its in 3D metric space
                @param frame_id (string): The name of the coordinate system where keypoints are
                @param user_id (string): The id of the body corresponding user
                @param time (float): The time of the acceleration detection
        '''

        super().__init__(pixel_space=pixel_space, frame_id=frame_id, user_id=user_id, time=time)

    @classmethod
    def from_bodyvel(cls, bodyvel, bodyvel_last):
        '''!
            Create a BodyAccel object from two BodyVel objects.

            Returns:
                @return body_accel (BodyAccel): the body accel object

            @todo Calcular covariância a partir da covariância das velocidades
        '''

        body_accel = BodyAccel(bodyvel._pixel_space, bodyvel._frame_id, bodyvel.user_id, bodyvel._time)

        deltaT = bodyvel.time - bodyvel_last.time

        for name in BodyPose.keypoints_names:
            kp1 = bodyvel.get_keypoint(name)
            kp2 = bodyvel_last.get_keypoint(name)

            if (not np.isinf(kp1[0])) and  (not np.isinf(kp2[0])):
                accel = kp2-kp1
                accel /= deltaT

                body_accel.add_keypoint_array(name, accel)

        return body_accel
        