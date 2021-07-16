import time

from redrawing.components.stage import Stage
from redrawing.data_interfaces.bodypose import *
from redrawing.data_interfaces.frame import Frame_TF

class User_Manager_Stage(Stage):
    '''!
        Handle the user data incoming from other stages.

        It checks wich user the BodyPose belongs to, and
        compute other features like velocity and acceleration
    '''

    configs_default = {"delete_time" : 300,
                        "threshold" : 1000,
                        "change_frame" : False,
                        "estimate_missing": False}

    def __init__(self, configs={}):
        super().__init__(configs=configs)

        self.addInput("bodypose_list", list)
        self.addInput("bodypose", BodyPose)

        self.addOutput("bodypose_list", list)
        self.addOutput("bodyvel_list", list)
        self.addOutput("bodyaccel_list", list)

        if self._configs["change_frame"]:
            self.addOutput("frame_tf_list", list)

    def setup(self):
        self._config_lock = True

        self._delete_time = self._configs["delete_time"]
        self._threshold = self._configs["threshold"]
        self._change_frame = self._configs["change_frame"]

        self._last_id = -1

        self._last_pose = {}
        self._actual_pose = {}
        self._last_time_seen = {}
        self._vel = {}
        self._last_vel = {}

        self._accel = {}
        self._last_accel = {}

        if self._change_frame:
            self._frame_tf = {}

    def _change_bd_frame(self, bodypose):
        neck = bodypose.NECK

        if neck is None:
            return

        t = - neck

        R = np.eye(3,dtype=np.float64)
        R[0,0] = -1
        R[2,2] = -1

        t = R@t

        original_frame = bodypose.frame_id
        destiny_frame = bodypose.user_id+"_neck"

        bodypose.apply_transformation(R,t, destiny_frame)

        return Frame_TF(original_frame, destiny_frame, R, t)


    def _compute_vel(self):
        '''!
            Computes the velocity from the bodyposes
        '''

        for actual_id in self._actual_pose:
            if actual_id in self._vel:
                self._last_vel[actual_id] = self._vel[actual_id]

            
            if actual_id in self._last_pose:
                deltaT = self._actual_pose[actual_id].time - self._last_pose[actual_id].time
                if deltaT > 0.0:
                    vel = BodyVel.from_bodyposes(self._actual_pose[actual_id], self._last_pose[actual_id])
                    self._vel[actual_id] = vel

    def _compute_accel(self):
        '''!
            Computes the acceleration from the bodyposes
        '''

        for actual_id in self._vel:
            if actual_id in self._accel:
                self._last_accel[actual_id] = self._accel[actual_id]

            
            if actual_id in self._last_vel:
                deltaT = self._vel[actual_id].time - self._last_vel[actual_id].time
                if deltaT > 0.0:
                    accel = BodyVel.from_bodyposes(self._vel[actual_id], self._last_vel[actual_id])
                    self._accel[actual_id] = accel

    #Completes the key name keypoint if the average of the values keypoints
    mean_complete = {"SPINE_SHOULDER"    :   ["SHOULDER_R", "SHOULDER_L"],
                    }

    def _complete(self, bp):
        for missing_name in User_Manager_Stage.mean_complete:
            missing_kp = bp.get_keypoint(missing_name)
            
            if np.isinf(missing_kp[0]):
                kps = []

                have_all = True

                for name in User_Manager_Stage.mean_complete[missing_name]:
                    kps.append(bp.get_keypoint(name))

                    if np.isinf(kps[-1][0]):
                        have_all = False
                        break

                if not have_all:
                    continue
                    
                missing_kp = np.mean(kps,0)


                bp.add_keypoint_array(missing_name, missing_kp)


    def process(self):
        '''!
            Handle the user data incoming

            Gets the BodyPose messages, check which user it belongs, 
            computes its velocities and accelerations.
        '''

        input_list = []
        if self.has_input("bodypose"):
            input_list.append(self._getInput("bodypose"))
        if self.has_input("bodypose_list"):
            input_list += self._getInput("bodypose_list")



        for bodypose in input_list:
            
            frametf = None

            if self._change_frame == True:
                frametf = self._change_bd_frame(bodypose)

            if bodypose.user_id == "UNKOWN":
                for actual_id in self._actual_pose:
                    if self._actual_pose[actual_id].frame_id != bodypose.frame_id:
                        continue

                    if BodyPose.distance(self._actual_pose[actual_id], bodypose) <= self._threshold:
                        self._last_pose[actual_id] = self._actual_pose[actual_id]
                        bodypose.user_id = actual_id
                        self._actual_pose[actual_id] = bodypose
                        
                        if self._change_frame:
                            self._frame_tf[actual_id] = frametf

                        break
                else:
                    self._last_id+=1
                    user_id = "user"+str(self._last_id)
                    bodypose.user_id = user_id
                    self._actual_pose[user_id] = bodypose

                    if self._change_frame:
                        self._frame_tf[user_id] = frametf
        

        actual_time = time.time()
        min_time = actual_time - self._delete_time
        
        
        for actual_id in self._actual_pose:
            if self._actual_pose[actual_id].time < min_time:
                del self._actual_pose[actual_id]
                continue
            
            if self._configs["estimate_missing"]:
                self._complete(self._actual_pose[actual_id])

        self._compute_vel()
        self._compute_accel()

        self._setOutput(list(self._actual_pose.values()), "bodypose_list")
        self._setOutput(list(self._vel.values()), "bodyvel_list")
        self._setOutput(list(self._accel.values()), "bodyaccel_list")

        pass