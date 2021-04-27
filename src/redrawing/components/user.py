import time

from redrawing.components.stage import Stage
from redrawing.data_interfaces.bodypose import *

class User_Manager_Stage(Stage):
    configs_default = {"delete_time" : 300,
                        "threshold" : 1000}

    def __init__(self, configs={}):
        super().__init__(configs=configs)

        self.addInput("bodypose_list", list)
        self.addInput("bodypose", BodyPose)

        self.addOutput("bodypose_list", list)
        self.addOutput("bodyvel_list", list)
        self.addOutput("bodyaccel_list", list)

    def setup(self):
        self._config_lock = True

        self._delete_time = self._configs["delete_time"]
        self._threshold = self._configs["threshold"]

        self._last_id = -1

        self._last_pose = {}
        self._actual_pose = {}
        self._last_time_seen = {}
        self._vel = {}
        self._last_vel = {}

        self._accel = {}
        self._last_accel = {}

    def _compute_vel(self):
        for actual_id in self._actual_pose:
            if actual_id in self._vel:
                self._last_vel[actual_id] = self._vel[actual_id]

            
            if actual_id in self._last_pose:
                deltaT = self._actual_pose[actual_id].time - self._last_pose[actual_id].time
                if deltaT > 0.0:
                    vel = BodyVel.from_bodyposes(self._actual_pose[actual_id], self._last_pose[actual_id])
                    self._vel[actual_id] = vel

    def _compute_accel(self):
        for actual_id in self._vel:
            if actual_id in self._accel:
                self._last_accel[actual_id] = self._accel[actual_id]

            
            if actual_id in self._last_vel:
                deltaT = self._vel[actual_id].time - self._last_vel[actual_id].time
                if deltaT > 0.0:
                    accel = BodyVel.from_bodyposes(self._vel[actual_id], self._last_vel[actual_id])
                    self._accel[actual_id] = accel

    def process(self):
        input_list = []
        if self.has_input("bodypose"):
            input_list.append(self._getInput("bodypose"))
        if self.has_input("bodypose_list"):
            input_list += self._getInput("bodypose_list")



        for bodypose in input_list:
            
            if bodypose.user_id == "UNKOWN":
                for actual_id in self._actual_pose:
                    if distance(self._actual_pose[actual_id], bodypose) <= self._threshold:
                        self._last_pose[actual_id] = self._actual_pose[actual_id]
                        bodypose.user_id = actual_id
                        self._actual_pose[actual_id] = bodypose
                        break
                else:
                    self._last_id+=1
                    user_id = "user"+str(self._last_id)
                    bodypose.user_id = user_id
                    self._actual_pose[user_id] = bodypose
        

        actual_time = time.time()
        min_time = actual_time - self._delete_time
        
        
        for actual_id in self._actual_pose:
            if self._actual_pose[actual_id].time < min_time:
                del self._actual_pose[actual_id]
                continue

        self._compute_vel()
        self._compute_accel()

        self._setOutput(list(self._actual_pose.values()), "bodypose_list")
        self._setOutput(list(self._vel.values()), "bodyvel_list")
        self._setOutput(list(self._accel.values()), "bodyaccel_list")


        pass