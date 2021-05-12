import numpy as np

from redrawing.communication.udp import UDP_Stage;
from redrawing.data_interfaces import BodyPose;

udp_stage = UDP_Stage()

bodypose = BodyPose(pixel_space=True,frame_id="test",user_id="dummy",time=0)

bodypose.NECK = np.array([100.0,100.0,1.0])
bodypose.NOSE = np.array([100.0,50.0,1.0])
bodypose.EYE_R = np.array([150.0, 60.0, 1.0])
bodypose.EYE_L = np.array([50.0, 60.0, 1.0])
bodypose.SHOULDER_R = np.array([180.0,110.0,1.0])
bodypose.SHOULDER_L = np.array([20.0,110.0,1.0])

udp_stage.setup()

while True:
    udp_stage.setInput(bodypose,"send_msg")
    udp_stage.run()

