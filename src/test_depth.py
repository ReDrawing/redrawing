import cv2 as cv
import numpy as np
from numpy.core.numeric import NaN

from redrawing.components.pipeline import SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.data_interfaces.bodypose import BodyPose


oak_configs = {"depth":True, "nn_enable":{"bodypose":True}}
oak_stage = OAK_Stage(oak_configs)

oak_stage.setup()


while True:
    oak_stage.run()
    depth = oak_stage.getOutput("depth_img")

    if depth is None:
        continue

    depth = depth.image

    cv.imshow("depth",depth)

    if cv.waitKey(1) == ord('q'):
        break

    bodyposes = oak_stage.getOutput("bodypose")

    if bodyposes is None:
        continue

    for bd in bodyposes:
        for name in BodyPose.keypoints_names:
            kp = bd.get_keypoint(name)
        
            if kp[0] == np.inf:
                continue

            print(name, kp[0],kp[1],kp[2])

    print("\n\n")