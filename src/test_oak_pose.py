import cv2 as cv
import numpy as np

from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import send_data
from redrawing.data_interfaces.bodypose import BodyPose


oak_configs = {"rgb_out" : True, "rgb_resolution": [456, 256], "nn_enable":{"bodypose":True}}

oak = OAK_Stage(oak_configs)

oak.setup()

while True:
    
    oak.process()

    poses = oak.getOutput("bodypose")
    image = oak.getOutput("rgb")
    img = None

    if image is not None:
        img = image.image

    if poses is not None:

        for pose in poses:
            print("Enviando pose")
            send_data(pose)

            if img is not None:
                for name in BodyPose.keypoints_names:
                    try:
                        kp = pose.get_keypoint(name)
                        kp = kp[:2]

                        cv.circle(img, (int(kp[0]),int(kp[1])), 2, (255,0,0), -1)
                        cv.putText(img, name, (int(kp[0]),int(kp[1])), cv.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255))
                    except Exception as e:
                        pass
        
    if img is not None:
        cv.imshow("result", img)

    if cv.waitKey(1) == ord('c'):
        break