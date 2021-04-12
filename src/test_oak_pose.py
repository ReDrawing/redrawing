import cv2 as cv
import numpy as np

from redrawing.components.oak import OAK_Pose
from redrawing.communication.udp import send_data
from redrawing.data_interfaces.bodypose import BodyPose

oak = OAK_Pose()

oak.setup()

while True:
    
    oak.process()

    poses = oak.getOutput("bodyposes")
    image = oak.getOutput("rgb")
    img = None

    if poses is None:
        continue

    for pose in poses:
        print("Enviando pose")
        send_data(pose)

        if image is not None:
            img = image.image
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