import cv2 as cv
import numpy as np

from redrawing.components.pipeline import SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage


oak_configs = {"depth":True, "nn_enable":{"bodypose":False}}
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