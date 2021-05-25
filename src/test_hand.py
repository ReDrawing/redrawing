from os import pipe
import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.components.user import User_Manager_Stage


if __name__ == '__main__':

    oak_configs = { "depth":False,
                    "nn_enable":{"bodypose":False, "hand_pose":True}, 
                    "force_reconnection": False}
    oak_stage = OAK_Stage(oak_configs)

    udp_stage = UDP_Stage()


    pipeline = SingleProcess_Pipeline()

    

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)

    pipeline.create_connection(oak_stage, "palm_detection", udp_stage, "send_msg", 1)

    while True:
        pipeline.runOnce()

        palm_detection = oak_stage.getOutput("palm_detection")

        if palm_detection is not None:

            print(palm_detection.bounding_box)