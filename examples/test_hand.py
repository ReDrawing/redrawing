from os import pipe
import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage

from redrawing.ai_models import OAK_Handpose


if __name__ == '__main__':

    oak_configs = {}
    oak_stage = OAK_Stage(oak_configs)

    handpose = OAK_Handpose()

    udp_stage = UDP_Stage({"inputs_list": ["hand_pose_list", "gesture_list"]})

    pipeline = SingleProcess_Pipeline()

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)
    
    pipeline.set_substage(oak_stage, handpose)

    pipeline.create_connection(handpose, "gesture_list", udp_stage, "gesture_list", 10)
    pipeline.create_connection(handpose, "hand_pose_list", udp_stage, "hand_pose_list", 10)

    pipeline.run()