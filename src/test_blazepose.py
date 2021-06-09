from os import pipe
import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.third_models.oak_models.blazepose import OAK_Blazepose

if __name__ == '__main__':

    oak_configs = {}
    oak_stage = OAK_Stage(oak_configs)

    blazepose = OAK_Blazepose()


    udp_stage = UDP_Stage()


    user_configs = {"estimate_missing": True}

    pipeline = SingleProcess_Pipeline()

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)

    pipeline.set_substage(oak_stage, blazepose)

    pipeline.create_connection(blazepose, "bodypose3d_list", udp_stage, "send_msg_list", 1)

    pipeline.run()