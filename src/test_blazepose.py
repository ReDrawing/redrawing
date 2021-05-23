import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage


if __name__ == '__main__':

    oak_configs = { "depth":False, "depth_close":True, "depth_filtering":True, "depth_point_mode": "", 
                    "nn_enable":{"bodypose":False, "blazepose":True}, 
                    "rgb_out": True, "rgb_resolution":[128,128]}
    oak_stage = OAK_Stage(oak_configs)

    udp_stage = UDP_Stage()

    pipeline = SingleProcess_Pipeline()

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)

    pipeline.create_connection(oak_stage, "bodypose_3d", udp_stage, "send_msg", 1)

    pipeline.run()