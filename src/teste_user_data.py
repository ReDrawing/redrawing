import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.components.user import User_Manager_Stage
from redrawing.communication.udp import UDP_Stage

if __name__ == '__main__':

    oak_configs = {"rgb_out" : True, "rgb_resolution": [456, 256], "nn_enable":{"bodypose":True}}
    oak_stage = OAK_Stage(oak_configs)

    udp_stage = UDP_Stage()

    user_configs = {"change_frame" : True}
    user_manager = User_Manager_Stage(user_configs)

    pipeline = MultiProcess_Pipeline()

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)
    pipeline.insert_stage(user_manager)

    pipeline.create_connection(oak_stage, "bodypose", user_manager, "bodypose_list", 1)
    pipeline.create_connection(user_manager, "bodypose_list", udp_stage, "send_msg_list", 1)

    pipeline.run()