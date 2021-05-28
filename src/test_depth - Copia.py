import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.components.pc_viewer import PCR_Viewer

if __name__ == '__main__':

    oak_configs = { "depth":True, "depth_close":True, "depth_filtering":"", "depth_point_mode": "min", 
                    "depth_roi_size":25, "depth_host": True,
                    "nn_enable":{"bodypose":True}, 
                    "rgb_out": True, "rgb_resolution" : [640,400],
                    "force_reconnection": False}
    oak_stage = OAK_Stage(oak_configs)

    udp_stage = UDP_Stage()

    pipeline = MultiProcess_Pipeline()

    pcr = PCR_Viewer({"bodypose":True})
    pcr.camera_intrinsics = OAK_Stage.color_intrinsic
    pcr.calib_size = OAK_Stage.color_calib_size

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)
    pipeline.insert_stage(pcr)

    pipeline.create_connection(oak_stage, "bodypose", udp_stage, "send_msg_list", 1)
    pipeline.create_connection(oak_stage, "depth_map", pcr, "depth", 1)
    pipeline.create_connection(oak_stage, "rgb", pcr, "rgb", 1)
    pipeline.create_connection(oak_stage, "bodypose", pcr, "bodypose_list", 1)

    #pipeline.start()
    pipeline.run()