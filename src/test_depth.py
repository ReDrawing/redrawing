import cv2 as cv
import numpy as np

from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.components.pc_viewer import PCR_Viewer

from redrawing.third_models.oak_models.human_pose import OAK_BodyPose

show_pcr = True

if __name__ == '__main__':

    oak_configs = { "depth":True, "depth_close":True, "depth_filtering":"", "depth_point_mode": "min", 
                    "depth_roi_size":25, "depth_host": True,
                    "color_out": True, "color_size" : [640,400]}
    oak_stage = OAK_Stage(oak_configs)

    udp_stage = UDP_Stage()

    bodypose = OAK_BodyPose({"3d":True})

    pipeline = SingleProcess_Pipeline()

    pipeline.insert_stage(oak_stage)
    pipeline.insert_stage(udp_stage)
    pipeline.set_substage(oak_stage, bodypose)

    pipeline.create_connection(bodypose, "bodypose3d_list", udp_stage, "send_msg_list", 1)
    
    if show_pcr:
        pcr = PCR_Viewer({"bodypose":True})
        pipeline.insert_stage(pcr)

        pipeline.create_connection(oak_stage, "depth_map", pcr, "depth", 1)
        pipeline.create_connection(oak_stage, "color", pcr, "rgb", 1)
        pipeline.create_connection(bodypose, "bodypose3d_list", pcr, "bodypose_list", 1)

    pipeline.start()


    if show_pcr:
        pcr.calib_size = oak_stage.camera_calibration_size[OAK_Stage.COLOR]
        pcr.camera_intrinsics = oak_stage.camera_intrinsics[OAK_Stage.COLOR]

    pipeline.run()