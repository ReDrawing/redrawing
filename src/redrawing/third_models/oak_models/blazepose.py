import os

import depthai as dai
import numpy as np
import cv2

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.components.oak_constants import *
from redrawing.components.oak_model import OAK_NN_Model
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.BlazeposeDepthai import BlazeposeDepthai, to_planar
import redrawing.third_models.oak_models.mediapipe_utils as mpu

class OAK_Blazepose(OAK_NN_Model):

    input_size = [128,128]

    outputs = {"bodypose": BodyPose, "bodypose_3d":BodyPose}

    kp_name = ["NOSE",
                None,
                "EYE_R",
                None,
                None,
                "EYE_L",
                None,
                "EAR_R",
                "EAR_L",
                None,
                None,
                "SHOULDER_R",
                "SHOULDER_L",
                "ELBOW_R",
                "ELBOW_L",
                "WRIST_R",
                "WRIST_L",
                "PINKY_MCP_R",
                "PINKY_MCP_L",
                "INDEX_FINGER_MCP_R",
                "INDEX_FINGER_MCP_L",
                "THUMB_MCP_R",
                "THUMB_MCP_L",
                "HIP_R",
                "HIP_L",
                "KNEE_R",
                "KNEE_L",
                "ANKLE_R",
                "ANKLE_L",
                "FOOT_R",
                "FOOT_L",
                None,
                None]

    def __init__(self):
        self.nodes = None
        
        self.input_size = OAK_Blazepose.input_size

        self.blazepose = BlazeposeDepthai()

        self.blazepose.lm_input_length = 256
        self.blazepose.pad_h = 0
        self.blazepose.pad_w = 0

        self.videoframe = None
        self.pd_inference = None
        self.lm_inference = None

    def create_node(self, oak_stage):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]
        
        pd_blob_path = base_path + "blazepose_pose_detection" + ".blob"
        lm_blob_path = base_path + "blazepose_pose_landmark_full_body"  + ".blob"

        oak_stage.pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

        pd_node = oak_stage.pipeline.createNeuralNetwork()
        pd_node.setBlobPath(pd_blob_path)
        pd_node.input.setQueueSize(1)
        pd_node.input.setBlocking(False)

        lm_node = oak_stage.pipeline.createNeuralNetwork()
        lm_node.setBlobPath(lm_blob_path)
        lm_node.input.setQueueSize(1)
        lm_node.input.setBlocking(False)

        if(oak_stage.preview_size["rgb"] != self.input_size):
            img_manip = oak_stage.pipeline.createImageManip()
            img_manip.setResize(self.input_size[0], self.input_size[1])
            oak_stage.cam["rgb"].preview.link(img_manip.inputImage)
            img_manip.out.link(pd_node.input)
        else:
            oak_stage.cam["rgb"].preview.link(pd_node.input)
        

        xLink_in = oak_stage.pipeline.createXLinkIn()
        xLink_in.setStreamName("lm_in")
        xLink_in.out.link(lm_node.input)
        oak_stage.input_link['lm_in'] = xLink_in
        
        self.nodes = {"blazepose_pd": pd_node,
                        "blazepose_lm": lm_node}
        
        return self.nodes

    def decode_result(self, oak_stage):
        
        
        video_frame = oak_stage.cam_output["rgb"]

        if video_frame is None:
            if self.videoframe is None:
                return
            else:
                video_frame = self.videoframe
        else:
            video_frame = video_frame.getCvFrame() 
            self.videoframe = video_frame
        
        
        self.blazepose.frame_size = video_frame.shape[0]


        pd_inference = oak_stage.nn_output["blazepose_pd"]

        if pd_inference is not None:
            self.blazepose.pd_postprocess(pd_inference)
            self.pd_inference = pd_inference
        elif self.pd_inference is None:
            return


        lm_in = oak_stage._oak_input_queue['lm_in']

        self.blazepose.nb_active_regions = 0

        for i,r in enumerate(self.blazepose.regions):
            frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.blazepose.lm_input_length, self.blazepose.lm_input_length)
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(frame_nn, (self.blazepose.lm_input_length, self.blazepose.lm_input_length)))
            lm_in.send(nn_data)

            lm_inference = oak_stage.nn_output["blazepose_lm"]

            if lm_inference is not None:
                self.blazepose.lm_postprocess(r, lm_inference)

                frame = self.videoframe.copy()

                bp = BodyPose()
                bp_3d = BodyPose()

                points = r.landmarks_abs


                for i,x_y in enumerate(r.landmarks_padded[:,:2]):

                    name = OAK_Blazepose.kp_name[i]

                    if name is None:
                        continue
                    
                    bp.add_keypoint(name,x_y[0],x_y[1])
                    bp_3d.add_keypoint(name,points[i][0],points[i][1],points[i][2])

                oak_stage._setOutput(bp, "bodypose")
                oak_stage._setOutput(bp_3d, "bodypose_3d")

            

        