from math import exp
import os

import depthai as dai
import numpy as np
import cv2

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.components.oak import OAK_Stage
from redrawing.components.oak_substage import OAK_Substage
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.BlazeposeDepthai import BlazeposeDepthai, to_planar
import redrawing.third_models.oak_models.mediapipe_utils as mpu

class OAK_Blazepose(OAK_Substage):

    kp_name = ["NOSE",
                "EYE_R_INNER",
                "EYE_R",
                "EYE_R_OUTER",
                "EYE_L_INNER",
                "EYE_L",
                "EYE_L_OUTER",
                "EAR_R",
                "EAR_L",
                "MOUTH_R",
                "MOUTH_L",
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
                "FOOT_R_INDEX",
                "FOOT_L_INDEX"]

    configs_default = {}

    def __init__(self, configs={}):
        super().__init__(configs, name="blazepose", color_input_size=[128,128], color_out=True)

        self.addOutput("bodypose_list", list)
        self.addOutput("bodypose3d_list", list)


    def setup(self):
        self.blazepose = BlazeposeDepthai()

        self.blazepose.lm_input_length = 256
        self.blazepose.pad_h = 0
        self.blazepose.pad_w = 0

        self.videoframe = None
        self.lm_inference = None

    def create_nodes(self, pipeline):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]     
        
        pd_blob_path = base_path + "blazepose_pose_detection" + ".blob"
        lm_blob_path = base_path + "blazepose_pose_landmark_full_body"  + ".blob"

        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

        pd_node = pipeline.createNeuralNetwork()
        pd_node.setBlobPath(pd_blob_path)
        pd_node.input.setQueueSize(1)
        pd_node.input.setBlocking(False)

        lm_node = pipeline.createNeuralNetwork()
        lm_node.setBlobPath(lm_blob_path)
        lm_node.input.setQueueSize(1)
        lm_node.input.setBlocking(False)

        nodes = {"blazepose_pd": pd_node,
                    "blazepose_lm": lm_node}
        
        return nodes

    def link(self, pipeline, nodes, rgb_out, left_out, right_out, depth_out):
        rgb_out.link(nodes["blazepose_pd"].input)

        xLink_in = pipeline.createXLinkIn()
        xLink_in.setStreamName("blazepose_lm_in")
        xLink_in.out.link(nodes["blazepose_lm"].input)
        
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("blazepose_pd")
        nodes["blazepose_pd"].out.link(pd_out.input)

        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("blazepose_lm")
        nodes["blazepose_lm"].out.link(lm_out.input)

        links = {"blazepose_lm_in": xLink_in,
                "blazepose_pd" : pd_out,
                "blazepose_lm" : lm_out}
        
        return links

    def create_output_queues(self, device):
        pd_out = device.getOutputQueue("blazepose_pd", maxSize=1, blocking=False)
        lm_out = device.getOutputQueue("blazepose_lm", maxSize=1, blocking=False)

        return {"blazepose_pd" : pd_out, "blazepose_lm" : lm_out}

    def create_input_queues(self, device):
        lm_in = device.getInputQueue("blazepose_lm_in", maxSize=1, blocking=False)

        return {"blazepose_lm_in": lm_in}

    def process(self, context):
        video_frame = context["frame"][OAK_Stage.COLOR]

        if video_frame is None:
            if self.videoframe is None:
                return
            else:
                video_frame = self.videoframe
        else:
            self.videoframe = video_frame

        
        self.blazepose.frame_size = video_frame.shape[0]
        pd_inference = context["output_queues"]["blazepose_pd"].tryGet()

        if pd_inference is not None:
            self.blazepose.pd_postprocess(pd_inference)
        else: 
            return

        self.blazepose.nb_active_regions = 0

        bodyposes = []
        bodyposes_3d = []

        for i,r in enumerate(self.blazepose.regions):
            frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.blazepose.lm_input_length, self.blazepose.lm_input_length)
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(frame_nn, (self.blazepose.lm_input_length, self.blazepose.lm_input_length)))
            context["input_queues"]['blazepose_lm_in'].send(nn_data)

            lm_inference = context["output_queues"]['blazepose_lm'].get()

            
        
            self.blazepose.lm_postprocess(r, lm_inference)

            if r.lm_score < self.blazepose.lm_score_threshold:

                continue

            bp = BodyPose()
            bp_3d = None

            points = r.landmarks_abs
            bp_3d = BodyPose()

            for i,x_y in enumerate(r.landmarks_padded[:,:2]):

                name = OAK_Blazepose.kp_name[i]

                if name is None:
                    continue
                
                bp.add_keypoint(name,x_y[0],x_y[1])
                bp_3d.add_keypoint(name,points[i][0],points[i][1],points[i][2])

            bodyposes.append(bp)
            bodyposes_3d.append(bp_3d)
            
        self._setOutput(bodyposes, "bodypose_list")
        self._setOutput(bodyposes_3d, "bodypose3d_list")
            

        