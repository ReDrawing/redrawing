from math import exp
import os
import time

import depthai as dai
import numpy as np
import cv2

from redrawing.data_interfaces.gesture import Gesture
from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.data_interfaces.object_detection import ObjectDetection

from redrawing.components.oak import OAK_Stage
from redrawing.components.oak_substage import OAK_Substage

import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.HandTracker import HandTracker, to_planar
import redrawing.third_models.oak_models.mediapipe_utils_hand as mpu



class OAK_Handpose(OAK_Substage):
    '''!
        Handles the palm detector for OAK
    '''

    input_size = [128,128]

    

    kp_name = ["WRIST_R",
                "THUMB_CMC_R",
                "THUMB_MCP_R",
                "THUMB_IP_R",
                "THUMB_TIP_R",
                "INDEX_FINGER_MCP_R",
                "INDEX_FINGER_PIP_R",
                "INDEX_FINGER_DIP_R",
                "INDEX_FINGER_TIP_R",
                "MIDDLE_FINGER_MCP_R",
                "MIDDLE_FINGER_PIP_R", 
                "MIDDLE_FINGER_DIP_R",
                "MIDDLE_FINGER_TIP_R",
                "RING_FINGER_MCP_R",
                "RING_FINGER_PIP_R",
                "RING_FINGER_DIP_R",
                "RING_FINGER_TIP_R",
                "PINKY_MCP_R",
                "PINKY_PIP_R",
                "PINKY_DIP_R",
                "PINKY_TIP_R",
                ]

    configs_default = {}

    def __init__(self, configs={}):
        super().__init__(configs, name="handpose", color_input_size=[128,128], color_out=True)

        self.addOutput("palm_detection_list", list)
        self.addOutput("gesture_list", list)
        self.addOutput("hand_pose_list", list)

    def setup(self):
        self.HandTracker = HandTracker()

        self.HandTracker.lm_input_length = 224
        self.HandTracker.use_gesture = True

        self.videoframe = None

    def create_nodes(self, pipeline):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]

        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        
        blob_path = base_path + "palm_detection" +".blob"
        palm_node = pipeline.createNeuralNetwork()
        palm_node.setBlobPath(blob_path)
        palm_node.setNumInferenceThreads(2)
        palm_node.input.setQueueSize(1)
        palm_node.input.setBlocking(False)

        blob_path = base_path + "hand_landmark" +".blob"
        lm_node = pipeline.createNeuralNetwork()
        lm_node.setBlobPath(blob_path)
        lm_node.setNumInferenceThreads(1)
        lm_node.input.setQueueSize(1)
        lm_node.input.setBlocking(False)

        return {"palm_detector": palm_node, "hand_lm": lm_node}

    def link(self, pipeline, nodes, rgb_out, left_out, right_out, depth_out):
        rgb_out.link(nodes["palm_detector"].input)

        xLink_in = pipeline.createXLinkIn()
        xLink_in.setStreamName("hand_lm_in")
        xLink_in.out.link(nodes["hand_lm"].input)

        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("palm_detector")
        nodes["palm_detector"].out.link(pd_out.input)

        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("hand_lm")
        nodes["hand_lm"].out.link(lm_out.input)

        links = {"hand_lm_in": xLink_in,
                "palm_detector" : pd_out,
                "hand_lm" : lm_out}
        
        return links
    
    def create_output_queues(self, device):
        pd_out = device.getOutputQueue("palm_detector", maxSize=1, blocking=False)
        lm_out = device.getOutputQueue("hand_lm", maxSize=1, blocking=False)

        return {"palm_detector" : pd_out, "hand_lm" : lm_out}

    def create_input_queues(self, device):
        lm_in = device.getInputQueue("hand_lm_in", maxSize=1, blocking=False)

        return {"hand_lm_in": lm_in}

    def process(self, context):

        video_frame = context["frame"][OAK_Stage.COLOR]

        if video_frame is None:
            if self.videoframe is None:
                return
            else:
                video_frame = self.videoframe
        else:
            self.videoframe = video_frame

        self.HandTracker.video_size = video_frame.shape[0]
        pd_inference = context["output_queues"]["palm_detector"].tryGet()

        if pd_inference is not None:

            self.HandTracker.pd_postprocess(pd_inference)

            results_palm = []

            for r in self.HandTracker.regions:
                box = (np.array(r.pd_box) * self.HandTracker.video_size).astype(int)
                ponto =np.array([[box[0], box[1]], [box[0]+box[2], box[1]+box[3]]])
                result = ObjectDetection()
                result.bounding_box = ponto

                results_palm.append(result)

            self._setOutput(results_palm, 'palm_detection_list')

            bodyposes = []
            gestures = []

            for i,r in enumerate(self.HandTracker.regions):
                img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.HandTracker.lm_input_length, self.HandTracker.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.HandTracker.lm_input_length, self.HandTracker.lm_input_length)))
                context["input_queues"]['hand_lm_in'].send(nn_data)
            
                inference = context["output_queues"]['hand_lm'].get()

                self.HandTracker.lm_postprocess(r, inference)

                if r.lm_score< self.HandTracker.lm_score_threshold:
                    continue
            
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array([ (x, y) for x,y in r.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in r.landmarks]), axis=0)
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

                bp = BodyPose(frame_id=context["frame_id"], pixel_space=True)
                for i in range(lm_xy.shape[0]):
                    name = OAK_Handpose.kp_name[i]
                    bp.add_keypoint(name,lm_xy[i][0],lm_xy[i][1])

                bodyposes.append(bp)

                gesture = Gesture()
                gesture._gesture=r.gesture
                gestures.append(gesture)
                        
            self._setOutput(bodyposes, "hand_pose_list")
            self._setOutput(gestures, 'gesture_list')
        