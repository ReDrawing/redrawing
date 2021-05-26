from math import exp
import os
import time
import depthai as dai
import numpy as np
import cv2
from redrawing.data_interfaces.gesture import Gesture
from redrawing.third_models.oak_models.HandTracker import HandTracker, to_planar
from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.data_interfaces.object_detection import ObjectDetection
from redrawing.components.oak_constants import *
from redrawing.components.oak_model import OAK_NN_Model
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models import hand_pose
import redrawing.third_models.oak_models.mediapipe_utils_hand as mpu


   
def now():
    return time.perf_counter()

class OAK_PalmDetector(OAK_NN_Model):
    '''!
        Handles the palm detector for OAK
    '''

    input_size = [128,128]

    outputs = {"palm_detection_list": list,
                "gesture_list" : list,
                "hand_pose_list":list}

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

    def __init__(self):
        self.node = None
        self.HandTracker = HandTracker()

        self.HandTracker.video_size = 128
        self.HandTracker.lm_input_length = 224
        self.HandTracker.use_gesture = True
        self.input_size = OAK_PalmDetector.input_size
        self.input_type = COLOR
        self.videoframe = None

    def create_node(self, oak_stage):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]
        blob_path = base_path + "palm_detection" +".blob"

        oak_stage.pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        
        nn_node = oak_stage.pipeline.createNeuralNetwork()
        nn_node.setBlobPath(blob_path)
        nn_node.setNumInferenceThreads(2)
        nn_node.input.setQueueSize(1)
        nn_node.input.setBlocking(False)

        if(oak_stage.preview_size["rgb"] != self.input_size):
            img_manip = oak_stage.pipeline.createImageManip()
            img_manip.setResize(self.input_size[0], self.input_size[1])
            oak_stage.cam["rgb"].preview.link(img_manip.inputImage)
            img_manip.out.link(nn_node.input)
        else:
            oak_stage.cam["rgb"].preview.link(nn_node.input)

        blob_path = base_path + "hand_landmark" +".blob"
        lm_node = oak_stage.pipeline.createNeuralNetwork()
        lm_node.setBlobPath(blob_path)
        lm_node.setNumInferenceThreads(1)
        lm_node.input.setQueueSize(1)
        nn_node.input.setBlocking(False)

        xLink_in = oak_stage.pipeline.createXLinkIn()
        xLink_in.setStreamName("hand_lm_in")
        xLink_in.out.link(lm_node.input)
        oak_stage.input_link['hand_lm_in'] = xLink_in
        
        self.nodes = {"palm_detector": nn_node,
                        "hand_landmark": lm_node}

        return self.nodes

    def decode_result(self, oak_stage): 
        '''!

            @todo Implementar saída assíncrona da OAK
        '''

        video_frame = oak_stage.cam_output["rgb"]

        if video_frame is None:
            if self.videoframe is None:
                return
            else:
                video_frame = self.videoframe
        else:
            video_frame = video_frame.getCvFrame() 
            self.videoframe = video_frame

        nn_output = oak_stage.nn_output["palm_detector"]

        if nn_output is None:
            return

        self.HandTracker.pd_postprocess(nn_output)

        results_palm = []

        for r in self.HandTracker.regions:
            box = (np.array(r.pd_box) * self.HandTracker.video_size).astype(int)
            ponto =np.array([[box[0], box[1]], [box[0]+box[2], box[1]+box[3]]])
            result = ObjectDetection()
            result.bounding_box = ponto

            results_palm.append(result)

        oak_stage._setOutput(results_palm, 'palm_detection_list')
        
        # Features/Keypoints -> Thiago
        ##################################################
        # Hand landmarks

        lm_in = oak_stage._oak_input_queue['hand_lm_in']

        

        for i,r in enumerate(self.HandTracker.regions):
            img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.HandTracker.lm_input_length, self.HandTracker.lm_input_length)
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(img_hand, (self.HandTracker.lm_input_length, self.HandTracker.lm_input_length)))
            lm_in.send(nn_data)
        
            inference = oak_stage.nn_output["hand_landmark"] 

            if inference is not None:

                self.HandTracker.lm_postprocess(r, inference)
        
        bodyposes = []

        try:
            self.HandTracker.regions[0].landmarks
        except:
            return
        
        for region in self.HandTracker.regions:
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

            bp = BodyPose()
            for i in range(lm_xy.shape[0]):
                name = OAK_PalmDetector.kp_name[i]
                bp.add_keypoint(name,lm_xy[i][0],lm_xy[i][1])

            bodyposes.append(bp)
                
        oak_stage._setOutput(bodyposes, "hand_pose_list")
        ##################################################

        list_regions = []
        for i,r in enumerate(self.HandTracker.regions):
            gesture = Gesture()
            gesture._gesture=r.gesture
            list_regions.append(gesture)
        
        oak_stage._setOutput(list_regions, 'gesture_list')
