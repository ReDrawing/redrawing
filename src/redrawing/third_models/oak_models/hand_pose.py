import os
import time
import depthai as dai
import numpy as np
import cv2
from redrawing.third_models.oak_models.HandTracker import HandTracker

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.data_interfaces.object_detection import ObjectDetection
from redrawing.components.oak_constants import *
from redrawing.components.oak_model import OAK_NN_Model
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models import hand_pose


   
def now():
    return time.perf_counter()

class OAK_PalmDetector(OAK_NN_Model):
    '''!
        Handles the palm detector for OAK
    '''

    input_size = [128,128]

    outputs = {"palm_detection": ObjectDetection}

    def __init__(self):
        self.node = None
        self.HandTracker = HandTracker()

        self.HandTracker.video_size = 128

        self.input_size = OAK_PalmDetector.input_size
        self.input_type = COLOR

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
        
        self.nodes = {"palm_detector": nn_node}

        return self.nodes

    def decode_result(self, oak_stage): #Daniele
        nn_output = oak_stage.nn_output["palm_detector"]

        if nn_output is None:
            return

        self.HandTracker.pd_postprocess(nn_output)
        for r in self.HandTracker.regions:
            box = (np.array(r.pd_box) * self.HandTracker.video_size).astype(int)
            ponto =np.array([[box[0], box[1]], [box[0]+box[2], box[1]+box[3]]])
            result = ObjectDetection()
            result.bounding_box = ponto

            oak_stage._setOutput(result, 'palm_detection')

class OAK_HandFeature(OAK_NN_Model):
    '''!
        Handles the hand feature detector for oak
    '''

    input_size = [456,256]

    def __init__(self):
        self.node = None
        
        self.input_size = OAK_HandFeature.input_size
        self.input_type = OTHER_MODEL

    def create_node(self, oak_stage):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]
        blob_path = base_path + "hand_landmark" +".blob"

        nn_node = oak_stage.pipeline.createNeuralNetwork()
        nn_node.setBlobPath(blob_path)
        nn_node.setNumInferenceThreads(2)
        nn_node.input.setQueueSize(1)
        nn_node.input.setBlocking(False)

        xLink_in = oak_stage.pipeline.createXLinkIn()
        xLink_in.setStreamName("hand_feature_in")
        xLink_in.out.link(nn_node.input)
        oak_stage.input_link['hand_feature'] = xLink_in

        self.node = nn_node
        return nn_node

    def decode_result(self, oak_stage):
        input_queue = oak_stage.input_queue['hand_feature']
        palm_result = oak_stage.nn_output['palm']
        hand_result = oak_stage.nn_output['hand_feature']

        #Thiago: pegar o resultado da rede de palma e colocar na rede de hand_feature

        result = BodyPose()

        #Thiago:  processar o resultado da rede de hand_feature e colocar em um bodypose 

        oak_stage._setOutput(result,'hand_feature')
