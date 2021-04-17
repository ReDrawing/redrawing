import os

import depthai as dai
import numpy as np
import cv2

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.data_interfaces.object_detection import ObjectDetection
from redrawing.components.oak_constants import *
from redrawing.components.oak_model import OAK_NN_Model
import redrawing.third_models.oak_models as oak_models

class OAK_PalmDetector(OAK_NN_Model):

    input_size = [128,128]

    def __init__(self):
        self.node = None
        
        self.input_size = OAK_PalmDetector.input_size
        self.input_type = COLOR

    def create_node(self, oak_stage):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]
        blob_path = base_path + "palm_detection" +".blob"

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
        
        self.node = nn_node
        return nn_node

    def decode_result(self, oak_stage): #Daniele
        nn_output = oak_stage.nn_output["palm"]
    
        result = ObjectDetection()

        #Daniele: processar o resultado da rede, e colocar em result

        oak_stage._setOutput(result, 'palm')

class OAK_HandFeature(OAK_NN_Model):
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
