import os
from abc import ABC, abstractmethod

import depthai as dai
import numpy as np

from redrawing.components.stage import Stage
from redrawing.components.oak_constants import *
from redrawing.data_interfaces.image import Image
from redrawing.data_interfaces.bodypose import BodyPose
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.human_pose import OAK_BodyPose

class OAK_Stage(Stage):


    configs_default = {"frame_id": "oak",
                        "rgb_out": False,
                        "rgb_resolution": [1920,1080],
                        "nn_enable":{"bodypose":True},
                        "nn_model" : {"bodypose":OAK_BodyPose},
                    }
    
    def __init__(self, configs={}):
        super().__init__(configs)

        self._config_lock = True

        for nn in self._configs["nn_enable"]:
            if self._configs["nn_enable"][nn] == True:
                self.addOutput(nn, list)
        if(self._configs["rgb_out"] == True):
            self.addOutput("rgb", Image)
        
        self.preview_size = {"rgb" : self._configs["rgb_resolution"]}

        self._device = None  
        self._nnQueeu = None
        self._cameraQueeu = None

        self.pipeline = None
        self.cam = {}
        
    
    def setup(self):
        '''!
            Configs the stage, creating the pipeline for the OAK

            @todo Criar nó para as câmeras monocromáticas
        '''

        self._config_lock = True

        if self._device is not None:
            return

        pipeline = dai.Pipeline()
        
        self.pipeline = pipeline

        rgb_size = [0,0]
        if self._configs["rgb_out"] == True:
            rgb_size = list(self._configs["rgb_resolution"])
        left_size = [0,0]
        right_size = [0,0]
        nn_list = {}

        for nn in self._configs["nn_enable"]:
            base_path = os.path.abspath(oak_models.__file__)
            base_path = base_path[:-11]

            if self._configs["nn_enable"][nn] == True:
                nn_list[nn] = self._configs["nn_model"][nn]()
                
                size = nn_list[nn].input_size
                input_type = nn_list[nn].input_type
                target_size = None

                if input_type == COLOR:
                    target_size = rgb_size
                elif input_type == LEFT:
                    target_size = left_size
                elif input_type == RIGHT:
                    target_size = right_size

                if target_size is not None:
                    if size[0] > target_size[0]:
                        rgb_size[0] = size[0]
                    if size[1] > target_size[1]:
                        rgb_size[1] = size[1]
        
        self._nn_list = nn_list

        rgb_cam = None
        left_cam = None
        right_cam = None
        if rgb_size != [0,0]:
            rgb_cam = pipeline.createColorCamera()
            rgb_cam.setPreviewSize(rgb_size[0], rgb_size[1])
            rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            rgb_cam.setInterleaved(False)
            rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        
        self.cam['rgb'] = rgb_cam
        self.cam['left'] = left_cam
        self.cam['right'] = right_cam

        nn_xout = {}

        for nn in nn_list:

            nn_node = nn_list[nn].create_node(self)
        
            xout = pipeline.createXLinkOut()
            xout.setStreamName(nn)
            nn_node.out.link(xout.input)
            nn_xout[nn] = xout
            
        cam_xout = {}
        if rgb_cam is not None:
            xout = pipeline.createXLinkOut()
            xout.setStreamName("rgb")
            rgb_cam.preview.link(xout.input)
            cam_xout["rgb"] = xout

        self._device = dai.Device(pipeline)
        self._device.startPipeline()

        nn_queeu = {}
        cam_queeu = {}

        for nn in nn_xout:
            nn_queeu[nn] = self._device.getOutputQueue(nn, maxSize=1, blocking=False)
        
        for cam in cam_xout:
            cam_queeu[cam] = self._device.getOutputQueue(cam, maxSize=1, blocking=False)

        self._nn_queeu = nn_queeu
        self._cam_queeu = cam_queeu

    

    def process(self):
        
        self.nn_output = {}
        self.cam_output = {}

        for nn in self._nn_queeu:
            output = self._nn_queeu[nn].tryGet()
            
            self.nn_output[nn] = output
        
        for cam in self._cam_queeu:
            self.cam_output[cam] = self._cam_queeu[cam].tryGet()

            if self.cam_output[cam] is not None:
                img = self.cam_output[cam].getCvFrame()

                img = Image(self._configs["frame_id"], img)

                self._setOutput(img, cam)

        for nn in self._nn_list:
            self._nn_list[nn].decode_result(self)

    def __del__(self):
        if self._device is not None:
            print("Interrompendo conexão com device")
            self._device.close()


