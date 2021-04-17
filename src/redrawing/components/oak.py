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
        '''!
            OAK_Stage constructor.

            Parameters:
                @param configs : dict = The configurations changes from the defaults configs of the stage
        '''
        super().__init__(configs)

        self._config_lock = True

        for nn in self._configs["nn_enable"]:
            if self._configs["nn_enable"][nn] == True:
                self.addOutput(nn, list)
        if(self._configs["rgb_out"] == True):
            self.addOutput("rgb", Image)
        
        self.preview_size = {"rgb" : self._configs["rgb_resolution"]}

        self._device = None  
        self._nnqueue = None
        self._cameraqueue = None

        self.pipeline = None
        self.cam = {}
        self.input_queue = {}
        self.input_link = {}
        
    
    def setup(self):
        '''!
            Configs the OAK stage.

            Creates the pipeline for the OAK with all the camera and NN nodes, 
            initiate and creates the queues

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

        nn_queue = {}
        cam_queue = {}

        for nn in nn_xout:
            nn_queue[nn] = self._device.getOutputQueue(nn, maxSize=1, blocking=False)
        
        for cam in cam_xout:
            cam_queue[cam] = self._device.getOutputQueue(cam, maxSize=1, blocking=False)

        for link in self.input_link:
            self.input_queue[link] = self._device.getInputQueue(link)

        self._nn_queue = nn_queue
        self._cam_queue = cam_queue

    

    def process(self):
        '''!
            Process the data received from the OAK.

            Decodes the neural results and pass with the camera images
            to the outputs channels
        '''
        
        self.nn_output = {}
        self.cam_output = {}

        for nn in self._nn_queue:
            output = self._nn_queue[nn].tryGet()
            
            self.nn_output[nn] = output
        
        for cam in self._cam_queue:
            self.cam_output[cam] = self._cam_queue[cam].tryGet()

            if self.cam_output[cam] is not None:
                img = self.cam_output[cam].getCvFrame()

                img = Image(self._configs["frame_id"], img)

                self._setOutput(img, cam)

        for nn in self._nn_list:
            self._nn_list[nn].decode_result(self)

    def __del__(self):
        if self._device is not None:
            self._device.close()

