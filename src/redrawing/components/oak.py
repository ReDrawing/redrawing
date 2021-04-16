import os

import depthai as dai
import numpy as np

from redrawing.components.stage import Stage
from redrawing.components.oak_constants import *
from redrawing.data_interfaces.image import Image
from redrawing.data_interfaces.bodypose import BodyPose
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.human_pose import bodypose_configs

class OAK_Stage(Stage):

    
    COLOR=1
    LEFT=2
    RIGHT=3

    configs_default = {"frame_id": "oak",
                        "rgb_out": False,
                        "rgb_resolution": (1920,1080),
                        "nn_enable":{"bodypose":True},
                        "nn_config": {"bodypose":bodypose_configs}
                    }

    def __init__(self, configs={}):
        super().__init__(configs)

        self._config_lock = True

        for nn in self._configs["nn_enable"]:
            if self._configs["nn_enable"][nn] == True:
                self.addOutput(nn, list)
        if(self._configs["rgb_out"] == True):
            self.addOutput("rgb", Image)

        self._device = None
        self._nnQueeu = None
        self._cameraQueeu = None
        
    
    def setup(self):
        '''!
            Configs the stage, creating the pipeline for the OAK

            @todo Criar nó para as câmeras monocromáticas
        '''

        self._config_lock = True

        if self._device is not None:
            return

        pipeline = dai.Pipeline()

        rgb_size = [0,0]
        left_size = [0,0]
        right_size = [0,0]
        nn_list = {}

        for nn in self._configs["nn_enable"]:
            base_path = os.path.abspath(oak_models.__file__)
            base_path = base_path[:-11]

            if self._configs["nn_enable"][nn] == True:

                blob_path = base_path + self._configs["nn_config"][nn]["blob_name"] +".blob"

                nn_node = pipeline.createNeuralNetwork()
                nn_node.setBlobPath(blob_path)
                nn_node.setNumInferenceThreads(2)
                nn_node.input.setQueueSize(1)
                nn_node.input.setBlocking(False)
                
                size = self._configs["nn_config"][nn]["img_size"]
                target_size = None

                if  self._configs["nn_config"][nn]["img_type"] == OAK_Stage.COLOR:
                    target_size = rgb_size
                elif self._configs["nn_config"][nn]["img_type"] == OAK_Stage.LEFT:
                    target_size = left_size
                elif self._configs["nn_config"][nn]["img_type"] == OAK_Stage.RIGHT:
                    target_size = right_size
                
                if target_size is not None:
                    if size[0] > target_size[0]:
                        rgb_size[0] = size[0]
                    if size[1] > target_size[1]:
                        rgb_size[1] = size[1]

                nn_list[nn] = nn_node

        rgb_cam = None
        left_cam = None
        right_cam = None
        if rgb_size != [0,0]:
            rgb_cam = pipeline.createColorCamera()
            rgb_cam.setPreviewSize(rgb_size[0], rgb_size[1])
            rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            rgb_cam.setInterleaved(False)
            rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        

        nn_xout = {}

        for nn in nn_list:
            target_size = None
            nn_size = self._configs["nn_config"][nn]["img_size"]
            img_type = self._configs["nn_config"][nn]["img_type"]
            cam = None
        
            if  img_type == OAK_Stage.COLOR:
                target_size = rgb_size
                cam = rgb_cam
            elif img_type == OAK_Stage.LEFT:
                target_size = left_size
                cam = left_cam
            elif img_type == OAK_Stage.RIGHT:
                target_size = right_size
                cam = right_cam

            if list(nn_size) != target_size:
                img_manip = pipeline.createImageManip()
                img_manip.setNumFramesPool(1)
                img_manip.setResize(nn_size[0],nn_size[1])
                cam.out.link(img_manip.inputImage)
                img_manip.out.link(nn_list[nn].input)
                
            else:
                cam.preview.link(nn_list[nn].input)

            xout = pipeline.createXLinkOut()
            xout.setStreamName(nn)
            nn_list[nn].out.link(xout.input)
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
        
        for nn in self._nn_queeu:
            nn_output = self._nn_queeu[nn].tryGet()
            
            if nn_output is not None:
                self._configs["nn_config"][nn]["process_function"](self, nn_output)
        
        for cam in self._cam_queeu:
            cam_output = self._cam_queeu[cam].tryGet()

            if cam_output is not None:
                img = cam_output.getCvFrame()

                img = Image(self._configs["frame_id"], img)

                self._setOutput(img, cam)


    def __del__(self):
        if self._device is not None:
            print("Interrompendo conexão com device")
            self._device.close()
