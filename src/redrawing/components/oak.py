import os

import depthai as dai
import numpy as np

from redrawing.components.stage import Stage
from redrawing.data_interfaces.image import Image
from redrawing.data_interfaces.bodypose import BodyPose
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.human_pose import process_output

class OAK_Pose(Stage):

    keypointDict = {'Nose'  : "NOSE"      , 
                    'Neck'  : "NECK"      ,
                    'R-Sho' : "SHOULDER_R", 
                    'R-Elb' : "ELBOW_R"   , 
                    'R-Wr' : "WRIST_R"   ,   
                    'L-Sho' : "SHOULDER_L", 
                    'L-Elb' : "ELBOW_L"   , 
                    'L-Wr' : "WRIST_L"   ,
                    'R-Hip' : "HIP_R"     , 
                    'R-Knee': "KNEE_R"    , 
                    'R-Ank' : "ANKLE_R"   , 
                    'L-Hip' : "HIP_L"     , 
                    'L-Knee': "KNEE_L"    , 
                    'L-Ank' : "ANKLE_L"   ,
                    'R-Eye' : "EYE_R"     , 
                    'L-Eye' : "EYE_L"     ,
                    'R-Ear' : "EAR_R"     , 
                    'L-Ear' : "EAR_L"     }

    

    def __init__(self):
        super().__init__()
        self.addOutput("bodyposes", list)
        self.addOutput("rgb", Image)

        self._device = None
        self._nnQueeu = None

        
    
    def setup(self):

        if self._device is not None:
            return

        blob_path = os.path.abspath(oak_models.__file__)
        blob_path = blob_path[:-11]
        blob_path += "human-pose-estimation-0001_openvino_2021.2_6shave.blob"

        pipeline = dai.Pipeline()

        #Nó da câmera
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(456, 256)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        #Nó da rede neural
        pose_nn = pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(blob_path)
        pose_nn.setNumInferenceThreads(2)
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)

        #Links
        cam.preview.link(pose_nn.input)

        #X-links de saída
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("rgb")
        cam.preview.link(cam_xout.input)

        self._device = dai.Device(pipeline)
        self._device.startPipeline()

        self._nnQueeu = self._device.getOutputQueue("pose_nn", maxSize=1, blocking=False)
        self._rgbQueeu = self._device.getOutputQueue("rgb", maxSize=1, blocking=False)


    def process(self):
        nn_output = self._nnQueeu.tryGet()
        rbg_output = self._rgbQueeu.tryGet()

        if nn_output is not None:
            poses = process_output(nn_output, 256, 456)
            
            

            bodyposes = []
            for pose in poses:
                bodypose = BodyPose(pixel_space=True, frame_id="oak")

                for keypoint_name in pose:
                    keypoint_key = OAK_Pose.keypointDict[keypoint_name]
                    bodypose.add_keypoint(keypoint_key, pose[keypoint_name][0], pose[keypoint_name][1])    

                bodyposes.append(bodypose)
            
            self._setOutput(bodyposes, "bodyposes")

        

        if rbg_output is not None:
            rgb_img = rbg_output.getCvFrame()

            img = Image("oak", rgb_img)

            self._setOutput(img, "rgb")


    def __del__(self):
        if self._device is not None:
            print("Interrompendo conexão com device")
            self._device.close()
