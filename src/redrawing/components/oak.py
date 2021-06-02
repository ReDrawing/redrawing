import os
from abc import ABC, abstractmethod

import depthai as dai
import numpy as np
import cv2 as cv

from redrawing.components.stage import Stage
from redrawing.components.oak_constants import *
from redrawing.data_interfaces.depth_map import Depth_Map
from redrawing.data_interfaces.image import Image
from redrawing.data_interfaces.bodypose import BodyPose
import redrawing.third_models.oak_models as oak_models
from redrawing.third_models.oak_models.human_pose import OAK_BodyPose
from redrawing.third_models.oak_models.blazepose import OAK_Blazepose
from redrawing.third_models.oak_models.hand_pose import OAK_PalmDetector

class OAK_Stage(Stage):
    '''!
        @todo Ler intrinsics da câmera do EEPROM dela, quando implementado pela depthai
    '''


    configs_default = {"frame_id": "oak",
                        "rgb_out": False,
                        "rgb_resolution": [1280,720],
                        "nn_enable":{"bodypose":True, "blazepose": False, "hand_pose": False},
                        "nn_model" : {"bodypose":OAK_BodyPose, "blazepose": OAK_Blazepose, "hand_pose": OAK_PalmDetector},
                        "depth" : False,
                        "depth_close" : False,
                        "depth_far": False,
                        "force_reconnection": True,
                        "depth_filtering" : True,
                        "depth_point_mode" : "median",
                        "depth_roi_size" : 50,
                        "depth_host" : True
                    }

    gray_intrinsic = np.array([[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]],dtype=np.float64)
    gray_intrinsic_inv = np.linalg.inv(gray_intrinsic)

    #color_intrinsic = np.array([[373.95694075, 0, 158.39368282], [0, 375.78372531, 170.28561667], [0,0,1.]],dtype=np.float64)
    #color_calib_size = [300,300]

    #color_intrinsic = np.array([[1488.843994140625, 0.0, 956.4694213867188], [0.0, 1486.9603271484375, 546.5672607421875], [0.0, 0.0, 1.0]],dtype=np.float64)
    color_intrinsic = np.array([[1486.9603271484375, 0.0, 546.5672607421875], [0.0,1488.843994140625 , 956.4694213867188], [0.0, 0.0, 1.0]],dtype=np.float64)
    color_calib_size = [1080,1920]

    color_intrinsic_inv = np.linalg.inv(color_intrinsic)

    d = -1
    sigma = 3

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
                for id_name in self._configs["nn_model"][nn].outputs:
                    self.addOutput(id_name, self._configs["nn_model"][nn].outputs[id_name])

        if(self._configs["rgb_out"] == True):
            self.addOutput("rgb", Image)
        if(self._configs["depth"] == True):
            self.addOutput("depth_img", Image)
            self.addOutput("left", Image)
            self.addOutput("right", Image)

            self.addOutput("depth_map", Depth_Map)
        
        self.preview_size = {"rgb" : self._configs["rgb_resolution"]}

        self._device = None  
        self._nnqueue = None
        self._cameraqueue = None

        self.pipeline = None
        self.cam = {}
        self._oak_input_queue = {}
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

        using_nn = False

        for nn in self._configs["nn_enable"]:
            base_path = os.path.abspath(oak_models.__file__)
            base_path = base_path[:-11]

            if self._configs["nn_enable"][nn] == True:
                using_nn = True

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

            if self._configs["depth"] == True:
                rgb_cam.initialControl.setManualFocus(130)
        
        self._rgb_size = rgb_size

        if self._configs["depth"] == True:
            left_cam = pipeline.createMonoCamera()
            left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)

            right_cam = pipeline.createMonoCamera()
            right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            depth_node = pipeline.createStereoDepth()
            depth_node.setLeftRightCheck(True)
            depth_node.setConfidenceThreshold(200)

            
            if (not using_nn) and self._configs["depth_close"] == True: #See close -> Extended
                depth_node.setExtendedDisparity(True)
                depth_node.setSubpixel(False)
            elif (not using_nn) and self._configs["depth_far"] == True: #See longer -> Subpixel
                depth_node.setSubpixel(True)
                depth_node.setExtendedDisparity(False)
            else:
                #depth_node.setDepthAlign(dai.CameraBoardSocket.RGB)
                depth_node.setDepthAlign(dai.StereoDepthProperties.DepthAlign.CENTER)
                depth_node.setSubpixel(False)
                depth_node.setExtendedDisparity(False)

            left_cam.out.link(depth_node.left)
            right_cam.out.link(depth_node.right)

            self.depth_node = depth_node

            if self._configs["depth_host"] == False:
                spatialLocationCalculator = pipeline.createSpatialLocationCalculator()
                
                topLeft = dai.Point2f(0.4, 0.4)
                bottomRight = dai.Point2f(0.6, 0.6)
                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000
                config.roi = dai.Rect(topLeft, bottomRight)
                spatialLocationCalculator.setWaitForConfigInput(False)
                spatialLocationCalculator.initialConfig.addROI(config)

                depth_node.disparity.link(spatialLocationCalculator.inputDepth)

                self.spatialLocationCalculator = spatialLocationCalculator

        self.cam['rgb'] = rgb_cam
        self.cam['left'] = left_cam
        self.cam['right'] = right_cam

        nn_xout = {}

        for nn in nn_list:

            nn_node_dict = nn_list[nn].create_node(self)

            for stream_name in nn_node_dict:
                xout = pipeline.createXLinkOut()
                xout.setStreamName(stream_name)
                nn_node_dict[stream_name].out.link(xout.input)
                nn_xout[stream_name] = xout
            
        cam_xout = {}
        if rgb_cam is not None and self._configs["rgb_out"]:
            xout = pipeline.createXLinkOut()
            xout.setStreamName("rgb")
            rgb_cam.preview.link(xout.input)
            cam_xout["rgb"] = xout
        if left_cam is not None:
            xout = pipeline.createXLinkOut()
            xout.setStreamName("left")
            left_cam.out.link(xout.input)
            cam_xout["left"] = xout
        if left_cam is not None:
            xout = pipeline.createXLinkOut()
            xout.setStreamName("right")
            right_cam.out.link(xout.input)
            cam_xout["right"] = xout

        depth_xout = None

        if self._configs["depth"] == True:

            if self._configs["depth_host"] == False:
                xout = pipeline.createXLinkOut()
                xout.setStreamName("spartialData")
                self.spatialLocationCalculator.out.link(xout.input)

                xin = pipeline.createXLinkIn()
                xin.setStreamName("spatialConfig")
                xin.out.link(self.spatialLocationCalculator.inputConfig)

                xout = pipeline.createXLinkOut()
                xout.setStreamName("depth")
                self.spatialLocationCalculator.passthroughDepth.link(xout.input)
                depth_xout = xout

            else:
                xout = pipeline.createXLinkOut()
                xout.setStreamName("depth")
                self.depth_node.depth.link(xout.input)
                depth_xout = xout

        self._device = dai.Device(pipeline)
        self._device.startPipeline()

        nn_queue = {}
        cam_queue = {}

        for nn in nn_xout:
            nn_queue[nn] = self._device.getOutputQueue(nn, maxSize=1, blocking=False)
        
        for cam in cam_xout:
            cam_queue[cam] = self._device.getOutputQueue(cam, maxSize=1, blocking=False)


        self._oak_input_queue = {}

        for link in self.input_link:
            self._oak_input_queue[link] = self._device.getInputQueue(link, maxSize=1, blocking=False)

        if self._configs["depth"]:
            self._depth_queue = self._device.getOutputQueue("depth",maxSize=1, blocking=False)

            if self._configs["depth_host"] == False:
                self._spatial_queue = self._device.getOutputQueue("spartialData",maxSize=1, blocking=False)
                self._oak_input_queue["spatialConfig"] = self._device.getInputQueue("spatialConfig")

        self._nn_queue = nn_queue
        self._cam_queue = cam_queue

    

    def process(self):
        '''!
            Process the data received from the OAK.

            Decodes the neural results and pass with the camera images
            to the outputs channels
        '''
        try:
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

            if self._configs["depth"]:
                depth_output = self._depth_queue.tryGet()

                if depth_output is not None:
                    self._depth_output = depth_output

                    depth = depth_output.getCvFrame()

                    if  self._configs["depth_filtering"] == "bilateral":
                        depth = depth.astype(np.float32)
                        depth = cv.bilateralFilter(depth, self.d, self.sigma, self.sigma)
                        depth = depth.astype(np.uint16)
                    elif self._configs["depth_filtering"] == "median":
                        depth = depth.astype(np.float32)
                        depth = cv.medianBlur(depth, 5)

                        depth = depth.astype(np.uint16)

                    

                    depth_map = Depth_Map(self._configs["frame_id"], depth.astype(np.float64)/1000.0)
                    depth_img = Image(self._configs["frame_id"], (depth/np.max(depth))*256)

                    self._depth_frame = depth
                    self._depth_map = depth_map

                    self._setOutput(depth_map,"depth_map")
                    self._setOutput(depth_img,"depth_img")


            for nn in self._nn_list:
                self._nn_list[nn].decode_result(self)
        except KeyboardInterrupt as ki:
            raise ki
        except Exception as err:
            if self._configs["force_reconnection"]:
                self.setup()
            else:
                raise err

    def get3DPosition(self, point_x,point_y, size):
        '''!
            @todo Alterar setup para alinhar o depth com o centro, e utilizar intrisics apropriadas
        '''

        try:
            self._depth_frame
        except:
            return np.array([np.inf,np.inf,np.inf] ,dtype=np.float64)

        point_x *= self._depth_frame.shape[0]/size[0]
        point_y *= self._depth_frame.shape[1]/size[1]

        x_pixel = np.array([point_x,point_y,1.0],dtype=np.float64)

        point_x = int(point_x)
        point_y = int(point_y)


        x_space = np.zeros(3,dtype=np.float64)

        n_pixel = 0

        scale = [self._depth_frame.shape[0]/OAK_Stage.color_calib_size[0], self._depth_frame.shape[1]/OAK_Stage.color_calib_size[1]]


        K = OAK_Stage.color_intrinsic.copy()
        K[0] *= scale[0]
        K[1] *= scale[1]

        k_inv = np.linalg.inv(K)

        n_point = self._configs["depth_roi_size"]
        
        x_min = point_x-(n_point//2)-1
        x_max = point_x+(n_point//2)

        y_min = point_y-(n_point//2)-1
        y_max = point_y+(n_point//2)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max >= self._depth_frame.shape[0]:
            x_max = self._depth_frame.shape[0]
        if y_max >= self._depth_frame.shape[1]:
            y_max = self._depth_frame.shape[1]

        if self._configs["depth_host"] == False:
            topLeft = dai.Point2f(x_min, y_min)
            bottomRight = dai.Point2f(x_max, y_max)

            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(topLeft, bottomRight).normalize(width=self._depth_frame.shape[1], height=self._depth_frame.shape[0])

            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)

            self._oak_input_queue["spatialConfig"].send(cfg)

            spatialdata = self._spatial_queue.get()

            x_space = np.zeros(3, dtype = np.float64)

            x_space[0] = spatialdata[0].spatialCoordinates.x
            x_space[1] = spatialdata[0].spatialCoordinates.y
            x_space[2] = spatialdata[0].spatialCoordinates.z

            return []
        
        samples = self._depth_frame[x_min:x_max,y_min:y_max].flatten()
        samples = samples[samples != 0]
        samples = samples[samples>100]

        if samples.shape[0] == 0:
            return np.array([np.inf,np.inf,np.inf] ,dtype=np.float64)
        
        

        z = 0

        if self._configs["depth_point_mode"] == "median":
            z = np.median(samples)
        elif self._configs["depth_point_mode"] == "mean":
            z = np.mean(samples)
        elif self._configs["depth_point_mode"] == "min":
            samples = np.sort(samples)
            z = np.mean(samples[:int(samples.shape[0]*0.1)])
        elif self._configs["depth_point_mode"] == "max":
            samples = np.sort(samples)
            z = np.median(samples[int(samples.shape[0]*0.9):])
        else:


            hist, edge = np.histogram(samples, bins=1000, range=(0.0,5000.0), density=True)

            a = []

            for i in range(edge.shape[0]-1):
                a.append((edge[i]+edge[i+1])/2)

            edge = np.array(a)
            best_z = 0
            best_z_inliers = 0

            for i in range(16):
                index = int(np.random.rand(1)[0]*samples.shape[0])
                z_test = samples[index]
                
                if i == 0:
                    z_test = np.median(samples)
                if i == 1:
                    z_test = np.mean(samples)
                    
                z_min = z_test-100.0
                z_max = z_test+100.0

                mask = np.logical_and(edge>z_min,edge<z_max)
                values = edge[mask]

                weights = hist[np.where(mask)]

                if np.sum(weights) == 0:
                    continue

                #z_test = np.average(values, weights=weights)
                n_inlier = values.shape[0]

                if n_inlier > best_z_inliers:
                    best_z_inliers = n_inlier
                    best_z = z_test
            
            z = best_z

        if z == 0:
            return np.array([np.inf,np.inf,np.inf] ,dtype=np.float64)

        x_space = k_inv @ (z*x_pixel) 

        return x_space/1000.0
        
    def __del__(self):
        if self._device is not None:
            self._device.close()


