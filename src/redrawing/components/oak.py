import os
from abc import ABC, abstractmethod

import depthai as dai
import numpy as np
import cv2 as cv

from redrawing.components.stage import Stage
from redrawing.data_interfaces.depth_map import Depth_Map
from redrawing.data_interfaces.image import Image

class OAK_Stage(Stage):
    '''!
        @todo Ler intrinsics da câmera do EEPROM dela, quando implementado pela depthai
    '''

    COLOR = 0
    LEFT = 1
    RIGHT = 2
    DEPTH = 3

    CAMERAS_TYPES = [COLOR, LEFT, RIGHT]

    configs_default = {"frame_id": "oak",
                        "color_resolution": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                        "color_out": False,
                        "color_size": [1280,720],
                        "mono_resolution" : dai.MonoCameraProperties.SensorResolution.THE_400_P,
                        "depth" : False,
                        "depth_close" : False,
                        "depth_far" : False,
                        "depth_filtering" : "",
                        "depth_point_mode" : "median",
                        "depth_roi_size" : 50,
                        "depth_host" : True,
                    }

    d = -1
    sigma = 3

    def __init__(self, configs={}):
        '''!
            OAK_Stage constructor.

            Parameters:
                @param configs : dict = The configurations changes from the defaults configs of the stage
        '''
        super().__init__(configs)


        if(self._configs["color_out"] == True):
            self.addOutput("color", Image)
        if(self._configs["depth"] == True):
            self.addOutput("depth_img", Image)
            self.addOutput("left", Image)
            self.addOutput("right", Image)

            self.addOutput("depth_map", Depth_Map)
        
        self.preview_size = {OAK_Stage.COLOR : [0,0], OAK_Stage.LEFT: [0,0], OAK_Stage.RIGHT : [0,0]}

        self._device = None
        self.pipeline = None

        
    
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

        self.depth = self._configs["depth"]
        self.nodes = {}

        #Set previews sizes and if needs depth

        if self._configs["color_out"]:
            self.preview_size[OAK_Stage.COLOR] = self._configs["color_size"]

        for substage in self.substages:
            for cam in OAK_Stage.CAMERAS_TYPES:
                for i in range(2):
                    if self.preview_size[cam][i] < substage.preview_size[cam][i]:
                        self.preview_size[cam][i] = substage.preview_size[cam][i]
                
            if substage.uses_depth:
                self.depth = True

        needs_manip = {}

        #Instantiate cameras

        if self.preview_size[OAK_Stage.COLOR] != [0,0]:
            color_cam = pipeline.createColorCamera()
            color_cam.setPreviewSize(self.preview_size[OAK_Stage.COLOR][0], self.preview_size[OAK_Stage.COLOR][1])
            color_cam.setResolution(self._configs["color_resolution"])
            color_cam.setInterleaved(False)
            color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)

            if self.depth == True:
                color_cam.initialControl.setManualFocus(130)

            self.nodes[OAK_Stage.COLOR] = color_cam

        if self.preview_size[OAK_Stage.LEFT] != [0,0] or self.depth:
            left_cam = pipeline.createMonoCamera()
            left_cam.setResolution(self._configs["mono_resolution"])
            left_cam.setBoardSocket(dai.CameraBoardSocket.OAK_Stage.LEFT)

            self.nodes[OAK_Stage.LEFT] = left_cam

        if self.preview_size[OAK_Stage.RIGHT] != [0,0] or self.depth:
            right_cam = pipeline.createMonoCamera()
            right_cam.setResolution(self._configs["mono_resolution"])
            right_cam.setBoardSocket(dai.CameraBoardSocket.OAK_Stage.RIGHT)

            self.nodes[OAK_Stage.RIGHT] = right_cam

        #Instantiate other nodes

        for substage in self.substages:
            nodes = substage.create_nodes(pipeline)

            for key in nodes:
                if key in self.nodes[key]:
                    raise RuntimeError("Already exists a node with name "+str(key)+", substage "+type(substage).__name__+" creating another")

                self.nodes[key] = nodes[key]

        #Instantiate depth

        using_nn = False

        for node in self.nodes:
            if isinstance(self.nodes[node], dai.NeuralNetwork):
                using_nn = True
                break

        if self.depth:

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

            self.nodes[OAK_Stage.DEPTH] = depth_node

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

                self.nodes["spartial_location_calculator"] = spatialLocationCalculator

        # Create manips
        self.data_out = {}

        for substage in self.substages:
            self.data_out[substage] = [None,None,None]

            for cam in OAK_Stage.CAMERAS_TYPES:
                if substage.input_size[cam] == [0,0]:
                    continue

                if substage.input_size[cam] != self.preview_size[cam]:
                    img_manip = pipeline.createImageManip()
                    img_manip.setResize(substage.input_size[cam][0], substage.input_size[cam][1])

                    if cam == OAK_Stage.COLOR:
                        self.nodes[cam].preview.link(img_manip.inputImage)
                    else:
                        self.nodes[cam].out.link(img_manip.inputImage)

                    self.nodes[substage.name+"_manip_"+str(cam)] = img_manip

                    self.data_out[substage][cam] = img_manip.out
                else:

                    if cam == OAK_Stage.COLOR:
                        self.data_out[substage][cam] = self.nodes[cam].preview
                    else:
                        self.data_out[substage][cam] = self.nodes[cam].out
        # Link

        self.links = {}

        if self.depth:
            self.nodes[OAK_Stage.LEFT].link(self.nodes[OAK_Stage.DEPTH].left)
            self.nodes[OAK_Stage.RIGHT].link(self.nodes[OAK_Stage.DEPTH].right)

            self.data_out[OAK_Stage.DEPTH] = self.nodes[OAK_Stage.DEPTH].depth


            xout = pipeline.createXLinkOut()
            xout.setStreamName()
            self.depth_node.depth.link(xout.input)
            self.links[OAK_Stage.DEPTH] = xout

            if self._configs["depth_host"] == False:
                xout = pipeline.createXLinkOut()
                xout.setStreamName("spartialData")
                self.nodes["spartial_location_calculator"].out.link(xout.input)

                xin = pipeline.createXLinkIn()
                xin.setStreamName("spatialConfig")
                xin.out.link(self.spatialLocationCalculator.inputConfig)

                self.links["spartialData"] = xout
                self.links["spatialConfig"] = xin

        if self._configs["color_out"]:
            xout = pipeline.createXLinkOut()
            xout.setStreamName(str(OAK_Stage.COLOR))
            self.nodes[OAK_Stage.COLOR].preview.link(xout.input)
            self.links[OAK_Stage.COLOR]

        for substage in self.substages:
            links = substage.link(self.data_out[substage][OAK_Stage.COLOR], 
                            self.data_out[substage][OAK_Stage.LEFT], 
                            self.data_out[substage][OAK_Stage.RIGHT], 
                            self.data_out[OAK_Stage.DEPTH])

            for key in links:
                if key in self.links[key]:
                    raise RuntimeError("Already exists a stream with name "+str(key)+", substage "+type(substage).__name__+" creating another")

                self.links[key] = links[key]

        #Start device

        device = dai.Device(pipeline)
        self._device = device
        device.startPipeline()

        #Queue creation

        self.oak_input_queue = {}
        self.oak_output_queue = {}
        
        if self._configs["color_out"]:
            self.oak_output_queue[OAK_Stage.COLOR] = device.getOutputQueue(str(OAK_Stage.COLOR), maxSize=1, blocking=False)
        
        if self.depth:
            self.oak_output_queue[OAK_Stage.DEPTH] = device.getOutputQueue(str(OAK_Stage.DEPTH), maxSize=1, blocking=False)

            if self._configs["depth_host"] == False:
                self.oak_output_queue["spatialData"] = device.getOutputQueue("spartialData", maxSize=1, blocking=False)
                self.oak_input_queue["spatialConfig"] = device.getInputQueue("spatialConfig", maxSize=1, blocking=False )

                self.links["spartialData"] = xout
                self.links["spatialConfig"] = xin

        for substage in self.substages:
            out_queues = substage.create_output_queues(device)
            in_queue = substage.create_input_queues(device)

            for key in out_queues:
                if key in self.oak_output_queue:
                    raise RuntimeError("Already exists a output queue with name "+str(key)+", substage "+type(substage).__name__+" creating another")

                self.oak_output_queue[key] = out_queues[key]
            
            for key in in_queue:
                if key in self.oak_output_queue:
                    raise RuntimeError("Already exists a input queue with name "+str(key)+", substage "+type(substage).__name__+" creating another")

                self.oak_input_queue[key] = out_queues[key]

        #Calibration read
        self.camera_intrinsics = {}
        self.camera_calibration_size = {}

        calibObj = device.readCalibration()
        M_color, width, height = np.array(calibObj.getDefaultIntrinsics(dai.CameraBoardSocket.RGB))
        M_left = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
        M_right = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))

        self.camera_intrinsics[OAK_Stage.COLOR] = M_color
        self.camera_intrinsics[OAK_Stage.LEFT] = M_left
        self.camera_intrinsics[OAK_Stage.RIGHT] = M_right

        self.camera_calibration_size[OAK_Stage.COLOR] = np.array([width, height])
        self.camera_calibration_size[OAK_Stage.LEFT] = np.array([1280, 720])
        self.camera_calibration_size[OAK_Stage.RIGHT] = np.array([1280, 720])
    

    def process(self, context={}):
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


