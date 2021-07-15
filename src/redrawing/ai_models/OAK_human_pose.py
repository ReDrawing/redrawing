import os

import depthai as dai
import numpy as np
import cv2
from redrawing.components.oak import OAK_Stage

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.components.oak_substage import OAK_Substage
import redrawing.ai_models.third_models.oak_openvino_humanpose as oak_openvino_humanpose
from redrawing.ai_models.third_models.oak_openvino_humanpose.pose import getKeypoints, getValidPairs, getPersonwiseKeypoints

#https://github.com/luxonis/depthai-experiments/blob/master/gen2-human-pose/main.py

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

keypointDict = {'Nose'  : "NOSE"      , 
                    'Neck'  : "NECK"      ,
                    'R-Sho' : "SHOULDER_R", 
                    'R-Elb' : "ELBOW_R"   , 
                    'R-Wr'  : "WRIST_R"   ,   
                    'L-Sho' : "SHOULDER_L", 
                    'L-Elb' : "ELBOW_L"   , 
                    'L-Wr'  : "WRIST_L"   ,
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


class OAK_BodyPose(OAK_Substage):

    configs_default = {"3d": False}

    def __init__(self, configs):
        if configs["3d"]:
            uses_depth = True

        super().__init__(configs, name="bodypose", color_input_size=[456,256], uses_depth=uses_depth)

        self.addOutput("bodypose_list", list)

        if self._configs["3d"]:
            self.addOutput("bodypose3d_list", list)
            

    def create_nodes(self, pipeline):
        base_path = os.path.abspath(oak_openvino_humanpose.__file__)
        base_path = base_path[:-11]
        blob_path = base_path + "human-pose-estimation-0001_openvino_2021.2_6shave" +".blob"

        nn_node = pipeline.createNeuralNetwork()
        nn_node.setBlobPath(blob_path)
        nn_node.setNumInferenceThreads(2)
        nn_node.input.setQueueSize(1)
        nn_node.input.setBlocking(False)

        return {"bodypose": nn_node}

    def link(self, pipeline, nodes, rgb_out, left_out, right_out, depth_out):
        rgb_out.link(nodes["bodypose"].input)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("bodypose")
        nodes["bodypose"].out.link(xout.input)

        return {"bodypose": xout}

    def create_output_queues(self, device):
        queue = device.getOutputQueue("bodypose", maxSize=1, blocking=False)

        return {"bodypose" : queue}
    
    def process(self, context):
        inference = context["output_queues"]["bodypose"].tryGet()    

        if inference is None:
            return

        poses = process_output(inference, 256, 456)

        bodyposes = []
        bodyposes3d = []

        for pose in poses:
            bodypose = BodyPose(pixel_space=True, frame_id=context["frame_id"])

            if self._configs["3d"]:
                bodypose3d = BodyPose()

            for keypoint_name in pose:
                keypoint_key = keypointDict[keypoint_name]

                x = pose[keypoint_name][0]
                y = pose[keypoint_name][1]
                z = 1.0
                bodypose.add_keypoint(keypoint_key, x, y, z)

                if self._configs["3d"]:
                    x_space = context["get3DPosition"](x,y, np.flip(self.input_size[OAK_Stage.COLOR]))
                    x = x_space[0]
                    y = x_space[1]
                    z = x_space[2]

                    bodypose3d.add_keypoint(keypoint_key, x, y, z)

            bodyposes.append(bodypose)

            if self._configs["3d"]:
                bodyposes3d.append(bodypose3d)
    
        self._setOutput(bodyposes, "bodypose_list")

        if self._configs["3d"]:
            self._setOutput(bodyposes3d, "bodypose3d_list")

def process_output(nn_output, h, w):
    heatmaps = np.array(nn_output.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
    pafs = np.array(nn_output.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
    heatmaps = heatmaps.astype('float32')
    pafs = pafs.astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    for row in range(18):
        probMap = outputs[0, row, :, :]
        probMap = cv2.resize(probMap, (w, h))  # (456, 256)
        keypoints = getKeypoints(probMap, 0.3)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
    newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

    detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)
    
    poses = []

    for i in range(len(personwiseKeypoints)):
        pose = {}
        for j in range(18):
            if personwiseKeypoints[i][j] == -1:
                continue
            
            index = int(personwiseKeypoints[i][j])
            pose[keypointsMapping[j]] = keypoints_list[index][:2]

        poses.append(pose)
    return poses