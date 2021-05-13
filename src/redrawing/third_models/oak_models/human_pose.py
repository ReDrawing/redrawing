import os

import depthai as dai
import numpy as np
import cv2

from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.components.oak_constants import *
from redrawing.components.oak_model import OAK_NN_Model
import redrawing.third_models.oak_models as oak_models
from .pose import getKeypoints, getValidPairs, getPersonwiseKeypoints

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

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


class OAK_BodyPose(OAK_NN_Model):

    input_size = [456,256]

    def __init__(self):
        self.node = None
        
        self.input_size = OAK_BodyPose.input_size

    def create_node(self, oak_stage):
        base_path = os.path.abspath(oak_models.__file__)
        base_path = base_path[:-11]
        blob_path = base_path + "human-pose-estimation-0001_openvino_2021.2_6shave" +".blob"

        nn_node = oak_stage.pipeline.createNeuralNetwork()
        nn_node.setBlobPath(blob_path)
        nn_node.setNumInferenceThreads(2)
        nn_node.input.setQueueSize(1)
        nn_node.input.setBlocking(False)

        if(oak_stage.preview_size["rgb"] != [456, 256]):
            img_manip = oak_stage.pipeline.createImageManip()
            img_manip.setResize(456, 256)
            oak_stage.cam["rgb"].preview.link(img_manip.inputImage)
            img_manip.out.link(nn_node.input)
        else:
            oak_stage.cam["rgb"].preview.link(nn_node.input)
        
        self.node = nn_node
        return nn_node

    def decode_result(self, oak_stage):
        nn_output = oak_stage.nn_output["bodypose"]
        if nn_output is not None:
            bd_process_result(oak_stage,nn_output)


def bd_process_result(oak_stage, nn_output):
    poses = process_output(nn_output, 256, 456)

    depth = oak_stage._configs["depth"]

    pixel_space = True
    if depth:
        pixel_space = False

    bodyposes = []
    for pose in poses:
        bodypose = BodyPose(pixel_space=pixel_space, frame_id=oak_stage._configs["frame_id"])

        for keypoint_name in pose:
            keypoint_key = keypointDict[keypoint_name]

            x = pose[keypoint_name][0]
            y = pose[keypoint_name][1]
            z = 1.0

            if depth:
                x_space = oak_stage.get3DPosition(x,y, OAK_BodyPose.input_size)
                x = x_space[0]
                y = x_space[1]
                z = x_space[2]

            bodypose.add_keypoint(keypoint_key, x, y, z)    

        bodyposes.append(bodypose)

    


    oak_stage._setOutput(bodyposes, "bodypose")

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