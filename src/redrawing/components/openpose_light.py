import inspect
import os

import numpy as np
import cv2 as cv
import torch

from redrawing.data_interfaces.bodypose import BodyPose


import redrawing.third_models.lightweight_human_modules as lhm 
from ..third_models.lightweight_human_modules.models.with_mobilenet import PoseEstimationWithMobileNet
from ..third_models.lightweight_human_modules.keypoints import extract_keypoints, group_keypoints
from ..third_models.lightweight_human_modules.load_state import load_state
from ..third_models.lightweight_human_modules.pose import Pose, track_poses
from ..third_models.lightweight_human_modules.image_tools import normalize, pad_width

class OpenPose_Light():
    
    ## Dicionário que mapeia os keypoints do modelo para a mensagem
    keypointDict = {'nose'  : "NOSE"      , 
                    'neck'  : "NECK"      ,
                    'r_sho' : "SHOULDER_R", 
                    'r_elb' : "ELBOW_R"   , 
                    'r_wri' : "WRIST_R"   ,   
                    'l_sho' : "SHOULDER_L", 
                    'l_elb' : "ELBOW_L"   , 
                    'l_wri' : "WRIST_L"   ,
                    'r_hip' : "HIP_R"     , 
                    'r_knee': "KNEE_R"    , 
                    'r_ank' : "ANKLE_R"   , 
                    'l_hip' : "HIP_L"     , 
                    'l_knee': "KNEE_L"    , 
                    'l_ank' : "ANKLE_L"   ,
                    'r_eye' : "EYE_R"     , 
                    'l_eye' : "EYE_L"     ,
                    'r_ear' : "EAR_R"     , 
                    'l_ear' : "EAR_L"     }

    ## Lista que mapeia os keypoints do modelo para a mensagem.
    keypointList = list(keypointDict.items())

    def __init__(self, gpu=False):
        self.gpu = gpu

        lhmPath = os.path.abspath(lhm.__file__)
        lhmPath = lhmPath[:-11]
        checkpointPath = lhmPath + "openpose_light.pth"
        checkpoint = torch.load(checkpointPath, map_location='cpu')

        self.net = PoseEstimationWithMobileNet()
        load_state(self.net, checkpoint)

        self.net = self.net.eval()
        if self.gpu :
            self.net = self.net.cuda()

    def imageFormat(self, img, net_input_height_size, stride, pad_value, img_mean, img_scale):
        '''!
            Formata a imagem para o formato necessário para a rede
            
        '''

        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
        
        return padded_img, scale, pad
    
    def do_inference(self, img, upsample_ratio, padded_img):
        '''!
            Realiza a inferência.

            Paramêtros:
                @param upsample_ratio - Tava de redimensionamento
                @param padded_img (numpy array) - Imagem Formatada

            Retorno:
                @return heatmap - Heatmap de onde estão keypoints
                @return pafs - 
                
        '''

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        
        if self.gpu:
            tensor_img = tensor_img.cuda()
        
        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv.INTER_CUBIC)
        
        return heatmaps, pafs
    
    def inference(self, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        '''!
            Formata a imagem original para realizar a infêrencia.

            Paramêtros:
                @param img (numpy array) - Imagem Original
                @param net_input_height_size - Dimensão de altura do input da rede
                @param stride - 
                @param upsample_ratio - 

            Retorno:
                @return heatmap - Heatmap de onde estão keypoints
                @return pafs -
                @return scale - Escala da Imagem modificada
                @return pad - 
                
        '''

        padded_img, scale, pad = self.imageFormat(img, net_input_height_size, stride, pad_value, img_mean, img_scale)

        heatmaps, pafs = self.do_inference(img, upsample_ratio, padded_img)  
        return heatmaps, pafs, scale, pad

    def getPose(self, img, height_size=256):
        '''!
            Realiza a inferência e gera o vetor de poses corporais.

            Parâmetros
                @param img (numpy array) - imagem para realizar a inferência
                @param height_size (int)

            Retorno
                @return poses (list) - vetor de poses
        '''
           
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        heatmaps, pafs, scale, pad = self.inference(img, height_size, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        all_poses = []
        
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            all_poses.append(pose)

        return all_poses

    pass

    def process(self, img):
        
        frame_id = "UNKNOW"

        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image):
            frame_id = img.get_frame_id()
            img = img.get_image()

        poses = self.getPose(img)

        bodyposes = []

        for pose in poses:
            bodypose = BodyPose(pixel_space = True)

            for i in range(18):
            
                if pose.keypoints[i][0] == -1:
                    continue
                
                joint_type = OpenPose_Light.keypointList[i][1]

                bodypose.set_keypoint(joint_type, pose.keypoints[i][0], pose.keypoints[i][1])
            
            bodyposes.append(bodypose)

        return bodyposes

