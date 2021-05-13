import time

import cv2 as cv
import numpy as np
import depthai as dai


pipeline = dai.Pipeline()

xout_preview = pipeline.createXLinkOut()
xout_preview.setStreamName("color")

cam_rgb = pipeline.createColorCamera()
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.preview.link(xout_preview.input)

casa = (7,9)
tamanho = 20 #cm
corners_hist = []

with dai.Device(pipeline) as device:

    q_rgb = device.getOutputQueue("color", maxSize=1, blocking=False)


    index = 0
    

    while(index <10):
        rgb_data = q_rgb.tryGet()

        if rgb_data is None:
            continue
            
        img = rgb_data.getCvFrame()
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Verifica se existe um tabuleiro de xadrez na imagem
        ret, corners = cv.findChessboardCorners(gray, casa, None)

        if ret == True:
            index += 1
            corners_hist.append(corners)
            
            print("Tabuleiro Detectado! Faltam "+str(10-index)+" imagens")
            
            time.sleep(0.5)

        


#Cria um vetor com a posição dos cantos 
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

objp *= 20

#Listas para guardar os pontos
objpoints = [objp]*10 #3D = n* objp
imgpoints = corners_hist #Imagem


#Parâmetros: pontos3D, pontos na imagem, img size, matriz da câmera
#Retorno: erro RMS médio de reprojeção, matriz da câmera (K), coeficientes de distorção, rotação das vistas, translação das vistas
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[1::-1], None, None)
mtx

print("\nErro de reprojeção:")
print(ret)
print("\nMatrix da câmera:")
print(mtx)
print("\nCoeficientes de distorção:")
print(dist)