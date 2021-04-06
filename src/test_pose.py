import cv2 as cv
import numpy as np

from redrawing.components.openpose_light import OpenPose_Light
from redrawing.communication.udp import send_data
from redrawing.data_interfaces.image import Image

cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Câmera não disponível")
    exit()

opl = OpenPose_Light()

while(True):
    ret, frame = cap.read()

    if not ret:
        print("Frame não disponível")
        exit()

    img = Image(image=frame)

    opl.setInput(img, "image")

    opl.process()

    poses = opl.getOutput("bodyposes")

    for pose in poses:
        print("Enviando pose")
        send_data(pose)
        