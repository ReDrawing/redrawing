import urllib.request
from urllib.error import URLError
import json

import numpy as np
import cv2 as cv

from redrawing.components.stage import Stage
from redrawing.data_interfaces.imu import IMU
from redrawing.data_interfaces.image import Image

class CameraReceiver(Stage):

    def __init__(self, ip="192.168.2.101", port="8080", frame_id="UNKNOW"):
        super().__init__()
        self.addOutput("frame", Image)

        self._ip = ip
        self._port = port
        self._frame_id = frame_id

class IMUReceiver(Stage):

    def __init__(self, ip="192.168.2.101", port="8080", frame_id="UNKNOW"):

        super().__init__()
        self.addOutput("imu", IMU)

        self._ip = ip
        self._port = port
        self._frame_id = frame_id

        url = "http://"+ip+":"+port+"/video"

        self._cap = cv.VideoCapture(url)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
    
    def process(self):
        
        ret, frame = self.cap.read()

        img = Image(image=frame, frame_id=self._frame_id)

        self._setOutput(img, "frame")