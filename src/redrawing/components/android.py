import urllib.request
from urllib.error import URLError
import json

import numpy as np
import cv2 as cv

from redrawing.components.stage import Stage
from redrawing.data_interfaces.imu import IMU
from redrawing.data_interfaces.image import Image

class CameraReceiver(Stage):
    '''!
        Receives frames from a IP camera
    '''

    configs_default = {"ip": "192.168.2.101",
                        "port": "8080",
                        "frame_id": "UNKNOW"}

    def __init__(self, configs={}):
        '''!
            Constructor

            @param configs:
                ip: IP camera address (default: 192.168.2.101)
                port: IP camera port (default: 8080)
                frame_id: frame id (default: UNKNOW)
        '''
        super().__init__(configs)
        self.addOutput("frame", Image)
        
    
    def setup(self):
        '''!
            Initializes the stage
        '''
        self._config_lock = True

        self._ip = self._configs["ip"]
        self._port = self._configs["port"]
        self._frame_id = self._configs["frame_id"]

        url = "http://"+self._ip+":"+self._port+"/video"

        self._cap = cv.VideoCapture(url)
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 0)


    def process(self, context={}):
        '''!
            Receives the frame and puts it in the output
        '''
        
        ret, frame = self.cap.read()

        img = Image(image=frame, frame_id=self._frame_id)

        self._setOutput(img, "frame")

class IMUReceiver(Stage):
    '''!
        Receives data from a IMU sensor in a smartphone 

        Use with 'IP Webcam' android app
    '''

    configs_default = {"ip": "192.168.2.101",
                        "port": "8080",
                        "frame_id": "UNKNOW"}

    def __init__(self, configs={}):
        '''!
            Constructor

            @param configs:
                ip: smartphone address (default: 192.168.2.101)
                port: smartphone port (default: 8080)
                frame_id: smartphone IMU frame id (default: UNKNOW)
        '''

        super().__init__(configs)
        self.addOutput("imu", IMU)
    
    def setup(self):
        '''!
            Initializes the stage
        '''

        self._config_lock = False

        self._ip = self._configs["ip"]
        self._port = self._configs["port"]
        self._frame_id = self._configs["frame_id"]

    
    def process(self, context={}):
        '''!
            Receives the IMU data and puts it in the output

            Receive accelerometer, gyroscope and magnetometer data
        '''
        url = "http://"+self._ip+":"+self._port+"/sensors.json"

        time = None
        accel = None
        gyro = None
        mag = None
        

        try:
            with urllib.request.urlopen(url) as req:
                data = req.read()
                dataDict = json.loads(data)

                time = dataDict["accel"]["data"][-1][0]
                accel = dataDict["accel"]["data"][-1][1]
                gyro = dataDict["gyro"]["data"][-1][1]
                mag = dataDict["mag"]["data"][-1][1]

        except URLError:
            print("Nao foi possivel conectar. O endereco ip foi definido corretamente?")
        except KeyError:
            print("O celular esta conectado?")
        except ConnectionResetError:
            print("Conexao cancelada. O celular foi desconectado?")

        imu = IMU(frame_id=self._frame_id)
        
        for i in range(len(mag)):
            mag[i] *= 0.000001

        if not time is None:
            imu.time = float(time)/1000.0
        if not accel is None:
            imu.accel = accel
        if not gyro is None:
            imu.gyro = gyro
        if not mag is None:
            imu.mag = mag

        self._setOutput(imu, "imu")