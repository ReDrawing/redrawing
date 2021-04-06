import urllib.request
from urllib.error import URLError
import json

import numpy as np

from redrawing.components.stage import Stage
from redrawing.data_interfaces.imu import IMU

class IMUReceiver(Stage):

    def __init__(self, ip="192.168.2.101", port="8080", frame_id="UNKNOW"):

        super().__init__()
        self.addOutput("imu", IMU)

        self._ip = ip
        self._port = port
        self._frame_id = frame_id
    
    def process(self):
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