from redrawing.data_interfaces.imu import IMU
from redrawing.components.android import IMUReceiver

imuReceiver = IMUReceiver()

while(True):
    imuReceiver.process()