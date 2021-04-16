from scipy.spatial.transform import Rotation

from redrawing.data_interfaces.imu import IMU
from redrawing.components.android import IMUReceiver
from redrawing.components.er_ukf_imu import UKF_IMU
from redrawing.communication.udp import send_data

imuReceiver = IMUReceiver()
ukf = UKF_IMU()

imuReceiver.setup()
ukf.setup()


while(True):
    imuReceiver.process()
    imu = imuReceiver.getOutput("imu")


    ukf.setInput(imu,"imu")
    ukf.process()
    orientation = ukf.getOutput("orientation")

    send_data(orientation)

    print("Enviando")

    #r = Rotation.from_quat(orientation.orientation)
    #euler = r.as_euler('xyz')
    #print( imu.gyro, orientation.orientation, euler)
    #print(euler[0],euler[1],euler[2] )