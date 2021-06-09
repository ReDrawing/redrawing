import numpy as np
from scipy.spatial.transform import Rotation

from redrawing.data_interfaces.imu import IMU, Orientation
from redrawing.components.stage import Stage

from redrawing.components.er_ukf_imu_modules.attitude_computation import AttitudeComputation
from redrawing.components.er_ukf_imu_modules.error_compensation import GyroErrorCompensation
from redrawing.components.er_ukf_imu_modules.measurement_handler import MeasurementHandler
from redrawing.components.er_ukf_imu_modules.er_ukf_imu import ErUkfImu

class UKF_IMU(Stage):
    '''!
        @todo UKF_IMU: atualizar estágios apenas quando houver alteração de entradas (ver original em cti_sensors)
    '''
    configs_default = {"gravity": 9.78613,
                        "magneticIntensity": 22902.5e-9,
                        "inclination": -39.2722,
                        "frame_id" : "PASS"}

    def __init__(self, configs={}):

        super().__init__(configs)

        self.addInput("imu", IMU)
        self.addOutput("orientation", Orientation)

        

    def setup(self):
        self._config_lock = True

        self._gravity = self._configs["gravity"]
        self._magneticIntensity = self._configs["magneticIntensity"]
        self._inclination = self._configs["inclination"]

        self._accel = np.array([0.0,0.0,0.0], dtype=np.float64)
        self._gyro = np.array([0.0,0.0,0.0], dtype=np.float64)
        self._mag = np.array([0.0,0.0,0.0], dtype=np.float64)
        self._frame_id = self._configs["frame_id"]
        self._last_time = 0.0
        self._deltaT = 0.0

        self.attitudeComputation = AttitudeComputation()
        self.gyroErrorCompensation = GyroErrorCompensation()
        self.measurementHandler = MeasurementHandler(self._magneticIntensity, self._inclination, self._gravity)
        self.ukf = ErUkfImu()


    def _parse_inputs(self):
        imu = self._getInput("imu")
        self._gyro = imu.gyro
        self._accel = imu.accel
        self._mag = imu.mag
        
        self._frame_id = imu.frame_id
        
        time = imu.time
        self._deltaT = time-self._last_time
        self._last_time = time

        if(self._deltaT == 0):
            self._deltaT = 0.0000001


    def _pass_inputs(self):
        self.gyroErrorCompensation.setMeasuredOmega(self._gyro)
        self.measurementHandler.setAccelRead(np.copy(self._accel))
        self.measurementHandler.setMagRead(self._mag)

    def _kalman_process(self, context={}):
        self.ukf.setMeasurement(self.measurementHandler.getErrorMeasurement())
        self.ukf.setEstimateOmega(self.gyroErrorCompensation.getCorrectedOmega())
        self.ukf.setEstimateTheta(self.attitudeComputation.getTheta())

        try:
            self.ukf.compute(self._deltaT)

            self.attitudeComputation.setThetaError(self.ukf.getThetaError())
            self.gyroErrorCompensation.setPredictedOmegaError(self.ukf.getOmegaError())
        
        except Exception as e:
            print("Filtro gerou excecao: "+str(e)+". Reiniciando filtro")
            self.ukf = ErUkfImu()

    def _orientation_process(self, context={}):
        omega = self.gyroErrorCompensation.getCorrectedOmega()

        self.attitudeComputation.setOmega(omega)
        self.attitudeComputation.computeAll(self._deltaT)

        self._theta = self.attitudeComputation.getTheta()

        self.measurementHandler.setTheta(self._theta)
    
    def thetaToRotation(self, theta):
        return Rotation.from_euler('xyz',theta)

    def thetaToQuat(self, theta):
        r = self.thetaToRotation(theta)
        quat = r.as_quat()

        return quat

    def _update_output(self):
        rotation = self.thetaToRotation(self._theta)

        frame_id = self._frame_id

        if(self._frame_id == "PASS"):
            self._frame_id = self._getInput('imu').frame_id

        orientation = Orientation(frame_id=frame_id)

        orientation.orientation = rotation

        self._setOutput(orientation, "orientation")

    def _compute_PureAccel(self):
        grav = np.array([0,0,self.gravity], dtype=np.float64)
        
        r = self.thetaToRotation(self._theta)

        grav = r.apply(grav, True)

        pureAcceleration = self._accel - grav

    def process(self, context={}):
        self._parse_inputs()
        self._pass_inputs()
        self._kalman_process()
        self._orientation_process()
        self._update_output()
        pass
