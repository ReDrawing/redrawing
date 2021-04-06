import numpy as np
from numpy import arctan2, arccos, arcsin, cos, sin
from scipy.spatial.transform import Rotation

class MeasurementHandler():
    def __init__(self, magneticIntensity=22902.5e-9, inclination=-39.2538, gravity=9.78613):
        self.referenceOrientation = np.array([0,0,0], dtype=np.float64)
        self.measurement = np.array([0,0,0], dtype=np.float64)
        self.weight = np.array([1,1])

        self.correctedTheta = np.array([0,0,0], dtype=np.float64)

        self.calculated = True

        self.magneticIntensity = magneticIntensity
        self.inclination = np.radians(inclination)

        self.gravity = gravity

        self.accel = np.array([0,0,gravity], dtype=np.float64)
        self.mag = np.array([sin(self.inclination), 0, cos(self.inclination)], dtype=np.float64)

        self.r = np.array([0,0,0,1])

    def setMagneticIntensity(self, magneticIntensity):
        self.magneticIntensity = magneticIntensity
    
    def setInclination(self, inclination):
        self.inclination = np.radians(inclination)

    def setGravity(self, gravity):
        self.gravity = gravity

    def setTheta(self, theta):
        self.correctedTheta = theta

        self.calculated = False

    def setAccelRead(self, accel):
        if(abs(np.linalg.norm(accel)-self.gravity)<1):
            self.accel = accel
            self.accel /= np.linalg.norm(self.accel)
            self.calculated = False
            self.weight[0] = 1
        else:
            self.weight[0] = 0.5

        
    
    def setMagRead(self, mag):
        if(abs(np.linalg.norm(mag)-self.magneticIntensity)<1e-5):
            self.mag = mag
            self.mag /= np.linalg.norm(self.mag)
            self.calculated = False
            self.weight[1] = 1
        else:
            self.weight[1] = 0.5   
        

    def computeReference(self):

        B = np.array([sin(self.inclination), 0, cos(self.inclination)], dtype=np.float64)
        A = np.array([0,0,1],dtype=np.float64)

        r, f = Rotation.align_vectors(np.array([A,B]),np.array([self.accel, self.mag]), self.weight)

        self.referenceOrientation = r.as_euler("xyz")
        self.r = r.as_quat()



    def compute(self):
        self.computeReference()

        self.measurement = self.referenceOrientation - self.correctedTheta

        for i in range(3):
            if self.measurement[i] > np.pi:
                self.measurement[i] -= 2*np.pi
            elif self.measurement[i] < -np.pi:
                self.measurement[i] += 2*np.pi

        self.calculated = True

    def getErrorMeasurement(self):
        if self.calculated == False:
            self.compute()

        return self.measurement

    def getReference(self):
        return self.referenceOrientation    


