import numpy as np
from numpy import sin, cos, tan
from scipy.linalg import ldl
from .ukf import UKF

class ErUkfImu(UKF):
    def __init__(self):
        UKF.__init__(self, 6, 3)


        I = np.eye(3, dtype=np.float64)
        O = np.zeros((3,3), dtype=np.float64)
        self.H = np.concatenate((I,O), axis=1)

        self.processNoise = np.eye(6,dtype=np.float64)*(72e-1) #Q
        self.measurementNoise = np.eye(3,dtype=np.float64)*(72e-1) #R



    def h(self, x):
        return self.H@x

    def computeWb(self, x):
        '''!
            Calcula a matriz Wb descrita na equação (6) de [3], que possibilita computar dtheta_dt
        '''

        wb = np.eye(3, dtype=np.float64)

        wb[0][1] = sin(x[0])*tan(x[1])
        wb[0][2] = cos(x[0])*tan(x[1])
        
        wb[1][1] = cos(x[0])
        wb[1][2] = -sin(x[0])

        wb[2][1] = sin(x[0])/cos(x[1])
        wb[2][2] = cos(x[0])/cos(x[1])

        return wb

    def computeVb(self, thetaVetor, omega):
        #equacao 11, Intertial Head-Tracker Sensor Fusion by a Complementary Separate-Bias Kalman Filter 

        psi = thetaVetor[0]
        theta = thetaVetor[1]
        phi = thetaVetor[2]

        wX = omega[0]
        wY = omega[1]
        wZ = omega[2]

        vB = np.zeros((3,3), dtype=np.float64)

        vB[0][0] = ((cos(psi)*sin(theta)*wY)/cos(theta)) - ((sin(psi)*sin(theta)*wZ)/cos(theta))
        vB[0][1] = ((sin(psi)*wY)/np.power(cos(theta), 2)) + ((cos(psi)*wZ)/np.power(cos(theta), 2))

        vB[1][0] = -(sin(psi)*wY) - (cos(psi)*wZ)

        vB[2][0] = ((cos(psi)*wY)/cos(theta)) - ((sin(psi)*wZ)/cos(theta))
        vB[2][1] = ((sin(psi)*sin(theta)*wY)/np.power(cos(theta),2))+((cos(psi)*sin(theta)*wZ)/np.power(cos(theta),2))

        return vB

    def f(self, x, deltaT):
        wB = self.computeWb(x[0:3])
        vB = self.computeVb(x[0:3],x[3:6])


        theta = x[0:3]
        w = x[3:6]

        theta2 = theta
        theta2 += wB@(w)*deltaT

        theta2 += ((vB@wB)@w)*np.power(deltaT,2)

        x2 = np.hstack((theta2, w))

        return x2

        '''A = np.zeros((6,6),dtype=np.float64)

        phi = self.estimateTheta[0]
        theta = self.estimateTheta[1]
        psi = self.estimateTheta[2]
        wX = self.estimateOmega[0]
        wY = self.estimateOmega[1]
        wZ = self.estimateOmega[2]

        A[0][0] = tan(theta)*((cos(phi)*wY)-(sin(phi)*wZ))
        A[0][1] = (1+np.power(tan(theta),2))*((sin(phi)*wY)+(cos(phi)*wZ))
        A[0][3] = 1
        A[0][4] = sin(phi)*tan(theta)
        A[0][5] = cos(phi)*tan(theta)

        A[1][0] = -((sin(phi)*wY)+(cos(phi)*wZ))
        A[1][4] = cos(phi)
        A[1][5] = -sin(phi)

        A[2][0] = (1/cos(theta))*((cos(phi)*wY)-(sin(phi)*wZ))
        A[2][1] = (sin(theta)/np.power(cos(theta),2))*((sin(phi)*wY)+(cos(phi)*wZ))
        A[2][4] = sin(phi) / cos(theta)
        A[2][5] = cos(phi)/cos(theta)

        deltaX = A@x

        return x + (deltaX*deltaT)'''

    def computeNoise(self):
        wMax = np.max(self.estimateOmega)

        sigmaW2 = (10**-8)*(1+np.power(wMax,2)+((10**-4)*np.power(wMax,6)))

        QTil = np.eye(3,dtype=np.float64)*sigmaW2

        wB = self.computeWb(self.estimateTheta)

        self.processNoise[3:,3:] = wB@QTil@wB.T

        sigmaV = 5

        self.measurementNoise = np.eye(3,dtype=np.float64)*np.power(sigmaV,2)

    def setEstimateTheta(self, theta):
        self.estimateTheta = theta

    def setEstimateOmega(self, omega):
        self.estimateOmega = omega

    def setMeasurement(self, measurement):
        self.measurement = measurement

    def getState(self):
        return self.estimateState

    def getOmegaError(self):
        return self.estimateState[3:6]

    def getThetaError(self):
        return self.estimateState[:3]




