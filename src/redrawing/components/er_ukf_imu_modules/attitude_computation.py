import numpy as np
from numpy import sin, cos, tan

class AttitudeComputation():
    def __init__(self):
        self.computedTheta = np.array([0,0,0], dtype=np.float64)
        self.correctedTheta = np.array([0,0,0], dtype=np.float64)
        self.correctedOmega = np.array([0,0,0], dtype=np.float64)
        self.estimateThetaError = np.array([0,0,0], dtype=np.float64)
        

        self.calculated = True

    def computeWb(self):
        '''!
            Calcula a matriz Wb descrita na equação (6) de [3], que possibilita computar dtheta_dt
        '''

        wb = np.eye(3, dtype=np.float64)

        wb[0][1] = sin(self.correctedTheta[0])*tan(self.correctedTheta[1])
        wb[0][2] = cos(self.correctedTheta[0])*tan(self.correctedTheta[1])
        
        wb[1][1] = cos(self.correctedTheta[0])
        wb[1][2] = -sin(self.correctedTheta[0])

        wb[2][1] = sin(self.correctedTheta[0])/cos(self.correctedTheta[1])
        wb[2][2] = cos(self.correctedTheta[0])/cos(self.correctedTheta[1])

        return wb

    def computevB(self):
        #equacao 11, Intertial Head-Tracker Sensor Fusion by a Complementary Separate-Bias Kalman Filter 
        psi = self.correctedTheta[0]
        theta = self.correctedTheta[1]
        phi = self.correctedTheta[2]
        wX = self.correctedOmega[0]
        wY = self.correctedOmega[1]
        wZ = self.correctedOmega[2]

        vB = np.zeros((3,3), dtype=np.float64)

        vB[0][0] = ((cos(psi)*sin(theta)*wY)/cos(theta)) - ((sin(psi)*sin(theta)*wZ)/cos(theta))
        vB[0][1] = ((sin(psi)*wY)/np.power(cos(theta), 2)) + ((cos(psi)*wZ)/np.power(cos(theta), 2))

        vB[1][0] = -(sin(psi)*wY) - (cos(psi)*wZ)

        vB[2][0] = ((cos(psi)*wY)/cos(theta)) - ((sin(psi)*wZ)/cos(theta))
        vB[2][1] = ((sin(psi)*sin(theta)*wY)/np.power(cos(theta),2))+((cos(psi)*sin(theta)*wZ)/np.power(cos(theta),2))

        return vB

    def computeDThetaDt(self):
        return self.computeWb() @ self.correctedOmega
    
    def computeDTheta2Dt2(self):
        return self.computevB()@self.computeWb()@self.correctedOmega

    def computeTheta(self, deltaT):
        dThetaDt = self.computeDThetaDt()
        dTheta2Dt2 = self.computeDTheta2Dt2()

        self.computedTheta = self.correctedTheta + (dThetaDt*deltaT) #+ (dTheta2Dt2*np.power(deltaT,2)/2)

    def correctTheta(self):
        '''!
            Calcula o vetor Theta corrigido
        '''

        self.correctedTheta = self.computedTheta + self.estimateThetaError


        self.estimateThetaError = np.array([0,0,0], dtype=np.float64)

    def computeAll(self, deltaT):
        self.computeTheta(deltaT)
        self.correctTheta()

        self.calculated = True

    def setThetaError(self, thetaError):
        self.estimateThetaError = thetaError

        self.calculated = False

    def setOmega(self, omega):
        self.correctedOmega = omega

        self.calculated = False
    
    def getTheta(self):
        for i in range(3):
            if self.correctedTheta[i] > np.pi:
                self.correctedTheta[i] -= 2*np.pi
            elif self.correctedTheta[i] < -np.pi:
                self.correctedTheta[i] += 2*np.pi

        return self.correctedTheta


