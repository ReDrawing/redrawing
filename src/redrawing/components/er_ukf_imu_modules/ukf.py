import abc
from abc import ABC

import numpy as np
from numpy import sin, cos, tan
from scipy.linalg import ldl

class UKF(ABC):
    def __init__(self, nStateVariable, nMeasurementVariable):
        self.nStateVariable = nStateVariable
        self.nMeasurementVariable = nMeasurementVariable

        self.measurement = np.zeros(nMeasurementVariable,dtype=np.float64)

        self.estimateState = np.zeros(nStateVariable, dtype=np.float64)
        self.stateCovariance = np.eye(nStateVariable,dtype=np.float64)

        self.processNoise = np.eye(nStateVariable,dtype=np.float64)
        self.measurementNoise = np.eye(nMeasurementVariable,dtype=np.float64)

        self.N = float(nStateVariable)
        self.k = 3.0-self.N

        self.nSigmaPoint = (2*self.nStateVariable) + 1

    @abc.abstractmethod
    def h(self, x):
        '''
            Measurement model function

            @param x - state

            @return y - the predicted measurement
        '''
        pass

    @abc.abstractmethod
    def f(self, x, deltaT):
        '''
            Motion model

            @param x - last state
            @deltaT - time since last update

            @return x2 - predicted state 
        '''
        pass

    def getState(self):
        return self.estimateState

    def computeSigmaPoint(self, state, covariance):

        L = np.linalg.cholesky(covariance)

        sigmaPoint = np.zeros((self.nSigmaPoint, self.nStateVariable), dtype=np.float64)

        sigmaPoint[0] = state

        for i in range(self.nStateVariable):
            sigmaPoint[i+1] = state + (np.sqrt(self.N+self.k)*L[:,i])

        for i in range(self.nStateVariable):
            sigmaPoint[i+7] = state - (np.sqrt(self.N+self.k)*L[:,i])

        return sigmaPoint

    def compute(self, deltaT):
        '''
            Update estimate state

            @param deltaT - time since last update
        '''

        previousState = self.estimateState
        previousCovariance = self.stateCovariance

        #self.computeNoise()

        ##Predicao##

        sigmaPoint = self.computeSigmaPoint(previousState, previousCovariance)
        
        #Propaga os sigma points
        predictedStateSigma = np.zeros((self.nSigmaPoint, self.nStateVariable), dtype=np.float64)

        for i in range(self.nSigmaPoint):
            predictedStateSigma[i] = self.f(sigmaPoint[i],deltaT)

        #Calcula a predicao da media e covariancia
        
        predictedState = np.zeros(self.nStateVariable,dtype=np.float64)
        predictedCovariance = np.zeros((self.nStateVariable, self.nStateVariable),dtype=np.float64)
        
        for i in range(self.nSigmaPoint):
            alpha = self.computeAlpha(i)

            predictedState += alpha*predictedStateSigma[i]
        
        for i in range(self.nSigmaPoint):
            alpha = self.computeAlpha(i)

            a = (predictedStateSigma[i]-predictedState)
            b = a

            a.shape = (self.nStateVariable,1)
            b.shape = (self.nStateVariable,1)

            predictedCovariance += alpha*(a@b.T)

        predictedCovariance += self.processNoise

        ##Correcao##

        #Propagar sigmaPoint para a medicao
        predictedMeasureSigma = np.zeros((self.nSigmaPoint, self.nMeasurementVariable), dtype=np.float64)

        for i in range(self.nSigmaPoint):
            predictedMeasureSigma[i] = self.h(predictedStateSigma[i])

        #Estimar media e covariancia das prediçoes das medicoes

        predictedMeasure = np.zeros(self.nMeasurementVariable,dtype = np.float64)
        predictedMeasureCov = np.zeros((self.nMeasurementVariable, self.nMeasurementVariable), dtype=np.float64)

        for i in range(self.nSigmaPoint):
            alpha = self.computeAlpha(i)

            predictedMeasure += alpha*predictedMeasureSigma[i]

        for i in range(self.nSigmaPoint):
            alpha = self.computeAlpha(i)

            a = (predictedMeasureSigma[i]-predictedMeasure)
            b = a

            a.shape = (self.nMeasurementVariable, 1)
            b.shape = (self.nMeasurementVariable,1)

            predictedMeasureCov += alpha*(a@b.T)

        predictedMeasureCov += self.measurementNoise

        #Calcular cross-covariance e Kalman gain
        crossCovariance = np.zeros((self.nStateVariable, self.nMeasurementVariable), dtype=np.float64)

        for i in range(self.nSigmaPoint):
            alpha = self.computeAlpha(i)

            a = (predictedStateSigma[i]-predictedState)
            b = (predictedMeasureSigma[i]-predictedMeasure)

            a.shape = (self.nStateVariable, 1)
            b.shape = (self.nMeasurementVariable, 1)

            crossCovariance += alpha*(a@b.T)
        
        kalmanGain = crossCovariance @ np.linalg.inv(predictedMeasureCov)

        #Corrigir media e covariancia
        self.estimateState = predictedState + (kalmanGain@(self.measurement-predictedMeasure))

        self.stateCovariance = predictedCovariance - ((kalmanGain@predictedMeasureCov)@kalmanGain.T)

    def computeAlpha(self, i):
        '''
            Calcula o peso alpha_i

            @param i - o indice do peso que deve ser calculado

            @return alpha_i
        '''

        if(i == 0):
            return self.k/(self.N+self.k)
        else:
            return 0.5*(1/(self.N+self.k))
