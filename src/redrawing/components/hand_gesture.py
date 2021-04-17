from redrawing.components.stage import Stage
from redrawing.data_interfaces.gesture import Gesture
from redrawing.data_interfaces.bodypose import BodyPose

class HandGesture(Stage):
    '''!
        Detects hand gestures.
    '''

    configs_default = {}

    def __init__(self, configs={}):
        super().__init__(configs)

        self.addOutput("gesture", Gesture)
        self.addInput("bodypose", BodyPose)

    def process(self):
        bodypose = self._getInput("bodypose")

        #Marcela: detectar os gestos e colocar no resultado

        result = Gesture()
        self._setOutput(result, "gesture")