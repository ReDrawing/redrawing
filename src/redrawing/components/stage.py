from abc import ABC, abstractmethod

class Stage(ABC):

    input_dict = {}
    output_dict = {}

    input_values = {}
    output_values = {}

    def addInput(self, id, inType):
        input_dict[id] = inType
        input_value[id] = None

    def addOutput(self, id, outtype):
        output_dict[id] = outtype
        output_value[id] = None

    def setInput(self, value, id):
        if not isinstance(value, input_dict[id]):
            raise ValueError("Incorrect type")
        
        input_values[id] = value


    def getOutput(self, id):

        return output_values[id]

        pass

    @abstractmethod
    def process(self):
        pass