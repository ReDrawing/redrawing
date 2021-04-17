from abc import ABC, abstractmethod
from copy import deepcopy

class Stage(ABC):
    '''!
        @todo Stage: ser possível criar inputs e outputs que são vetores de classes Data
        @todo Stage: verificar se tipo de entrada e saída são instancias de Data
        @todo Stage: adicionar variável para indicar que uma nova entrada foi recebida
    '''

    configs_default = {}

    def __init__(self, configs={}):
        self.input_dict = {}
        self.output_dict = {}

        self.input_values = {}
        self.output_values = {}

        new_configs = deepcopy(type(self).configs_default)

        for index in configs:
            if configs[index] != new_configs[index]:
                new_configs[index] = configs[index]

        self._configs = new_configs
        self._config_lock = False
    
    def setup(self):
        self._config_lock = True
    
    def change_config(self, config_key, new_value):
        if self._config_lock == False:
            self._configs[config_key] = new_value


    def addInput(self, id, inType):
        self.input_dict[id] = inType
        self.output_values[id] = None

    def addOutput(self, id, outType):
        self.output_dict[id] = outType
        self.output_values[id] = None

    def setInput(self, value, id):
        if not isinstance(value, self.input_dict[id]):
            raise ValueError("Incorrect type")
        
        self.input_values[id] = value

    def _getInput(self, id):
        return self.input_values[id]

    def _setOutput(self, value, id):
        if not isinstance(value, self.output_dict[id]):
            raise ValueError("Incorrect type")

        self.output_values[id] = value

    def getOutput(self, id):

        return self.output_values[id]

        pass

    @abstractmethod
    def process(self):
        pass