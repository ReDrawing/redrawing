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
        '''!
            Initiate the stage.

            Must be called by the subclass constructor

            @todo Verificar alterações na configuração padrão recursivamente
        '''

        self.input_dict = {}
        self.output_dict = {}
        
        self.output_queue = {}
        self.input_queue = {}
        self.output_size = {}
        self._output_changed = {}
        self._input_changed = {}

        self.input_values = {}
        self.output_values = {}

        new_configs = deepcopy(type(self).configs_default)

        for index in configs:
            if configs[index] != new_configs[index]:
                new_configs[index] = configs[index]

        self._configs = new_configs
        self._config_lock = False

    def setup(self):
        '''!
            Do the settings need to start the stage.

            Must be changed by subclass as needed
        '''
        self._config_lock = True
    
    def change_config(self, config_key, new_value):
        '''!
            Change a configuration of the stage.

            Parameters:
                @param config_key : String = The config that will be changed
                @param new_value = The new value for the config
        '''
        if self._config_lock == False:
            self._configs[config_key] = new_value


    def addInput(self, id, inType):
        '''!
            Add a input channel for the stage.
            
            Parameters:
                @param id : String = The name of the channel
                @param inType : Class = The type of the input
        '''

        self.input_dict[id] = inType
        self.input_values[id] = None
        self._input_changed[id] = False

    def addOutput(self, id, outType):
        '''!
            Add a output channel for the stage
            
            Parameters:
                @param id : String = The name of the channel
                @param outType : Class = The type of the output
        '''

        self.output_dict[id] = outType
        self.output_values[id] = None
        self.output_changed[id] = False

    def setInput(self, value, id):
        '''!
            Set an input for the stage.

            It checks if the type of the input is correct

            Parameters:
                @param value = The value to be passed to the input channel
                @param id : String = The id of the channel
        '''

        if not isinstance(value, self.input_dict[id]):
            raise ValueError("Incorrect type")
        
        self.input_values[id] = value
        self._input_changed[id] = True

    def _getInput(self, id):
        '''!
            Get the input from the channel

            Parameters:
                @param id : String = The id of the input channel 
        '''

        return self.input_values[id]

    def _setOutput(self, value, id):
        '''!
            Set an output for a channel of the stage.

            It checks if the type of the output is correct

            Parameters:
                @param value = The value to be passed to the output channel
                @param id : String = The id of the output channel
        '''

        if not isinstance(value, self.output_dict[id]):
            raise ValueError("Incorrect type")

        self.output_values[id] = value
        self._output_changed[id] = True

    def getOutput(self, id):
        '''!
            Gets an output from the stage.

            Parameters:
                @param id : String = The id of the output channel
        '''


        return self.output_values[id]

        pass
    
    def _setOutQueue(self, queue, max_size, id):
        self.output_queue[id] = queue
        self.output_size[id] = max_size
    
    def _setInputQueue(self, queue, id):
        self.input_queue[id] = queue

    def _sendOutputs(self):
        for id in self.output_queue:
            if self._output_changed[id] == True:
                self._output_changed[id] = False
                self.output_queue[id].insert(self.output_values[id])
        
    def _getInputs(self):
        for id in self.input_queue:
            if not self.input_queue[id].empty():
                self.setInput(self.input_queue[id].get(), id)

    def has_input(self, id):
        return self._input_changed[id]

    def run(self):
        _getInputs()

        process()

        _sendOutputs(self)

    @abstractmethod
    def process(self):
        '''!
            Do the processing of the Stage.

            Must be implemented by the subclass.
        '''
        pass