from abc import ABC, abstractmethod

from redrawing.components.oak_constants import *

class OAK_NN_Model(ABC):
    '''!
        Base class to handle NN models running on the OAK.
    '''

    input_type = COLOR
    input_size = [1920, 1080]
    outputs = {}

    @abstractmethod
    def create_node(self, oak_stage):
        '''!
            Creates the nn_node on the pipeline

            Must be implemented by the subclass
        '''

        ...
    
    @abstractmethod
    def decode_result(self, oak_stage):
        '''!
            Decodes the NN result received from the OAK.

            Must be implemented by the subclass
        '''

        ...
    