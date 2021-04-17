from abc import ABC, abstractmethod

from redrawing.components.oak_constants import *

class OAK_NN_Model(ABC):

    input_type = COLOR
    input_size = [1920, 1080]

    @abstractmethod
    def create_node(self, oak_stage):
        pass
    
    @abstractmethod
    def decode_result(self, oak_stage):
        pass
    