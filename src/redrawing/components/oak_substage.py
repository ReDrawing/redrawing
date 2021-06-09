from .stage import Stage
from abc import ABC

from .oak import OAK_Stage


class OAK_Substage(Stage, ABC):
    def __init__(self, configs, name="", color_input_size = [0,0], left_input_size = [0,0], right_input_size = [0,0], uses_depth = False, color_out = False):
        super().__init__(configs=configs)

        self.input_size = {}
        self.input_size[OAK_Stage.COLOR] = color_input_size
        self.input_size[OAK_Stage.LEFT] = left_input_size
        self.input_size[OAK_Stage.RIGHT] = right_input_size
        self.name = name
        self.uses_depth = uses_depth
        self.color_out = color_out
    
    def create_nodes(self, pipeline):
        return {}
    
    def link(self, pipeline, nodes, rgb_out, left_out, right_out, depth_out):
        return {}
    
    def create_output_queues(self, device):
        return {}
    
    def create_input_queues(self, device):
        return {}
