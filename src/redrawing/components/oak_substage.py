from .stage import Stage
from abc import ABC

from .oak import OAK_Stage


class OAK_Substage(Stage, ABC):
    '''!
        Base class for creating OAK substages.
    '''
    def __init__(self, configs, name="", color_input_size = [0,0], left_input_size = [0,0], right_input_size = [0,0], uses_depth = False, color_out = False):
        '''!
            Constructor.

            Params:
                @param configs: The configurations for this substage.

                @param name: The name of the substage.
                @param color_input_size: The size of the color camera image (default: [0,0], don't use)
                @param left_input_size: The size of the left camera image (default: [0,0], don't use)
                @param right_input_size: The size of the right camera image (default: [0,0], don't use)
                @param uses_depth: Whether or not to use the depth camera (default: False)
                @param color_out: Whether or not to use the color camera output (default: False)
        '''
        
        super().__init__(configs=configs)

        self.input_size = {}
        self.input_size[OAK_Stage.COLOR] = color_input_size
        self.input_size[OAK_Stage.LEFT] = left_input_size
        self.input_size[OAK_Stage.RIGHT] = right_input_size
        self.name = name
        self.uses_depth = uses_depth
        self.color_out = color_out
    
    def create_nodes(self, pipeline):
        '''!
            Create the nodes that the substage uses

            Params:
                @param pipeline: the OAK pipeline

            Returns:
                A dictionary of the nodes that the substage uses. 
        '''
        return {}
    
    def link(self, pipeline, nodes, rgb_out, left_out, right_out, depth_out):
        '''!
            Link the nodes used by the substage.

            Params:
                @param pipeline: the OAK pipeline
                @param nodes: all the nodes in the pipeline
                @param rgb_out: the output of the color camera
                @param left_out: the output of the left camera
                @param right_out: the output of the right camera
                @param depth_out: the output of the depth calculator node

            Returns:
                A dictionary with the XLinkIn and XLinkOut created.
        '''
        return {}
    
    def create_output_queues(self, device):
        '''!
            Create the output queues of this substage nodes.

            Params:
                @param device: the OAK device

            Returns:
                A dictionary with the output queues created.
        '''
        return {}
    
    def create_input_queues(self, device):
        '''!
            Create the input queues of this substage nodes.

            Params:
                @param device: the OAK device

            Returns:
                A dictionary with the input queues created.
        '''
        return {}
