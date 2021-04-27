import cv2 as cv
import numpy as np

from redrawing.components.pipeline import Pipeline
from redrawing.components.oak import OAK_Stage
from redrawing.communication.udp import UDP_Stage
from redrawing.data_interfaces.bodypose import BodyPose


oak_configs = {"rgb_out" : True, "rgb_resolution": [456, 256], "nn_enable":{"bodypose":True}}
oak_stage = OAK_Stage(oak_configs)

udp_stage = UDP_Stage()

pipeline = Pipeline()

pipeline.insert_stage(oak_stage)
pipeline.insert_stage(udp_stage)

pipeline.create_connection(oak_stage, "bodypose", udp_stage, "send_msg_list", 1)

pipeline.start()
pipeline.run()