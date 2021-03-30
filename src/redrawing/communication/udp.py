import socket
import time
from redrawing.data_interfaces.data_class import Data

def send_data(data):
    if not isinstance(data, Data):
        raise Exception()
    
    msg = data.toMessage()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, ("127.0.0.1", 6000))
