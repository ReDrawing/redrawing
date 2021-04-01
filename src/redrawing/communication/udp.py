import socket
import time
from redrawing.data_interfaces.data_class import Data

def send_data(data):
    '''!
        Sends the message by UDP

        Parameters:
            @param data (data_interfaces.Data): the data object that will be sent

        @todo udp.py - Implementar classe adequada: ela deve possuir capacidade de alterar endereço ip e porta de envio. Singleton?
    '''

    if not isinstance(data, Data):
        raise Exception()
    
    msg = data.toMessage()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, ("127.0.0.1", 6000))
