import socket
import time
from redrawing.data_interfaces.data_class import Data
from redrawing.components.stage import Stage

class UDP_Stage(Stage):
    '''!
        Stage for exchange data using UDP protocol
    '''

    configs_default = { "ip" : "127.0.0.1",
                        "port" : 6000}

    def __init__(self, configs={}):
        super().__init__(configs=configs)

        self.addInput("send_msg", Data)
        self.addInput("send_msg_list", list)

    def setup(self):
        self._config_lock = True
        self.ip = self._configs["ip"]
        self.port = self._configs["port"]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def _send_msg(self, data):
        '''!
            Sends the data. 

            Parameters:
                @param data - the data to be sended

            @todo Criar exceção adequada para quando objeto não for do tipo Data
        '''
        if not isinstance(data, Data):
            raise Exception("data must be of Data class")
        
        msg = data.toMessage()

        self.sock.sendto(msg, (self.ip, self.port))

    def process(self, context={}):
        '''!
            Gets the inputs and send to the address 
        '''

        if self.has_input("send_msg"):
            dataIn = self._getInput("send_msg")
            self._send_msg(dataIn)
        
        if self.has_input("send_msg_list"):
            dataIn = self._getInput("send_msg_list")
            for data in dataIn:
                self._send_msg(data)


def send_data(data):
    '''!
        Sends the message by UDP

        It is preferable to use the UDP_Stage stage

        Parameters:
            @param data (data_interfaces.Data): the data object that will be sent
    '''

    if not isinstance(data, Data):
        raise Exception()
    
    msg = data.toMessage()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, ("127.0.0.1", 6000))
