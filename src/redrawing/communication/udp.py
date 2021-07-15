import socket
import time
from redrawing.data_interfaces.data_class import Data
from redrawing.components.stage import Stage

class UDP_Stage(Stage):
    '''!
        Stage for exchange data using UDP protocol
    '''

    configs_default = { "ip" : "127.0.0.1",
                        "port" : 6000,
                        "inputs_list": [],
                        "inputs": []}

    def __init__(self, configs={}):
        '''!
            Constructor

            Parameters:
                @param configs - configs dictionary
                    ip - the ip for the UDP connection (default 127.0.0.1)
                    port - the port for the UDP connection (default 6000)
                    inputs - the inputs channels of the stage (default [])
                    inputs_list - the inputs channels with list data of the stage (default [])
        '''
        super().__init__(configs=configs)

        for input_channel in self._configs["inputs"]:
            self.addInput(input_channel, Data)

        for input_channel in self._configs["inputs_list"]:
            self.addInput(input_channel, list)

        self.addInput("send_msg", Data)
        self.addInput("send_msg_list", list)

    def setup(self):
        '''!
            Initializes the stage.
        '''

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

        data_list = []

        if self.has_input("send_msg"):
            dataIn = self._getInput("send_msg")
            data_list.append(dataIn)
        
        if self.has_input("send_msg_list"):
            dataIn = self._getInput("send_msg_list")
            for data in dataIn:
                data_list.append(data)

        for input_channel in self._configs["inputs"]:
            if self.has_input(input_channel):
                dataIn = self._getInput(input_channel)
                data_list.append(dataIn)

        for input_channel in self._configs["inputs_list"]:
            if self.has_input(input_channel):
                dataIn = self._getInput(input_channel)
                for data_item in dataIn:
                    data_list.append(data_item)
        
        for data in data_list:
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
