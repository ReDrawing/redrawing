import time
import pickle
from datetime import datetime

from redrawing.components.stage import Stage
from redrawing.data_interfaces.bodypose import *
from redrawing.data_interfaces.frame import Frame_TF

class Record(Stage):

    configs_default = {"channels":[], "file_name":"record"}

    # Construtor copiado pra ficar aqui de template a toa por enquanto
    def __init__(self, configs={}):
        super().__init__(configs=configs)

        self.dic_times = {}

        for name in self._configs["channels"]:
            self.addInput(name, object)
            

    def setup(self):
        # talvez criar o arquivo aqui, discussão abaixo
        ...


    # Recebe Nome do arquivo com a gravação e lista de canais gravados
    def configuration(self, file_name, channels):
        
        time_now = time.time() # float
        self.dic_times[file_name] = str(time_now)

        for channel in self._configs["channels"]:
        
            if channel.has_input("channels"):
                msg = channel._getInput("channels")
                time_now = time.time()
                self.dct_times[channel] = [msg, time_now]
                changed = True # Ainda não fazemos nada com isso


# Discussão sobre o impacto em performance/uso de memória cache dos salvamentos ciclicos:
#       Sempre que uma mensagem nova chegar, abrimos o arquivo e salvamos a nova informação
#       ao final dele. O problema disso é que ficar fechando e abrindo o arquivo pode impactar 
#       performance sempre que cicla o process().

#       Uma alternativa seria mudar a pipeline para chamar um método de finalização quando 
#       a execução é encerrada, ou mesmo se o usuário clicar em algum botão de salvar e etc
#       então só ai criamos o arquivo pickle (lembrar de adc assinatura depois) e salvamos 
#       o dicionário

# Ainda falta saber o que faz o método setup()

# Após isso tudo, criar a classe Play(), que faz a leitura do arquivo pickle armazenado e 
# formata o envio dos dados na saída. Também implementa a classe Stage, então se entendermos 
# o que tava rolando na classe Record, entender o que estaria na Play() deve ser + facil, acho




        
