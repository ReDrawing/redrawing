from collections import deque
from abc import ABC, abstractmethod
from multiprocessing import Queue as MQueue
from multiprocessing import Process

from redrawing.components.stage import Stage

class Queue(ABC):
    '''!
        Generic queue for communication between stages.
    '''

    def __init__(self, max_size):
        self.max_size = max_size
    
    @abstractmethod
    def get(self):
        '''!
            Returns the first element of the queue.

            Returns:
                @returns the first element of the queue
        '''
        ...
    
    @abstractmethod
    def insert(self, value):
        '''!
            Insert a value in the end of the queue.

            Parameters:
                @param value - the value
        '''

        ...
    
    @abstractmethod
    def empty(self):
        '''!
            See if the queue is empty.

            Returns:
                @returns True if empty
        '''

        ...

    @abstractmethod
    def full(self):
        '''!
            See if the queue if full

            Returns:
                @returns True if full
        '''

        ...

class SimpleQueue(Queue):
    '''!
        A simple queue. Must not be used for multiprocessing

        Uses collections.deque for implement the queue
    '''

    def __init__(self, max_size):
        super().__init__(max_size)
        self.queue = deque(maxlen=max_size)
    
    def get(self):
        '''!
            Returns the first element of the queue.

            Returns:
                @returns the first element of the queue
        '''

        return self.queue.popleft()

    def insert(self, value):
        '''!
            Insert a value in the end of the queue.

            Parameters:
                @param value - the value
        '''

        self.queue.append(value)
    
    def empty(self):
        '''!
            See if the queue is empty.

            Returns:
                @returns True if empty
        '''

        if len(self.queue) > 0:
            return False
        return True

    def full(self):
        '''!
            See if the queue if full

            Returns:
                @returns True if full
        '''

        if len(self.queue) >= self.max_size:
            return True
        return False

class ProcessQueue(Queue):
    '''!
        Queue for using in multiprocessing.

        For single process pipeline, SimpleQueue is better
        Uses multiprocessing.queue for implement the queue
    '''

    def __init__(self, max_size):
        super().__init__(max_size)
        self.queue = MQueue(maxsize=max_size)

    def get(self):
        '''!
            Returns the first element of the queue.

            Returns:
                @returns the first element of the queue
        '''

        return self.queue.get()
    
    def insert(self, value):
        '''!
            Insert a value in the end of the queue.

            Parameters:
                @param value - the value
        '''

        self.queue.put(value)
    
    def empty(self):
        '''!
            See if the queue is empty.

            Returns:
                @returns True if empty
        '''

        return self.queue.empty()
    
    def full(self):
        '''!
            See if the queue if full

            Returns:
                @returns True if full
        '''

        return self.queue.full()

class Pipeline(ABC):
    '''!
        Generic pipeline of stages
    '''

    def __init__(self):
        self.stages = []
        self.substages = []

        self.substages_configs = {}
    
    def insert_stage(self, stage):
        '''!
            Inserts a new stage to the pipeline

            Parameters:
                @param state - the stage
            
            @todo Alterar para o tipo correto de exceção
        '''

        if not isinstance(stage, Stage):
            raise Exception("Stages must be of Stage class")
        
        if stage in self.substages:
            return

        self.stages.append(stage)
        self.substages_configs[stage] = []

    @abstractmethod
    def create_queue(self, max_size):
        ...

    def create_connection(self, stage_out, id_out, stage_in, id_in, max_size):
        '''!
            Create a connection between stages

            Parameters:
                @param stage_out - Stage where the data will come from
                @param id_out - ID of the output communication channel
                @param stage_in - Stage from where the data will go
                @param id_in - ID of the input communication channel
                @param max_size - Maximum channel queue size
        '''
        
        queue = None
        if stage_in.has_input_queue(id_in):
            queue = stage_in.input_queue[id_in]
        else:
            queue = self.create_queue(max_size)
            stage_in._setInputQueue(queue, id_in)

        stage_out._setOutputQueue(queue, id_out)
        
        

    def set_substage(self, superstage, substage, run_before=False):
        if substage not in self.substages:
            self.substages.append(substage)

        if substage in self.stages:
            self.stages.remove(substage)
        
        if not superstage in self.substages_configs:
            self.substages_configs[superstage] = []
        
        self.substages_configs[superstage].append({"substage":substage, "run_before": run_before})

        superstage.substages.append(substage)

    @abstractmethod
    def run(self):
        '''!
            Runs the pipeline until error/exception
        '''

        ...
  

class SingleProcess_Pipeline(Pipeline):
    '''!
        Pipeline of stages to be runned on a single process
    '''

    def __init__(self):
        super().__init__()
        self.started = False

        pass

    def start(self):
        '''!
            Starts the stages

            Is automatically called by the run and runOnce method
        '''

        for stage in self.stages:
            for substage in self.substages_configs[stage]:
                substage["substage"].setup()

            stage.setup()

        self.started = True 
    
    def run(self):
        '''!
            Runs the pipeline until error/exception
        '''

        if not self.started:
            self.start()

        while True:
            self.runOnce()
            
    def runOnce(self):
        '''!
            Runs all the stages once
        '''

        if not self.started:
            self.start()

        for stage in self.stages:
            for substage in self.substages_configs[stage]:
                if substage["run_before"] == True:
                    substage["substage"].run(stage._context)

            stage.run()

            for substage in self.substages_configs[stage]:
                if substage["run_before"] == False:
                     substage["substage"].run(stage._context)


    def create_queue(self, max_size):
        return SimpleQueue(max_size)

class MultiProcess_Pipeline(Pipeline):
    '''!
        Pipeline of stages runned parallely on multiple process
    '''

    def __init__(self):
        super().__init__()
    

    def create_queue(self, max_size):
        return ProcessQueue(max_size)

    def _run_stage(self, stage):
        '''!
            Starts and runs a stage

            Parameters:
                @param stage - stage to be runned
        '''
        
        for substage in self.substages_configs[stage]:
             substage["substage"].setup()

        stage.setup()
        
        while True:
            for substage in self.substages_configs[stage]:
                if substage["run_before"] == True:
                    substage["substage"].run(stage._context)

            stage.run()

            for substage in self.substages_configs[stage]:
                if substage["run_before"] == False:
                    substage["substage"].run(stage._context)

    def run(self):
        '''!
            Run the stages on multiple process.

            Locks the code until the stages end
        '''

        process = []
        
        for stage in self.stages:
            p = Process(target=self._run_stage, args=(stage,))
            p.start()

            process.append(p)

        while 1:
            try:
                pass
            except KeyboardInterrupt:
                break

        print("TERMINANDO")

        for p in process:
            p.terminate()
            p.join(1)
            p.close()
        