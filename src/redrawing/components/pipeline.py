from collections import deque
from abc import ABC, abstractmethod
from multiprocessing import Queue as MQueue
from multiprocessing import Process

class Queue(ABC):
    def __init__(self, max_size):
        self.max_size = max_size
    
    @abstractmethod
    def get(self):
        ...
    
    @abstractmethod
    def insert(self, value):
        ...
    
    @abstractmethod
    def empty(self):
        ...

    @abstractmethod
    def full(self):
        ...

class SimpleQueue(Queue):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.queue = deque(maxlen=max_size)
    
    def get(self):
        return self.queue.popleft()

    def insert(self, value):
        self.queue.append(value)
    
    def empty(self):
        if len(self.queue) > 0:
            return False
        return True

    def full(self):
        if len(self.queue) >= self.max_size:
            return True
        return False

class ProcessQueue(Queue):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.queue = MQueue(maxsize=max_size)

    def get(self):
        return self.queue.get()
    
    def insert(self, value):
        self.queue.put(value)
    
    def empty(self):
        return self.queue.empty()
    
    def full(self):
        return self.queue.full()

class Pipeline(ABC):
    def __init__(self):
        self.stages = []
    
    def insert_stage(self, stage):
        self.stages.append(stage)
    
    @abstractmethod
    def create_connection(self, stage_out, id_out, stage_in, id_in, max_size):
        ...

    @abstractmethod
    def run(self):
        ...
  

class SingleProcess_Pipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.started = False

        pass

    def start(self):
        for stage in self.stages:
            stage.setup()

        self.started = True 
    
    def run(self):
        if not self.started:
            self.start()

        while True:
            self.runOnce()
            
    def runOnce(self):
        if not self.started:
            self.start()

        for stage in self.stages:
                stage.run()

    def create_connection(self, stage_out, id_out, stage_in, id_in, max_size):
        d = SimpleQueue(max_size)
        
        stage_out._setOutputQueue(d, id_out)
        stage_in._setInputQueue(d, id_in)

class MultiProcess_Pipeline(Pipeline):
    def __init__(self):
        super().__init__()
    
    def create_connection(self, stage_out, id_out, stage_in, id_in, max_size):
        d = ProcessQueue(max_size)
        
        stage_out._setOutputQueue(d, id_out)
        stage_in._setInputQueue(d, id_in)

    def _run_stage(self, stage):
        stage.setup()
        
        while True:
            stage.run()

    def run(self):
        process = []
        
        for stage in self.stages:
            p = Process(target=self._run_stage, args=(stage,))
            p.start()

            process.append(p)

        for p in process:
            p.join()
        