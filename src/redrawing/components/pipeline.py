from collections import deque
from abc import ABC, abstractmethod

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

class SimpleQueue():
    def __init__(self, max_size):
        super().__init__(max_size)
        self.queue = deque(max_len=max_size)
    
    def get(self):
        self.queue.popleft()

    def insert(self, value):
        self.queue.append(value)
    
    def empty(self):
        if len(self.queue) >= 0:
            return False
        return True

    def full(self):
        if len(self.queue) >= self.max_size:
            return True
        return False

class ProcessQueue():
    ...

class Pipeline():
    def __init__(self):
        self.stages = []

        pass

    def create_connection(self, stage_out, id_out, stage_in, id_in, max_size):
        d = SimpleQueue(max_size)
        
        stage_out._setOutQueue(d, id_out)
        stage_in._setInQueue(d, id_in)

    def start(self):
        for stage in self.stages:
            stage.setup()

        while True:
            for stage in self.stages:
                stage.run()