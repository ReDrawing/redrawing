import time

from .stage import Stage

class Debug_Stage(Stage):
    configs_default={"name":"debug_stage", "blank_line":False, "wait_key": False, "wait_seconds": 0}

    def __init__(self, configs={}):
        super().__init__(configs)

    def setup(self):
        print(self._configs["name"], "setup")

        if self._configs["blank_line"]:
            print()

        if self._configs["wait_key"]:
            input("Type anything to continue: ")
        
        if self._configs["wait_seconds"] != 0:
            time.sleep(self._configs["wait_seconds"])
    
    def process(self):
        print(self._configs["name"], "process")

        if self._configs["blank_line"]:
            print()
        if self._configs["wait_key"]:
            input("Type anything to continue: ")
        if self._configs["wait_seconds"] != 0:
            time.sleep(self._configs["wait_seconds"])
