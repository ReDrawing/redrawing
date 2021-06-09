import time

from .stage import Stage

class Debug_Stage(Stage):
    configs_default={"name":"debug_stage", "blank_line":False, "wait_key": False, "wait_seconds": 0, "context_debug":"context"}

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

        self.set_context("context_debug", self._configs["context_debug"])
    
    def process(self, context={}):
        print(self._configs["name"], "process")
        
        if "context_debug" in context:
            print(context["context_debug"])

        if self._configs["blank_line"]:
            print()
        if self._configs["wait_key"]:
            input("Type anything to continue: ")
        if self._configs["wait_seconds"] != 0:
            time.sleep(self._configs["wait_seconds"])

        

