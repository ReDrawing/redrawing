import time

from .stage import Stage

class Debug_Stage(Stage):
    '''!
        Stage for debugging, print messages in setup and process
    '''

    configs_default={"name":"debug_stage", "blank_line":False, "wait_key": False, "wait_seconds": 0, "context_debug":"context"}

    def __init__(self, configs={}):
        '''!
            Constructor

            @param configs:
                name: Stage name, will be printed in the screen (default: "debug_stage")
                blank_line: Print a blank line in the screen (default: False)
                wait_key: Wait for a key to be pressed after print (default: False)
                wait_seconds: Wait for a number of seconds after print (default: 0, no wait)
                context_debug: Word that will be placed in context, can be used for debbunging substages (default: "context") 
        '''

        super().__init__(configs)

    def setup(self):
        '''!
            Intiialize the stage

            Print the name of the stage, and according to the settings, 
            wait for a key to be pressed or print a blank line
        '''
        print(self._configs["name"], "setup")

        if self._configs["blank_line"]:
            print()

        if self._configs["wait_key"]:
            input("Type anything to continue: ")
        
        if self._configs["wait_seconds"] != 0:
            time.sleep(self._configs["wait_seconds"])

        self.set_context("context_debug", self._configs["context_debug"])
    
    def process(self, context={}):
        '''!
            Prints the name of the stage, and according to the settings, 
            wait for a key to be pressed or print a blank line

            If "context_debug" key is in the context, print the value.
        '''
        print(self._configs["name"], "process")
        
        if "context_debug" in context:
            print(context["context_debug"])

        if self._configs["blank_line"]:
            print()
        if self._configs["wait_key"]:
            input("Type anything to continue: ")
        if self._configs["wait_seconds"] != 0:
            time.sleep(self._configs["wait_seconds"])

        

