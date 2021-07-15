
from redrawing.components.pipeline import MultiProcess_Pipeline, SingleProcess_Pipeline
from redrawing.components.debug import Debug_Stage

if __name__ == '__main__':

    superstage = Debug_Stage(configs={"name":"superstage", "context_debug":"superstage_context"})
    
    before_substage = Debug_Stage(configs={"name":"before_substage"})
    after_substage = Debug_Stage(configs={"name":"after_substage", "blank_line": True, "wait_seconds":1})


    pipeline = SingleProcess_Pipeline()

    pipeline.insert_stage(before_substage)
    pipeline.insert_stage(superstage)

    pipeline.set_substage(superstage, before_substage, True)
    pipeline.set_substage(superstage, after_substage)

    pipeline.insert_stage(after_substage)

    pipeline.run()