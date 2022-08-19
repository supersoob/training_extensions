from abc import abstractmethod
from collections import OrderedDict

from mmcv.utils import build_from_cfg
from ote.core.config import OTEConfig
from ote.core.task import TASK_REGISTRY
from ote.logger import get_logger

logger = get_logger()



class TaskController():
    """"""
    def __init__(self, task_config: OTEConfig):
        # initialize tasks using task_config
        self.tasks = OrderedDict()
        for config in task_config.tasks:
            cfg = config.copy()
            spec = cfg.get("spec")
            if spec in self.tasks:
                raise ValueError(f"the same {spec} spec cannot be configured again for a recipe")
            self.tasks[spec] = TaskController.build_task(cfg)

    @staticmethod
    def build_task(config):
        return build_from_cfg(config, TASK_REGISTRY)

    def run_tasks(self, **kwargs):
        results = OrderedDict()
        for _type, task in self.tasks.items():
            results[_type] = task.run(**kwargs)
        return results

    def run_task(self, spec, **kwargs):
        results = None
        if spec in self.tasks.keys():
            results = self.tasks[spec].run(**kwargs)
        else:
            logger.warning(f"cannot find a task type '{spec}'")
        return results
