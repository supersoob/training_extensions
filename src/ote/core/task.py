from abc import abstractmethod

from mmcv.utils import Registry, build_from_cfg

from ote.core.config import Config

TASK_REGISTRY = Registry("ote-task")

class ITask():
    """"""
    def __init__(self, spec):
        """"""
        self.spec = spec

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()
