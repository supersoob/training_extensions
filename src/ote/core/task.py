from abc import abstractmethod

from mmcv.utils import Registry, build_from_cfg
from ote.core.config import OTEConfig

TASK_REGISTRY = Registry("ote-task")

class OTETask():
    """"""
    def __init__(self, spec):
        """"""
        self.spec = spec

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError()
