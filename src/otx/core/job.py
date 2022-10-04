from abc import abstractmethod
from enum import IntEnum

from mmcv.utils import Registry, build_from_cfg

from ote.core.config import Config

JOB_REGISTRY = Registry("otx-job")


class JobSpec(IntEnum):
    Trainer = 0
    Inferrer = 1
    Evaluator = 2


class IJob():
    """"""
    def __init__(self, spec):
        """"""
        self.spec = spec

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def configure(self, recipe: Config, **kwargs):
        """ job specific configuration update
        """
        raise NotImplementedError()