import os
from abc import abstractmethod
from enum import IntEnum

from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

class ModelStatus(IntEnum):
    CONFIGURED = 0
    BUILT = 1
    CONFIG_UPDATED = 2
    TRAINED = 3
    OPTIMIZED = 4


class ModelSpec(InitEnum):
    Classifier = 0
    Detector = 1
    Segmentor = 2


class IModel:
    def __init__(self, model_config: dict):
        self._config = Config(model_config)
        self._ckpt = None

    @abstractmethod
    def save(self):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def export(self, type="openvino"):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def update_config(self, config: dict):
        raise NotImplementedError()

    @property
    def config(self):
        return self._config

    @property
    def ckpt(self):
        if self._ckpt is not None:
            if not os.path.exists(self._ckpt):
                logger.warning(f"invalid model checkpoint path: {self._ckpt}")
        return self._ckpt
