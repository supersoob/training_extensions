from abc import abstractmethod
from enum import IntEnum

from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

class ModelStatus(IntEnum):
    CONFIGURED = 0
    BUILT = 1
    CONFIG_UPDATED = 2
    OPTIMIZED = 3


class IModel:
    def __init__(self, model_config: dict):
        self.config = Config(model_config)

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
    def update_model(self, config: dict):
        raise NotImplementedError()
