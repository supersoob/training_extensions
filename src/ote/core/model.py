from abc import abstractmethod

from ote.core.config import OTEConfig


class OTEModel:
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def save(self):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def export(self, type="openvino"):
        """"""
        raise NotImplementedError()


class ModelAdapter:
    @property
    @abstractmethod
    def model(self) -> OTEModel:
        raise NotImplementedError

    @abstractmethod
    def update_model(config: OTEConfig):
        raise NotImplementedError()
