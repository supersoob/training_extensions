import torch
import timm
from ote.core.model import OTEModel, ModelAdapter
from ote.core.config import OTEConfig
from ote.logger import get_logger
from torch import nn

logger = get_logger()


class TorchModelAdapter(ModelAdapter):
    def __init__(self, model_config: OTEConfig):
        if not hasattr(model_config, "hub") or not hasattr(model_config, "model"):
            raise ValueError("cannot find 'hub' or 'model' attribute in the model config")
        model = TorchModelAdapter.load_from_hub(model_config.hub, model_config.model)
        self._model = TorchModel(model)

    @property
    def model(self):
        return self._model

    @staticmethod
    def load_from_hub(hub, model, pretrained=True, **kwargs):
        if hub == "timm":
            model = timm.create_model(model, pretrained=pretrained)
        elif hub == "pytorch/vision":
            model = torch.hub.load(hub, model, **kwargs)
        else:
            raise ValueError(f"not supported model hub repo {hub_path}")
        return model

    def update_model(config):
        pass


class TorchModel(OTEModel):
    def __init__(self, model):
        super().__init__(model)

    def save(self):
        pass

    def export(self):
        pass