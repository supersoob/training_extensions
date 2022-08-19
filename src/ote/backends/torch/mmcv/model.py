import importlib

from mmcls.models.builder import CLASSIFIERS
from mmcv.utils import build_from_cfg
from mmcv.utils.config import Config
from ote.backends.torch.model import TorchModel
from ote.core.config import OTEConfig
from ote.core.model import ModelAdapter, OTEModel
from ote.utils import import_and_get_class_from_path


class MMModelAdapter(ModelAdapter):
    def __init__(self, model_config: OTEConfig):
        registry_path = model_config.pop("registry")

        # get registry class from full path of registry 
        self.task = model_config.pop("task")
        self.registry = import_and_get_class_from_path(registry_path)
        self.model_config = model_config

        # model = build_from_cfg(model_config, reg_cls)
    
    def update_model(options: dict):
        self.model_config.merge_from_dict(options)

    @property
    def model(self):
        _model = build_from_cfg(self.model_config, self.registry)
        return MMModel(_model)


class MMModel(OTEModel):
    def __init__(self, model):
        super().__init__(model)

    def save(self):
        pass

    def export(self):
        pass
