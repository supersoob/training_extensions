import os
import tempfile
from abc import abstractmethod

from mmcls.models.builder import build_classifier
from mmdet.models.builder import build_detector
from mmseg.models.builder import build_segmentor

from ote.backends.torch.mmcv.config import convert_config
from ote.backends.torch.model import TorchModel
from ote.core.config import Config
from ote.logger import get_logger
from ote.utils import import_and_get_class_from_path

logger = get_logger()

class MMModel(TorchModel):
    def __init__(self, model_config: dict, **kwargs):
        super().__init__(model_config)
        self._config = convert_config(self._config)

    def update_config(self, options: dict, overwrite=False, **kwargs):
        logger.info(f"options = {options}, overwrite = {overwrite}")
        if overwrite:
            self._config = Config(options)
        else:
            self._config.merge_from_dict(options)

    def build(self):
        return self.builder(self._config._cfg_dict)

    @property
    @abstractmethod
    def builder(self):
        raise NotImplementedError()


class MMClassifier(MMModel):
    def __init__(self, model_config: dict, **kwargs):
        super().__init__(model_config, **kwargs)

    @property
    def builder(self):
        return build_classifier


class MMDetector(MMModel):
    def __init__(self, model_config: dict, **kwargs):
        super().__init__(model_config)

    @property
    def builder(self):
        return build_detector


class MMSegmentor(MMModel):
    def __init__(self, model_config: Config, **kwargs):
        super().__init__(model_config)

    @property
    def builder(self):
        return build_segmentor
