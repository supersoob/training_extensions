import os
import tempfile
from abc import abstractmethod

from mmcv import build_from_cfg
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
        self.pretrained = kwargs.get("pretrained")
        self.config = convert_config(self.config)

    def update_model(self, options: dict):
        self.config.merge_from_dict(options)
        logger.info(f"updated model cfg = {self.config}")

    def build(self):
        # TODO: need to pop out attributes not related to the model initialization.
        # is there other ways to separate them? 
        self.task = self.config.pop("task")
        self.configure(load_from=self.pretrained)
        return self.builder(self.config._cfg_dict)

    @property
    @abstractmethod
    def builder(self):
        raise NotImplementedError()

    @abstractmethod
    def configure(self, **kwargs):
        raise NotImplementedError()

class MMClassifier(MMModel):
    def __init__(self, model_config: dict, **kwargs):
        super().__init__(model_config, **kwargs)

    @property
    def builder(self):
        return build_classifier

    def configure(self, **kwargs):
        # verify and update model configurations
        # check whether in/out of the model layers require updating
        cfg = self.config

        if kwargs.get('load_from', None) and cfg.backbone.get('pretrained', None):
            cfg.backbone.pretrained = None

        update_required = False
        if cfg.get('neck') is not None:
            if cfg.neck.get('in_channels') is not None and cfg.neck.in_channels <= 0:
                update_required = True
        if not update_required and cfg.get('head') is not None:
            if cfg.head.get('in_channels') is not None and cfg.head.in_channels <= 0:
                update_required = True
        if not update_required:
            return

        # update model layer's in/out configuration
        input_shape = [3, 224, 224]
        logger.debug(f'input shape for backbone {input_shape}')
        from mmcls.models.builder import BACKBONES as backbone_reg
        layer = build_from_cfg(cfg.backbone, backbone_reg)
        output = layer(torch.rand([1] + input_shape))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output = output.shape[1]
        if cfg.get('neck') is not None:
            if cfg.neck.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.neck is updated from "
                            f"{cfg.model.neck.in_channels} to {output}")
                cfg.model.neck.in_channels = output
                input_shape = [i for i in range(output)]
                logger.debug(f'input shape for neck {input_shape}')
                from mmcls.models.builder import NECKS as neck_reg
                layer = build_from_cfg(cfg.model.neck, neck_reg)
                output = layer(torch.rand([1] + input_shape))
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                output = output.shape[1]
        if cfg.get('head') is not None:
            if cfg.head.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.head is updated from "
                            f"{cfg.model.head.in_channels} to {output}")
                cfg.head.in_channels = output

            # checking task incremental model configurations


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
