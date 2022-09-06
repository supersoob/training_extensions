from abc import abstractmethod

from mmcls.datasets.builder import build_dataset as build_mmcls_dataset
from mmdet.datasets.builder import build_dataset as build_mmdet_dataset
from mmseg.datasets.builder import build_dataset as build_mmseg_dataset

from ote.backends.torch.dataset import TorchDataset
from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

class MMDataset(TorchDataset):
    def __init__(self, data_config: Config, **kwargs):
        super().__init__(data_config)

    def build(self):
        if self.datasets is None:
            self.datasets = dict()
        # self.configure()
        for subset in ["train", "val", "test"]:
            logger.info(f"data config for building = {self._config._cfg_dict}")
            if hasattr(self._config._cfg_dict, subset):
                self.datasets[subset] = self.builder(self._config._cfg_dict.get(subset))
            else:
                logger.warning(f"no attribute {subset} in the config")
        logger.info(f"datasets = {self.datasets}")
        return self.datasets

    @property
    @abstractmethod
    def builder(self):
        raise NotImplementedError()

    def update_config(self, options: dict, overwrite=False, **kwargs):
        logger.info(f"options = {options}, overwrite = {overwrite}")
        if overwrite:
            self._config = Config(options)
        else:
            self._config.merge_from_dict(options)


class MMClsDataset(MMDataset):
    """mmcls dataset dataset adapter"""
    def __init__(self, data_config, **kwargs):
        super().__init__(data_config)

    @property
    def builder(self):
        return build_mmcls_dataset


class MMDetDataset(MMDataset):
    """mmdet dataset dataset adapter"""
    def __init__(self, data_config, **kwrargs):
        super().__init__(data_config)

    @property
    def builder(self):
        return build_mmdet_dataset

class MMSegDataset(MMDataset):
    """mmseg dataset dataset adapter"""
    def __init__(self, data_config, **kwargs):
        super().__init__(data_config)

    @property
    def builder(self):
        return build_mmseg_dataset