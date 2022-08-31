from abc import abstractmethod

from mmcls.datasets.builder import build_dataset as build_mmcls_dataset
from mmdet.datasets.builder import build_dataset as build_mmdet_dataset
from mmseg.datasets.builder import build_dataset as build_mmseg_dataset

from ote.backends.torch.dataset import TorchDataset
from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

class MMDataset(TorchDataset):
    def __init__(self, data_config: Config):
        super().__init__(data_config)

    def build(self):
        if self.datasets is None:
            self.datasets = dict()
        self.configure()
        for subset in ["train", "val", "test"]:
            logger.info(f"data config for building = {self.config._cfg_dict}")
            if hasattr(self.config._cfg_dict, subset):
                self.datasets[subset] = self.builder(self.config._cfg_dict.get(subset))
            else:
                logger.warning(f"no attribute {subset} in the config")
        logger.info(f"datasets = {self.datasets}")
        return self.datasets

    @property
    @abstractmethod
    def builder(self):
        raise NotImplementedError()

    def update_data(self, options: dict):
        self.config.merge_from_dict(options)
        logger.info(f"updated data cfg = {self.config}")

    def configure(self, **kwargs):
        # update data configuration using image options
        def configure_split(target):

            def update_transform(opt, pipeline, idx, transform):
                if isinstance(opt, dict):
                    if '_delete_' in opt.keys() and opt.get('_delete_', False):
                        # if option include _delete_=True, remove this transform from pipeline
                        logger.info(f"configure_data: {transform['type']} is deleted")
                        del pipeline[idx]
                        return
                    logger.info(f"configure_data: {transform['type']} is updated with {opt}")
                    transform.update(**opt)

            def update_config(src, pipeline_options):
                logger.info(f'update_config() {pipeline_options}')
                if src.get('pipeline') is not None or \
                        (src.get('dataset') is not None and src.get('dataset').get('pipeline') is not None):
                    if src.get('pipeline') is not None:
                        pipeline = src.get('pipeline', None)
                    else:
                        pipeline = src.get('dataset').get('pipeline')
                    if isinstance(pipeline, list):
                        for idx, transform in enumerate(pipeline):
                            for opt_key, opt in pipeline_options.items():
                                if transform['type'] == opt_key:
                                    update_transform(opt, pipeline, idx, transform)
                    elif isinstance(pipeline, dict):
                        for _, pipe in pipeline.items():
                            for idx, transform in enumerate(pipe):
                                for opt_key, opt in pipeline_options.items():
                                    if transform['type'] == opt_key:
                                        update_transform(opt, pipe, idx, transform)
                    else:
                        raise NotImplementedError(f'pipeline type of {type(pipeline)} is not supported')
                else:
                    logger.info('no pipeline in the data split')

            split = cfg.get(target)
            if split is not None:
                if isinstance(split, list):
                    for sub_item in split:
                        update_config(sub_item, pipeline_options)
                elif isinstance(split, dict):
                    update_config(split, pipeline_options)
                else:
                    logger.warning(f"type of split '{target}'' should be list or dict but {type(split)}")

        cfg = self.config
        logger.info('configure_data()')
        logger.debug(f'[args] {cfg}')
        pipeline_options = cfg.pop('pipeline_options', None)
        if pipeline_options is not None and isinstance(pipeline_options, dict):
            configure_split('train')
            configure_split('val')
            configure_split('test')
            configure_split('unlabeled')

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