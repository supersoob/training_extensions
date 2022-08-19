from mmcls.datasets import BaseDataset as ClsBaseDataset
from ote.backends.torch.dataset import TorchDataset, data_config


class MMClsDataset(ClsBaseDataset):
    def __init__(self, torch_dataset: TorchDataset):
        """convert torch dataset to mmcls dataset"""
