from ote.core.config import Config
from ote.core.dataset import IDataset


class TorchDataset(IDataset):
    """"""
    def __init__(self, data_config: Config):
        super().__init__(data_config)