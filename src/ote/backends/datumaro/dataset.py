import datumaro as dm

from ote.core.dataset import IDataset


class DMDataset(IDataset):
    """
    """
    def __init__(self, data_config: Config, **kwargs):
        super().__init__(data_config)
        self.path = kwargs.get("path")
        self.format = kwargs.get("format")

    def build(self):
        dm_dataset = dm.Dataset.import_from(self.path, self.format)
        self.datasets["train"] = dm_dataset.get_subset("train")
        self.datasets["val"] = dm_dataset.get_subset("val")
        self.datasets["test"] = dm_dataset.get_subset("test")


    def update_config(self, config):
        pass

    # class DMDatasetWrapper(dm.DatasetSubset):
    #     def __init__(self, subset):
    #         self.subset = subset

    #     def __getitem__(self, id):
    #         return self.subset.get(id)
