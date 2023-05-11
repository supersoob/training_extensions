"""Balanced sampler for imbalanced data."""
import random

from torch.utils.data.sampler import Sampler

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class RandomBudgetSampler(Sampler):  # pylint: disable=too-many-instance-attributes
    """Balanced sampler for imbalanced data for class-incremental task.

    This sampler is a sampler that creates an effective batch
    In reduce mode,
    reduce the iteration size by estimating the trials
    that all samples in the tail class are selected more than once with probability 0.999

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
    """

    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, drop_last=False, budget_size=300):
        # breakpoint()
        self.batch_size = batch_size
        self.repeat = 1
        if hasattr(dataset, "times"):
            self.repeat = dataset.times
        if hasattr(dataset, "dataset"):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset

        img_indices = list(range(len(self.dataset)))
        self.data_length = len(self.dataset)
        self.budget_size = budget_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        random.shuffle(img_indices)
        self.new_img_indices = random.sample(img_indices, self.budget_size)

    def __iter__(self):
        """Iter."""
        return iter(self.new_img_indices)

    def __len__(self):
        """Return length of selected samples."""
        return self.budget_size
