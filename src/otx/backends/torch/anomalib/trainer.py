from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
)
from pytorch_lightning import Trainer, seed_everything

from otx.algorithms.anomaly.adapters.anomalib.callbacks import (
    ProgressCallback,
    ScoreReportingCallback,
)
from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.api.dataset import Dataset
from otx.backends.torch.job import TorchJob
from otx.utils.logger import get_logger

logger = get_logger()

class AnomalibTrainer(TorchJob):
    def run(self, model, data_module, recipe, **kwargs):

        trainer = Trainer(recipe.train, logger=logger, callbacks=callbacks)
