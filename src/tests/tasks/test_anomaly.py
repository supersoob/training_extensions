from typing import Optional

from otx.api.dataset import Dataset
from otx.api.recipe import Recipe

from sc_sdk.entities.datasets import DatasetEntity
from sc_sdk.entities.model import ModelEntity
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask

from utils import convert_sc_dataset_to_datumaro

class TrainingTask(ITrainingTask):
    def __init__(self, task_environment: TaskEnvironment):
        # get some context from the task env
        self.task_environment = task_environment
        task_type = task_environment.model_template.task_type
        model_name = task_environment.model_template.name

        # create a recipe instance for the anomaly task
        self.recipe = Recipe(task_type, model_name)

    def save_model(self, output, ckpt):
        pass

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] =  None,
    ) -> None:

        # convert sc_sdk.entities.DatasetEntity to otx.api.Dataset
        otx_dataset = Dataset(convert_sc_dataset_to_datumaro(dataset))
        self.recipe.seed = seed

        ckpt_path = self.recipe.train(otx_dataset, parameters=train_parameters)

        self.save_model(output_model, ckpt_path)
