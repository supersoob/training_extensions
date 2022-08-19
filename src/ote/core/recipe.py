from ote.core.config import OTEConfig
from ote.core.dataset import OTEDataset
from ote.core.model import OTEModel
from ote.core.task_controller import TaskController
from ote.logger import get_logger
from ote.utils import import_and_get_class_from_path

logger = get_logger()

class Recipe:
    def __init__(self, recipe_yaml):
        self.recipe = OTEConfig.fromfile(recipe_yaml)
        logger.info(f"recipe = {self.recipe}")
        self.task_controller = TaskController(self.recipe.task_cfg)
        # initialize adapters
        adapter = self.recipe.model.pop("adapter")
        adapter_class = import_and_get_class_from_path(adapter)
        self.model_adapter = adapter_class(self.recipe.model)

    def train(self, dataset:OTEDataset, **kwargs):
        return self.task_controller.run_task("trainer", model=self.model_adapter.model, dataset=dataset, **kwargs)

    def eval(self, dataset:OTEDataset, **kwargs):
        return self.task_controller.run_task("evaluator", model=self.model_adapter.model, dataset=dataset, **kwargs)

    def infer(self, dataset:OTEDataset,  **kwargs):
        return self.task_controller.run_task("inferrer", model=self.model_adapter.model, dataset=dataset, **kwargs)

    def export_model(self, **kwargs):
        return self.task_controller.run_task("exporter", model=self.model_adapter.model, **kwargs)

    def update_model(options: dict):
        self.model_adapter.update_model(options)

    def run(self, **kwargs):
        self.task_controller.run_tasks(**kwargs)
