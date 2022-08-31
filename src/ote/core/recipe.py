import os
from collections import OrderedDict

from mmcv.utils import build_from_cfg

from ote.core.config import Config
from ote.core.dataset import IDataset
from ote.core.model import IModel, ModelStatus
from ote.core.task import TASK_REGISTRY
from ote.logger import get_logger
from ote.utils import import_and_get_class_from_path

logger = get_logger()

class Recipe:
    def __init__(self, recipe_yaml):
        self.recipe = Config.fromfile(recipe_yaml)
        logger.info(f"recipe = {self.recipe}")

        if not hasattr(self.recipe, "gpu_ids"):
            gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            logger.info(f"CUDA_VISIBLE_DEVICES = {gpu_ids}")
            if gpu_ids is not None:
                if isinstance(gpu_ids, str):
                    self.recipe.gpu_ids = range(len(gpu_ids.split(",")))
                else:
                    raise ValueError(f"not supported type for gpu_ids: {type(gpu_ids)}")
            else:
                self.recipe.gpu_ids = range(1)

        if not hasattr(self.recipe, "work_dir"):
            self.recipe.work_dir = "./workspace"

        # initialize adapters
        self.adapters = {}
        adapters = self.recipe.pop("adapters")
        for adaptee, adapter in adapters.items():
            logger.info(f"initializing adapter {adapter} for {adaptee}")
            adapter_class = import_and_get_class_from_path(adapter)
            # TODO should we keep model and data config in the recipe?
            adapter_cfg = self.recipe.get(adaptee)
            logger.info(f"instantating {adapter} with config {adapter_cfg}")
            self.adapters[adaptee] = adapter_class(adapter_cfg)

        # initialize tasks
        task_config = self.recipe.pop("task_cfg")
        self.tasks = OrderedDict()
        for config in task_config.tasks:
            cfg = config.copy()
            spec = cfg.get("spec")
            if spec in self.tasks:
                logger.warning(f"the same {spec} spec cannot be configured again for a recipe. ignored it.")
                continue
            self.tasks[spec] = Recipe.build_task(cfg)

        # model update
        model_override = Config.fromfile(self.recipe.model_override)
        self.update_model(model_override.model)
        self.model: IModel = None
        self.model_status = ModelStatus.CONFIGURED

        # data update
        data_override = Config.fromfile(self.recipe.data_override)
        self.update_data(data_override.data)
        self.dataset: IDataset = None

    def train(self, **kwargs):
        self.model = self.get_model()
        train_dataset = kwargs.get("train_dataset", self.adapters["data"].get_subset("train"))
        val_dataset = kwargs.get("val_dataset", self.adapters["data"].get_subset("val"))
        datasets = dict(
            train=[train_dataset],
            val=[val_dataset],
        )
        return self.run_task("trainer", self.model, datasets, **kwargs)

    def eval(self, metric, **kwargs):
        self.model = self.get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_subset("test"))
        infer_results = self.infer(dataset, **kwargs)
        return self.run_task("evaluator", metric, infer_results, **kwargs)

    def infer(self, **kwargs):
        self.model = self.get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_test_dataset())
        return self.run_task("inferrer", model, dataset, **kwargs)

    def export_model(self, **kwargs):
        model = self.get_model()
        return self.run_task("exporter", model, **kwargs)

    def optimize(self, option, **kwargs):
        self.optimized_model = "optimized model"
        self.model_status = ModelStatus.OPTIMIZED
        return True

    def update_model(self, options: dict):
        self.adapters["model"].update_model(options)
        self.model_status = ModelStatus.CONFIG_UPDATED

    def update_data(self, options: dict):
        self.adapters["data"].update_data(options)

    def run(self, *args, **kwargs):
        self.run_tasks(*args, **kwargs)

    @staticmethod
    def build_task(config):
        return build_from_cfg(config, TASK_REGISTRY)

    def run_tasks(self, *args, **kwargs):
        results = OrderedDict()
        for _type, task in self.tasks.items():
            results[_type] = task.run(*args, **kwargs)
        return results

    def run_task(self, spec, *args, **kwargs):
        recipe = self.recipe.copy()
        results = None
        if spec in self.tasks.keys():
            results = self.tasks[spec].run(*args, recipe, **kwargs)
        else:
            logger.warning(f"cannot find a task type '{spec}'")
        return results

    def get_model(self):
        if self.model_status == ModelStatus.CONFIGURED or self.model_status == ModelStatus.CONFIG_UPDATED:
            adapter = self.adapters["model"]
            return adapter.build()
        if self.model_status == ModelStatus.OPTIMIZED:
            return self.optimized_model
        return self.model
