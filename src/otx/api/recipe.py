import os
from collections import OrderedDict

from mmcv.utils import build_from_cfg

from ote.core.config import Config
from ote.core.dataset import IDataset
from ote.core.model import IModel, ModelStatus
from ote.core.job import JOB_REGISTRY
from ote.logger import get_logger
from ote.utils import import_and_get_class_from_path

logger = get_logger()

class Recipe:
    def __init__(self, recipe_yaml):
        logger.info("*** init recipe ***")
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
        logger.info("=== init adapters ===")
        self.adapters = {}
        adapters = self.recipe.pop("adapters")
        for adaptee, adapter in adapters.items():
            logger.info(f"initializing adapter {adapter} for {adaptee}")
            adapter_class = import_and_get_class_from_path(adapter)
            # TODO should we keep model and data config in the recipe?
            adapter_cfg = self.recipe.get(adaptee)
            logger.info(f"instantating {adapter} with config {adapter_cfg}")
            self.adapters[adaptee] = adapter_class(adapter_cfg)

        # initialize jobs
        logger.info("=== init jobs ===")
        job_config = self.recipe.pop("job_cfg")
        self.jobs = OrderedDict()
        for config in job_config.jobs:
            cfg = config.copy()
            spec = cfg.get("spec")
            if spec in self.jobs:
                logger.warning(f"the same {spec} spec cannot be configured again for a recipe. ignored it.")
                continue
            self.jobs[spec] = Recipe._build_job(cfg)

        # model update
        logger.info("=== model cfg override ===")
        model_override = Config.fromfile(self.recipe.model_override)
        self.update_model_cfg(model_override.model)
        self.model: IModel = None
        self.model_status = ModelStatus.CONFIGURED

        # data update
        logger.info("=== data cfg override ===")
        data_override = Config.fromfile(self.recipe.data_override)
        self.update_data_cfg(data_override.data)
        self.dataset: IDataset = None

    def train(self, **kwargs):
        logger.info("*** recipe.train() ***")
        spec = kwargs.get("spec", "trainer")

        logger.info("=== configure task ===")
        self.jobs[spec].configure(self.recipe,
            model_cfg=self.adapters["model"].config,
            data_cfg=self.adapters["data"].config,
        )

        logger.info("=== update config ===")
        self.update_model_cfg(self.recipe.model, overwrite=True)
        self.update_data_cfg(self.recipe.data, overwrite=True)

        logger.info("=== prepare model ===")
        self.model = self._get_model()
        logger.info("=== prepare dataset ===")
        train_dataset = kwargs.pop("train_dataset", self.adapters["data"].get_subset("train"))
        val_dataset = kwargs.pop("val_dataset", self.adapters["data"].get_subset("val"))
        datasets = dict(
            train=train_dataset,
            val=val_dataset,
        )
        results = self._run_job(spec, self.model, datasets, **kwargs)
        ckpt = results.get("final_ckpt")
        if ckpt is not None:
            self.adapters["model"].ckpt = ckpt
            self.model_status = ModelStatus.TRAINED

    def eval(self, metric, **kwargs):
        logger.info("*** recipe.eval() ***")
        spec = kwargs.get("spec", "evaluator")

        logger.info("=== configure task ===")
        self.jobs[spec].configure(self.recipe,
            model_cfg=self.adapters["model"].config,
            data_cfg=self.adapters["data"].config,
            training=False,
            model_ckpt=self.adapters["model"].ckpt,
        )

        logger.info("=== update config ===")
        self.update_model_cfg(self.recipe.model, overwrite=True)
        self.update_data_cfg(self.recipe.data, overwrite=True)

        logger.info("=== prepare model ===")
        self.model = self._get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_subset("test"))
        infer_results = self.infer(dataset, **kwargs)
        return self._run_job("evaluator", metric, infer_results, **kwargs)

    def infer(self, **kwargs):
        self.model = self._get_model()
        dataset = kwargs.get("dataset", self.adapters["data"].get_test_dataset())
        return self._run_job("inferrer", model, dataset, **kwargs)

    def export_model(self, **kwargs):
        model = self._get_model()
        return self._run_job("exporter", model, **kwargs)

    def optimize(self, option, **kwargs):
        self.optimized_model = "optimized model"
        self.model_status = ModelStatus.OPTIMIZED
        return True

    def update_model_cfg(self, options: dict, **kwargs):
        self.adapters["model"].update_config(options, **kwargs)
        self.model_status = ModelStatus.CONFIG_UPDATED

    def update_data_cfg(self, options: dict, **kwargs):
        self.adapters["data"].update_config(options, **kwargs)

    def _run(self, *args, **kwargs):
        self._run_jobs(*args, **kwargs)

    @staticmethod
    def _build_job(config):
        return build_from_cfg(config, JOB_REGISTRY)

    def _run_jobs(self, *args, **kwargs):
        results = OrderedDict()
        for _type, task in self.jobs.items():
            results[_type] = task.run(*args, **kwargs)
        return results

    def _run_job(self, spec, *args, **kwargs):
        recipe = self.recipe.copy()
        results = None
        if spec in self.jobs.keys():
            results = self.jobs[spec].run(*args, recipe, **kwargs)
        else:
            logger.warning(f"cannot find a task type '{spec}'")
        return results

    def _get_model(self):
        if self.model_status == ModelStatus.CONFIGURED or self.model_status == ModelStatus.CONFIG_UPDATED:
            adapter = self.adapters["model"]
            return adapter.build()
        if self.model_status == ModelStatus.OPTIMIZED:
            return self.optimized_model
        return self.model
