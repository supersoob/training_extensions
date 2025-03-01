"""Base configurer for mmdet config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import json
import os
from typing import Any, Dict

import numpy as np
import torch
from mmcv import build_from_cfg
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict
from torch import distributed as dist

from otx.algorithms import TRANSFORMER_BACKBONES
from otx.algorithms.classification.adapters.mmcls.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    align_data_config_with_recipe,
    build_dataloader,
    build_dataset,
    patch_adaptive_interval_training,
    patch_default_config,
    patch_early_stopping,
    patch_fp16,
    patch_persistent_workers,
    patch_runner,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-public-methods
class ClassificationConfigurer:
    """Patch config to support otx train."""

    def __init__(self):
        self.task_adapt_type = None
        self.task_adapt_op = "REPLACE"
        self.org_model_classes = []
        self.model_classes = []
        self.data_classes = []

    # pylint: disable=too-many-arguments
    def configure(
        self,
        cfg,
        model_ckpt,
        data_cfg,
        training=True,
        subset="train",
        ir_options=None,
        data_classes=None,
        model_classes=None,
        **kwargs,
    ):
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={training}")

        self.configure_base(cfg, data_cfg, data_classes, model_classes, **kwargs)
        self.configure_device(cfg, training)
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_model(cfg, ir_options)
        self.configure_data(cfg, training, data_cfg)
        self.configure_task(cfg, training)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg, subset)
        self.configure_fp16_optimizer(cfg)
        self.configure_compat_cfg(cfg)
        return cfg

    def configure_base(self, cfg, data_cfg, data_classes, model_classes, **kwargs):
        """Basic configuration work for recipe.

        Patchings in this function are handled task level previously
        This function might need to be re-orgianized
        """

        options_for_patch_datasets = kwargs.get("options_for_patch_datasets", {"type": "OTXClsDataset"})
        options_for_patch_evaluation = kwargs.get("options_for_patch_evaluation", {"task": "normal"})

        patch_default_config(cfg)
        patch_runner(cfg)
        patch_datasets(
            cfg,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(cfg, **options_for_patch_evaluation)  # for OTX compatibility
        patch_fp16(cfg)
        patch_adaptive_interval_training(cfg)
        patch_early_stopping(cfg)
        patch_persistent_workers(cfg)

        if data_cfg is not None:
            align_data_config_with_recipe(data_cfg, cfg)

        # update model config -> model label schema
        cfg["model_classes"] = model_classes
        if data_classes is not None:
            train_data_cfg = self.get_data_cfg(data_cfg, "train")
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

    def configure_model(self, cfg, ir_options):  # noqa: C901
        """Patch config's model.

        Change model type to super type
        Patch for OMZ backbones
        """

        if ir_options is None:
            ir_options = {"ir_model_path": None, "ir_weight_path": None, "ir_weight_init": False}

        cfg.model_task = cfg.model.pop("task", "classification")
        if cfg.model_task != "classification":
            raise ValueError(f"Given cfg ({cfg.filename}) is not supported by classification recipe")

        super_type = cfg.model.pop("super_type", None)
        if super_type:
            cfg.model.arch_type = cfg.model.type
            cfg.model.type = super_type

        # OV-plugin
        ir_model_path = ir_options.get("ir_model_path")
        if ir_model_path:

            def is_mmov_model(key, value):
                if key == "type" and value.startswith("MMOV"):
                    return True
                return False

            ir_weight_path = ir_options.get("ir_weight_path", None)
            ir_weight_init = ir_options.get("ir_weight_init", False)
            recursively_update_cfg(
                cfg,
                is_mmov_model,
                {"model_path": ir_model_path, "weight_path": ir_weight_path, "init_weight": ir_weight_init},
            )

        self.configure_in_channel(cfg)
        self.configure_topk(cfg)

    def configure_data(self, cfg, training, data_cfg):  # noqa: C901
        """Patch cfg.data.

        Merge cfg and data_cfg
        Match cfg.data.train.type to super_type
        Patch for unlabeled data path ==> This may be moved to SemiClassificationConfigurer
        """
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        def configure_split(target):
            def update_transform(opt, pipeline, idx, transform):
                if isinstance(opt, dict):
                    if "_delete_" in opt.keys() and opt.get("_delete_", False):
                        # if option include _delete_=True, remove this transform from pipeline
                        logger.info(f"configure_data: {transform['type']} is deleted")
                        del pipeline[idx]
                        return
                    logger.info(f"configure_data: {transform['type']} is updated with {opt}")
                    transform.update(**opt)

            # pylint: disable=too-many-branches, too-many-nested-blocks
            def update_config(src, pipeline_options):
                logger.info(f"update_config() {pipeline_options}")
                if src.get("pipeline") is not None or (
                    src.get("dataset") is not None and src.get("dataset").get("pipeline") is not None
                ):
                    if src.get("pipeline") is not None:
                        pipeline = src.get("pipeline", None)
                    else:
                        pipeline = src.get("dataset").get("pipeline")
                    if isinstance(pipeline, list):
                        for idx, transform in enumerate(pipeline):
                            for opt_key, opt in pipeline_options.items():
                                if transform["type"] == opt_key:
                                    update_transform(opt, pipeline, idx, transform)
                    elif isinstance(pipeline, dict):
                        for _, pipe in pipeline.items():
                            for idx, transform in enumerate(pipe):
                                for opt_key, opt in pipeline_options.items():
                                    if transform["type"] == opt_key:
                                        update_transform(opt, pipe, idx, transform)
                    else:
                        raise NotImplementedError(f"pipeline type of {type(pipeline)} is not supported")
                else:
                    logger.info("no pipeline in the data split")

            split = cfg.data.get(target)
            if split is not None:
                if isinstance(split, list):
                    for sub_item in split:
                        update_config(sub_item, pipeline_options)
                elif isinstance(split, dict):
                    update_config(split, pipeline_options)
                else:
                    logger.warning(f"type of split '{target}'' should be list or dict but {type(split)}")

        logger.info("configure_data()")
        pipeline_options = cfg.data.pop("pipeline_options", None)
        if pipeline_options is not None and isinstance(pipeline_options, dict):
            configure_split("train")
            configure_split("val")
            if not training:
                configure_split("test")
            configure_split("unlabeled")
        super_type = cfg.data.train.pop("super_type", None)
        if super_type:
            cfg.data.train.org_type = cfg.data.train.type
            cfg.data.train.type = super_type

    def configure_task(self, cfg, training):
        """Patch config to support training algorithm."""
        if "task_adapt" in cfg:
            logger.info(f"task config!!!!: training={training}")
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

            src_data_cfg = self.get_data_cfg(cfg, "train")
            src_data_cfg.pop("old_new_indices", None)

    # pylint: disable=too-many-branches
    def configure_classes(self, cfg):
        """Patch classes for model and dataset."""
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if self.task_adapt_op == "REPLACE":
            if len(data_classes) == 0:
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif self.task_adapt_op == "MERGE":
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f"{self.task_adapt_op} is not supported for task_adapt options!")

        # Model architecture
        cfg.model.head.num_classes = len(model_classes)

        self.org_model_classes = org_model_classes
        self.model_classes = model_classes

    # Functions below are come from base stage
    def configure_ckpt(self, cfg, model_ckpt):
        """Patch checkpoint path for pretrained weight.

        Replace cfg.load_from to model_ckpt
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        if cfg.get("resume", False):
            cfg.resume_from = cfg.load_from
        if cfg.get("load_from", None) and cfg.model.backbone.get("pretrained", None):
            cfg.model.backbone.pretrained = None

    @staticmethod
    def get_model_ckpt(ckpt_path, new_path=None):
        """Get pytorch model weights."""
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
            if not new_path:
                new_path = ckpt_path[:-3] + "converted.pth"
            torch.save(ckpt, new_path)
            return new_path
        return ckpt_path

    @staticmethod
    def get_model_classes(cfg):
        """Extract trained classes info from checkpoint file.

        MMCV-based models would save class info in ckpt['meta']['CLASSES']
        For other cases, try to get the info from cfg.model.classes (with pop())
        - Which means that model classes should be specified in model-cfg for
          non-MMCV models (e.g. OMZ models)
        """

        def get_model_meta(cfg):
            ckpt_path = cfg.get("load_from", None)
            meta = {}
            if ckpt_path:
                ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
                meta = ckpt.get("meta", {})
            return meta

        def read_label_schema(ckpt_path, name_only=True, file_name="label_schema.json"):
            serialized_label_schema = []
            if any(ckpt_path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
                label_schema_path = os.path.join(os.path.dirname(ckpt_path), file_name)
                if os.path.exists(label_schema_path):
                    with open(label_schema_path, encoding="UTF-8") as read_file:
                        serialized_label_schema = json.load(read_file)
            if serialized_label_schema:
                if name_only:
                    all_classes = [labels["name"] for labels in serialized_label_schema["all_labels"].values()]
                else:
                    all_classes = serialized_label_schema
            else:
                all_classes = []
            return all_classes

        classes = []
        meta = get_model_meta(cfg)
        # for MPA classification legacy compatibility
        classes = meta.get("CLASSES", [])
        classes = meta.get("classes", classes)
        if classes is None:
            classes = []

        if len(classes) == 0:
            ckpt_path = cfg.get("load_from", None)
            if ckpt_path:
                classes = read_label_schema(ckpt_path)
        if len(classes) == 0:
            classes = cfg.model.pop("classes", cfg.pop("model_classes", []))
        return classes

    def get_data_classes(self, cfg):
        """Get data classes from train cfg."""
        data_classes = []
        train_cfg = self.get_data_cfg(cfg, "train")
        if "data_classes" in train_cfg:
            data_classes = list(train_cfg.pop("data_classes", []))
        elif "classes" in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_data_cfg(cfg, subset):
        """Get subset's data cfg."""
        assert subset in ["train", "val", "test"], f"Unknown subset:{subset}"
        if "dataset" in cfg.data[subset]:  # Concat|RepeatDataset
            dataset = cfg.data[subset].dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset
        return cfg.data[subset]

    @staticmethod
    def configure_hook(cfg):
        """Update cfg.custom_hooks based on cfg.custom_hook_options."""

        def update_hook(opt, custom_hooks, idx, hook):
            """Delete of update a custom hook."""
            if isinstance(opt, dict):
                if opt.get("_delete_", False):
                    # if option include _delete_=True, remove this hook from custom_hooks
                    logger.info(f"configure_hook: {hook['type']} is deleted")
                    del custom_hooks[idx]
                else:
                    logger.info(f"configure_hook: {hook['type']} is updated with {opt}")
                    hook.update(**opt)

        hook_cfg = ConfigDict(type="LoggerReplaceHook")
        update_or_add_custom_hook(cfg, hook_cfg)

        custom_hook_options = cfg.pop("custom_hook_options", {})
        custom_hooks = cfg.get("custom_hooks", [])
        for idx, hook in enumerate(custom_hooks):
            for opt_key, opt in custom_hook_options.items():
                if hook["type"] == opt_key:
                    update_hook(opt, custom_hooks, idx, hook)

    def configure_device(self, cfg, training):
        """Setting device for training and inference."""
        cfg.distributed = False
        if torch.distributed.is_initialized():
            cfg.gpu_ids = [int(os.environ["LOCAL_RANK"])]
            if training:  # TODO multi GPU is available only in training. Evaluation needs to be supported later.
                cfg.distributed = True
                self.configure_distributed(cfg)
        elif "gpu_ids" not in cfg:
            gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
            logger.info(f"CUDA_VISIBLE_DEVICES = {gpu_ids}")
            if gpu_ids is not None:
                cfg.gpu_ids = range(len(gpu_ids.split(",")))
            else:
                cfg.gpu_ids = range(1)

        # consider "cuda" and "cpu" device only
        if not torch.cuda.is_available():
            cfg.device = "cpu"
            cfg.gpu_ids = range(-1, 0)
        else:
            cfg.device = "cuda"

    def configure_samples_per_gpu(
        self,
        cfg: Config,
        subset: str,
    ):
        """Settings samples_per_gpu for training and inference."""

        dataloader_cfg = cfg.data.get(f"{subset}_dataloader", ConfigDict())
        samples_per_gpu = dataloader_cfg.get("samples_per_gpu", cfg.data.get("samples_per_gpu", 1))

        data_cfg = self.get_data_cfg(cfg, subset)
        if data_cfg.get("otx_dataset") is not None:
            dataset_len = len(data_cfg.otx_dataset)

            if getattr(cfg, "distributed", False):
                dataset_len = dataset_len // dist.get_world_size()

            # set batch size as a total dataset
            # if it is smaller than total dataset
            if dataset_len < samples_per_gpu:
                dataloader_cfg.samples_per_gpu = dataset_len

            # drop the last batch if the last batch size is 1
            # batch size of 1 is a runtime error for training batch normalization layer
            if subset in ("train", "unlabeled") and dataset_len % samples_per_gpu == 1:
                dataloader_cfg.drop_last = True

            cfg.data[f"{subset}_dataloader"] = dataloader_cfg

    @staticmethod
    def configure_fp16_optimizer(cfg: Config):
        """Configure Fp16OptimizerHook and Fp16SAMOptimizerHook."""

        fp16_config = cfg.pop("fp16", None)
        if fp16_config is not None:
            optim_type = cfg.optimizer_config.get("type", "OptimizerHook")
            opts: Dict[str, Any] = dict(
                distributed=getattr(cfg, "distributed", False),
                **fp16_config,
            )
            if optim_type == "SAMOptimizerHook":
                opts["type"] = "Fp16SAMOptimizerHook"
            elif optim_type == "OptimizerHook":
                opts["type"] = "Fp16OptimizerHook"
            else:
                # does not support optimizerhook type
                # let mm library handle it
                cfg.fp16 = fp16_config
                opts = dict()
            cfg.optimizer_config.update(opts)

    @staticmethod
    def configure_distributed(cfg):
        """Patching for distributed training."""
        if hasattr(cfg, "dist_params") and cfg.dist_params.get("linear_scale_lr", False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr

    # pylint: disable=too-many-branches
    @staticmethod
    def configure_in_channel(cfg):
        """Return whether in_channels need patch."""
        configure_required = False
        if cfg.model.get("neck") is not None:
            if cfg.model.neck.get("in_channels") is not None and cfg.model.neck.in_channels <= 0:
                configure_required = True
        if not configure_required and cfg.model.get("head") is not None:
            if cfg.model.head.get("in_channels") is not None and cfg.model.head.in_channels <= 0:
                configure_required = True
        if not configure_required:
            return

        # update model layer's in/out configuration
        from mmcv.cnn import MODELS as backbone_reg

        layer = build_from_cfg(cfg.model.backbone, backbone_reg)
        layer.eval()
        input_shape = [3, 224, 224]
        # MMOV model
        if hasattr(layer, "input_shapes"):
            input_shape = next(iter(getattr(layer, "input_shapes").values()))
            input_shape = input_shape[1:]
            if any(i < 0 for i in input_shape):
                input_shape = [3, 244, 244]
        logger.debug(f"input shape for backbone {input_shape}")
        output = layer(torch.rand([1] + list(input_shape)))
        if isinstance(output, (tuple, list)):
            output = output[-1]

        if layer.__class__.__name__ in TRANSFORMER_BACKBONES and isinstance(output, (tuple, list)):
            # mmcls.VisionTransformer outputs Tuple[List[...]] and the last index of List is the final logit.
            _, output = output
            if cfg.model.head.type != "VisionTransformerClsHead":
                raise ValueError(f"{layer.__class__.__name__ } needs VisionTransformerClsHead as head")

        in_channels = output.shape[1]
        if cfg.model.get("neck") is not None:
            if cfg.model.neck.get("in_channels") is not None:
                logger.info(
                    f"'in_channels' config in model.neck is updated from "
                    f"{cfg.model.neck.in_channels} to {in_channels}"
                )
                cfg.model.neck.in_channels = in_channels
                logger.debug(f"input shape for neck {input_shape}")
                from mmcls.models.builder import NECKS as neck_reg

                layer = build_from_cfg(cfg.model.neck, neck_reg)
                layer.eval()
                output = layer(torch.rand(output.shape))
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                in_channels = output.shape[1]
        if cfg.model.get("head") is not None:
            if cfg.model.head.get("in_channels") is not None:
                logger.info(
                    f"'in_channels' config in model.head is updated from "
                    f"{cfg.model.head.in_channels} to {in_channels}"
                )
                cfg.model.head.in_channels = in_channels

    @staticmethod
    def configure_topk(cfg):
        """Patch topk in case of num_classes is less than 5."""
        if cfg.model.head.get("topk", False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get("multilabel", False) or cfg.model.get("hierarchical", False):
                cfg.model.head.pop("topk", None)

    @staticmethod
    def configure_compat_cfg(
        cfg: Config,
    ):
        """Modify config to keep the compatibility."""

        def _configure_dataloader(cfg):
            global_dataloader_cfg = {}
            global_dataloader_cfg.update(
                {
                    k: cfg.data.pop(k)
                    for k in list(cfg.data.keys())
                    if k
                    not in [
                        "train",
                        "val",
                        "test",
                        "unlabeled",
                        "train_dataloader",
                        "val_dataloader",
                        "test_dataloader",
                        "unlabeled_dataloader",
                    ]
                }
            )

            for subset in ["train", "val", "test", "unlabeled"]:
                if subset not in cfg.data:
                    continue
                dataloader_cfg = cfg.data.get(f"{subset}_dataloader", None)
                if dataloader_cfg is None:
                    raise AttributeError(f"{subset}_dataloader is not found in config.")
                dataloader_cfg = {**global_dataloader_cfg, **dataloader_cfg}
                cfg.data[f"{subset}_dataloader"] = dataloader_cfg

        _configure_dataloader(cfg)


CLASS_INC_DATASET = [
    "OTXClsDataset",
    "OTXMultilabelClsDataset",
    "MPAHierarchicalClsDataset",
    "ClsTVDataset",
]
WEIGHT_MIX_CLASSIFIER = ["SAMImageClassifier"]


class IncrClassificationConfigurer(ClassificationConfigurer):
    """Patch config to support incremental learning for classification."""

    def configure_task(self, cfg, training):
        """Patch config to support incremental learning."""
        super().configure_task(cfg, training)
        if "task_adapt" in cfg and self.task_adapt_type == "mpa":
            self.configure_task_adapt_hook(cfg, training)

    def configure_task_adapt_hook(self, cfg, training):
        """Add TaskAdaptHook for sampler."""
        train_data_cfg = self.get_data_cfg(cfg, "train")
        if training:
            if train_data_cfg.type not in CLASS_INC_DATASET:
                logger.warning(f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!")
            if "new_classes" not in train_data_cfg:
                logger.warning('"new_classes" should be defined for incremental learning w/ current model.')

            if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                cfg.model.task_adapt = ConfigDict(
                    src_classes=self.org_model_classes,
                    dst_classes=self.model_classes,
                )
            else:
                logger.warning(f"Weight mixing for {cfg.model.type} is not yet supported!")

            train_data_cfg.classes = self.model_classes

        if not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False):
            efficient_mode = cfg["task_adapt"].get("efficient_mode", True)
            sampler_type = "balanced"
            self.configure_loss(cfg)
        else:
            efficient_mode = cfg["task_adapt"].get("efficient_mode", False)
            sampler_type = "cls_incr"

        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            sampler_flag = False
        else:
            sampler_flag = True
        # Update Task Adapt Hook
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.org_model_classes,
            dst_classes=self.model_classes,
            model_type=cfg.model.type,
            sampler_flag=sampler_flag,
            sampler_type=sampler_type,
            efficient_mode=efficient_mode,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_loss(self, cfg):
        """Patch classification loss."""
        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            cfg.model.head.loss = dict(type="CrossEntropyLoss", loss_weight=1.0)
        else:
            cfg.model.head.loss = ConfigDict(
                type="IBLoss",
                num_classes=cfg.model.head.num_classes,
            )
            ib_loss_hook = ConfigDict(
                type="IBLossHook",
                dst_classes=self.model_classes,
            )
            update_or_add_custom_hook(cfg, ib_loss_hook)


class SemiSLClassificationConfigurer(ClassificationConfigurer):
    """Patch config to support semi supervised learning for classification."""

    def configure_data(self, cfg, training, data_cfg):
        """Patch cfg.data."""
        super().configure_data(cfg, training, data_cfg)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg)

    @staticmethod
    def configure_unlabeled_dataloader(cfg: Config):
        """Patch for unlabled dataloader."""

        model_task = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}
        if "unlabeled" in cfg.data:
            task_lib_module = importlib.import_module(f"{model_task[cfg.model_task]}.datasets")
            dataset_builder = getattr(task_lib_module, "build_dataset")
            dataloader_builder = getattr(task_lib_module, "build_dataloader")

            dataset = build_dataset(cfg, "unlabeled", dataset_builder, consume=True)
            unlabeled_dataloader = build_dataloader(
                dataset,
                cfg,
                "unlabeled",
                dataloader_builder,
                distributed=cfg.distributed,
                consume=True,
            )

            custom_hooks = cfg.get("custom_hooks", [])
            updated = False
            for custom_hook in custom_hooks:
                if custom_hook["type"] == "ComposedDataLoadersHook":
                    custom_hook["data_loaders"] = [*custom_hook["data_loaders"], unlabeled_dataloader]
                    updated = True
            if not updated:
                custom_hooks.append(
                    ConfigDict(
                        type="ComposedDataLoadersHook",
                        data_loaders=unlabeled_dataloader,
                    )
                )
            cfg.custom_hooks = custom_hooks
