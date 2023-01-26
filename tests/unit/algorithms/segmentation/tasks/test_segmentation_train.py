# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from otx.algorithms.segmentation.tasks import SegmentationTrainTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.task_environment import TaskEnvironment
from tests.test_suite.e2e_test_system import e2e_pytest_unit

DEFAULT_SEG_TEMPLATE_DIR = os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2")


class TestOTXSegTaskTrain:
    def init_environment(self, params, model_template):
        labels_schema = LabelSchemaEntity()
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )
        return environment

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = self.init_environment(hyper_parameters, model_template)
        self.seg_train_task = SegmentationTrainTask(task_env)
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )

        self.model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        mocker.patch("torch.load", "")
        mocker.patch("torch.save")
        self.seg_train_task.save_model(self.model)

    @e2e_pytest_unit
    def test_cancel_training(self):
        self.seg_train_task.cancel_training()

    @e2e_pytest_unit
    def test_train(self):
        self.seg_train_task.train()
