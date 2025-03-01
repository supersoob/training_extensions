"""Tests for Action Classification with OTX CLI."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json
import os
from pathlib import Path
from timeit import default_timer as timer

import pytest

from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    REGRESSION_TEST_EPOCHS,
    TIME_LOG,
    get_result_dict,
    get_template_performance,
    load_regression_configuration,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_compare,
    otx_eval_e2e_eval_time,
    otx_eval_e2e_train_time,
    otx_eval_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)

# Configurations for regression test.
TASK_TYPE = "action_classification"
TRAIN_TYPE = "supervised"
LABEL_TYPE = "multi_class"

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/action").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]


result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results/{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

action_cls_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, LABEL_TYPE)
action_cls_data_args = action_cls_regression_config["data_path"]
action_cls_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionActionClassification:
    def setup_method(self):
        self.label_type = LABEL_TYPE
        self.performance = {}

    def teardown(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / TASK_TYPE
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, action_cls_data_args)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        otx_eval_compare(
            template,
            tmp_dir_path,
            otx_dir,
            action_cls_data_args,
            action_cls_regression_config["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][LABEL_TYPE][TRAIN_TYPE]["train"].append(self.performance)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, template):
        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        otx_eval_e2e_train_time(
            train_time_criteria=action_cls_regression_config["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        otx_eval_e2e_eval_time(
            eval_time_criteria=action_cls_regression_config["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_eval_openvino(self, template, tmp_dir_path):
        if template.name == "MoViNet":
            pytest.skip(reason="[CVS-106939] MoViNet export --> eval issue.")
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / TASK_TYPE
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            action_cls_data_args,
            threshold=0.02,
            criteria=action_cls_regression_config["regression_criteria"]["export"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["export"].append(self.performance)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / TASK_TYPE
        pot_start_time = timer()
        pot_optimize_testing(template, tmp_dir_path, otx_dir, action_cls_data_args)
        pot_elapsed_time = timer() - pot_start_time

        pot_eval_start_time = timer()
        pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            action_cls_data_args,
            criteria=action_cls_regression_config["regression_criteria"]["pot"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        pot_eval_elapsed_time = timer() - pot_eval_start_time

        self.performance[template.name][TIME_LOG["pot_time"]] = round(pot_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["pot_eval_time"]] = round(pot_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["pot"].append(self.performance)
