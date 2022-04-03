"""Tests for input parameters with OTE CLI demo tool"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
)

from ote_cli_test_common import (
    default_train_args_paths,
    wrong_paths,
    ote_common,
    logger,
    parser_templates,
    root,
    ote_dir,
    get_pretrained_artifacts,
)

params_values, params_ids, params_values_for_be, params_ids_for_be = parser_templates()


class TestDemoCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_template(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: the following arguments are required: template"
        command_args = []
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_weights(self, back_end, template, create_venv_fx):
        error_string = (
            "ote demo: error: the following arguments are required: --load-weights"
        )
        command_args = [
            template.model_template_id,
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_input(self, back_end, template, create_venv_fx):
        error_string = (
            "ote demo: error: the following arguments are required: -i/--input"
        )
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_wrong_weights(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_args = [
                template.model_template_id,
                "--load-weights",
                case,
                "--input",
                f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            ]
            ret = ote_common(template, root, "demo", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_wrong_input(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: argument -i/--input: expected one argument"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--input",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_no_value(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: argument --fit-to-size: expected 2 arguments"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--fit-to-size",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_float_input(self, back_end, template, create_venv_fx):
        error_string = "--fit-to-size: invalid int value:"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--fit-to-size",
            "0.0",
            "0.0",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_negative_input(self, back_end, template, create_venv_fx):
        error_string = "Both values of --fit_to_size parameter must be > 0"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--fit-to-size",
            "1",
            "-1",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_delay_wrong_type(self, back_end, template, create_venv_fx):
        error_string = "invalid int value"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--delay",
            "String",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"


class TestDemoDetectionTemplateArguments:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def get_pretrained_artifacts_fx(self, template, create_venv_fx):
        get_pretrained_artifacts(template, root, ote_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_demo_pp_confidence_threshold_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid float value"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--delay",
            "-1",
            "params",
            "--postprocessing.confidence_threshold",
            "String",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_demo_pp_confidence_threshold_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds"
        oob_values = ["-0.1", "1.1"]
        for value in oob_values:
            command_args = [
                template.model_template_id,
                "--load-weights",
                f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
                "--input",
                f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
                "--delay",
                "-1",
                "params",
                "--postprocessing.confidence_threshold",
                value,
            ]
            ret = ote_common(template, root, "demo", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_demo_pp_result_based_confidence_threshold_type(
        self, template, create_venv_fx, get_pretrained_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "Boolean value expected"
        command_args = [
            template.model_template_id,
            "--load-weights",
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth",
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "params",
            "--postprocessing.result_based_confidence_threshold",
            "NonBoolean",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_demo_pp_result_based_confidence_threshold(
        self, template, create_venv_fx, get_pretrained_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        pre_trained_weights = (
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        )
        logger.debug(f"Pre-trained weights path: {pre_trained_weights}")
        assert os.path.exists(
            pre_trained_weights
        ), f"Pre trained weights must be before test starts"
        command_args = [
            template.model_template_id,
            "--load-weights",
            pre_trained_weights,
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--delay",
            "-1",
            "params",
            "--postprocessing.result_based_confidence_threshold",
            "False",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_demo_pp_confidence_threshold_type(
        self, template, create_venv_fx, get_pretrained_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        pre_trained_weights = (
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        )
        logger.debug(f"Pre-trained weights path: {pre_trained_weights}")
        assert os.path.exists(
            pre_trained_weights
        ), f"Pre trained weights must be before test starts"
        command_args = [
            template.model_template_id,
            "--load-weights",
            pre_trained_weights,
            "--input",
            f'{os.path.join(ote_dir, default_train_args_paths["--input"])}',
            "--delay",
            "-1",
            "params",
            "--postprocessing.confidence_threshold",
            "0.5",
        ]
        ret = ote_common(template, root, "demo", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"
