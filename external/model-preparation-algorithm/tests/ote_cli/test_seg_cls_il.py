"""Tests for MPA Class-Incremental Learning for semantic segmentation with OTE CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_sdk.entities.model_template import parse_model_template

from ote_cli.registry import Registry

from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_hpo_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
)


args = {
    '--train-ann-file': 'data/segmentation/custom/annotations/training',
    '--train-data-roots': 'data/segmentation/custom/images/training',
    '--val-ann-file': 'data/segmentation/custom/annotations/training',
    '--val-data-roots': 'data/segmentation/custom/images/training',
    '--test-ann-files': 'data/segmentation/custom/annotations/training',
    '--test-data-roots': 'data/segmentation/custom/images/training',
    '--input': 'data/segmentation/custom/images/training',
    'train_params': [
        'params',
        '--learning_parameters.learning_rate_fixed_iters',
        '0',
        '--learning_parameters.learning_rate_warmup_iters',
        '25',
        '--learning_parameters.num_iters',
        '2',
        '--learning_parameters.batch_size',
        '2'
    ]
}

root = '/tmp/ote_cli_seg/'
ote_dir = os.getcwd()

default_template = parse_model_template(
    os.path.join(
        'external/model-preparation-algorithm/configs', 'segmentation', 'ocr-lite-hrnet-18-mod2', 'template.yaml'
    )
)
templates = [default_template] * 100
templates_ids = [template.model_template_id + f'-{i+1}' for i, template in enumerate(templates)]


class TestToolsSegClsIncr:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)
        print(f'algo_backend_dir: {algo_backend_dir}')
        print(f'work_dir: {work_dir}')

    @e2e_pytest_component
    @pytest.mark.parametrize('template', templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args)
        _, template_work_dir, _ = get_some_vars(template, root)
        args1 = args.copy()
        args1['--load-weights'] = f'{template_work_dir}/trained_{template.model_template_id}/weights.pth'
        ote_train_testing(template, root, ote_dir, args1)
