# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import glob
import os
import warnings

import numpy as np
import pytest

from otx.algorithms.common.tasks import BaseTask
from otx.api.utils.shape_factory import ShapeFactory

from otx.algorithms.segmentation.tasks import SegmentationInferenceTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.scored_label import ScoredLabel

from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.test_helpers import generate_random_annotated_image

DEFAULT_SEG_TEMPLATE_DIR = os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2")


class TestOTXSegTaskTrain:
    @staticmethod
    def generate_label_schema(label_names):
        label_domain = Domain.SEGMENTATION
        rgb = [int(i) for i in np.random.randint(0, 256, 3)]
        colors = [Color(*rgb) for _ in range(len(label_names))]
        not_empty_labels = [
            LabelEntity(name=name, color=colors[i], domain=label_domain, id=i) for i, name in enumerate(label_names)
        ]
        empty_label = LabelEntity(
            name="Empty label",
            color=Color(42, 43, 46),
            is_empty=True,
            domain=label_domain,
            id=len(not_empty_labels),
        )

        label_schema = LabelSchemaEntity()
        exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        label_schema.add_group(exclusive_group)
        label_schema.add_group(empty_group)
        return label_schema

    def init_environment(self, params, model_template, number_of_images=10):
        labels_names = ("rectangle", "ellipse", "triangle")
        labels_schema = self.generate_label_schema(labels_names)
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )

        return environment

    def generate_dataset(self, number_of_images=10):
        items = []
        labels_names = ("rectangle", "ellipse", "triangle")
        labels_schema = self.generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(
                image_width=640,
                image_height=480,
                labels=labels_list,
                max_shapes=20,
                min_size=50,
                max_size=100,
                random_seed=None,
            )
            # Convert all shapes to polygons
            out_shapes = []
            for shape in shapes:
                shape_labels = shape.get_labels(include_empty=True)

                in_shape = shape.shape
                if isinstance(in_shape, Rectangle):
                    points = [
                        Point(in_shape.x1, in_shape.y1),
                        Point(in_shape.x2, in_shape.y1),
                        Point(in_shape.x2, in_shape.y2),
                        Point(in_shape.x1, in_shape.y2),
                    ]
                elif isinstance(in_shape, Ellipse):
                    points = [Point(x, y) for x, y in in_shape.get_evenly_distributed_ellipse_coordinates()]
                elif isinstance(in_shape, Polygon):
                    points = in_shape.points

                out_shapes.append(Annotation(Polygon(points=points), labels=shape_labels))

            image = Image(data=image_numpy)
            annotation = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=out_shapes)
            items.append(DatasetItemEntity(media=image, annotation_scene=annotation))
        warnings.resetwarnings()

        dataset = DatasetEntity(items)

        return dataset

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = self.init_environment(hyper_parameters, model_template)
        self.seg_train_task = SegmentationInferenceTask(task_env)
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        self.model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        self.output_path = self.seg_train_task._output_path

    @e2e_pytest_unit
    def test_infer(self, mocker):
        dataset = self.generate_dataset(5)

        fake_output = {"outputs" : {"eval_predictions": np.zeros((5,1)), "feature_vectors" : np.zeros((5,1))}}
        fake_annotation = [Annotation(Polygon(points=[Point(0,0)]), id=0, labels=[ScoredLabel(LabelEntity(name="fake", domain="SEGMENTATION"), probability=1.0)])]
        
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        mocker.patch("numpy.transpose")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.create_hard_prediction_from_soft_prediction")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.create_annotation_from_segmentation_map", return_value=fake_annotation)
        mocker.patch("otx.algorithms.segmentation.tasks.inference.get_activation_map", return_value=np.zeros((1,1)))
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)

        updated_dataset = self.seg_train_task.infer(dataset, None)

        mock_run_task.assert_called_once()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name="fake", domain="SEGMENTATION")])

    @e2e_pytest_unit
    def test_evaluate(self):
        result_set = ResultSetEntity(
            model=self.model,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        with pytest.raises(ValueError):
            self.seg_train_task.evaluate(result_set)

    @e2e_pytest_unit
    def test_export(self, mocker):
        fake_output = {"outputs" : {"bin": None, "xml" : None}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        
        with pytest.raises(RuntimeError):
            self.seg_train_task.export(ExportType.OPENVINO, self.model)
            mock_run_task.assert_called_once()
    
    @e2e_pytest_unit
    def test_export_with_xml_file(self, mocker):
        with open(f"{self.output_path}/model.xml", "wb") as f:
            f.write(b"foo")
        with open(f"{self.output_path}/model.bin", "wb") as f:
            f.write(b"bar")

        fake_output = {"outputs" : {"bin": f"{self.output_path}/model.xml", "xml" : f"{self.output_path}/model.bin"}}

        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        self.seg_train_task.export(ExportType.OPENVINO, self.model)

        mock_run_task.assert_called_once()
        assert self.model.get_data("openvino.bin")
        assert self.model.get_data("openvino.xml")

    @e2e_pytest_unit
    def test_unload(self, mocker):
        mock_cleanup = mocker.patch.object(BaseTask, "cleanup")
        self.seg_train_task.unload()
        
        mock_cleanup.assert_called_once()

