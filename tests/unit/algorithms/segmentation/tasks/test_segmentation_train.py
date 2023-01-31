# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import random
import warnings

import numpy as np
import pytest

from otx.algorithms.segmentation.tasks import SegmentationTrainTask
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
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
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
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )

        items = []
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

        rng = random.Random()
        rng.shuffle(items)
        for i, _ in enumerate(items):
            subset_region = i / number_of_images
            if subset_region >= 0.8:
                subset = Subset.TESTING
            elif subset_region >= 0.6:
                subset = Subset.VALIDATION
            else:
                subset = Subset.TRAINING

            items[i].subset = subset

        dataset = DatasetEntity(items)

        return environment, dataset

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = 1
        hyper_parameters.learning_parameters.batch_size = 5
        task_env, self.dataset = self.init_environment(hyper_parameters, model_template, 5)
        self.seg_train_task = SegmentationTrainTask(task_env)
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        self.model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        self.model_ckpt_path = self.seg_train_task._output_path

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        mocker.patch("torch.load", return_value="")
        self.seg_train_task.save_model(self.model)

        assert self.model.get_data("weights.pth")
        assert self.model.get_data("label_schema.json")

    @e2e_pytest_unit
    def test_train(self, mocker):
        mocker.patch.object(SegmentationTrainTask, "save_model")
        self.seg_train_task.train(self.dataset, self.model)

        assert self.model.performance != NullPerformance()

    @e2e_pytest_unit
    def test_cancel_training(self):
        self.seg_train_task.cancel_training()

        assert self.seg_train_task._should_stop is True
