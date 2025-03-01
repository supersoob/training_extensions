"""Base MMDataset for Detection Task."""

# Copyright (C) 2022 Intel Corporation
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

import tempfile
from collections import OrderedDict
from copy import copy
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from mmcv import Config
from mmcv.utils import print_log
from mmdet.core import PolygonMasks, eval_map
from mmdet.datasets.builder import DATASETS, build_dataset
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose

from otx.algorithms.common.utils.data import get_old_new_img_indices
from otx.algorithms.detection.adapters.mmdet.evaluation import eval_segm
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.shape_factory import ShapeFactory

from .tiling import Tile


# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes, super-init-not-called
@check_input_parameters_type()
def get_annotation_mmdet_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
    domain: Domain,
    min_size: int = -1,
) -> dict:
    """Function to convert a OTX annotation to mmdetection format.

    This is used both in the OTXDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels that are used in the task
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height

    # load annotations for item
    gt_bboxes = []
    gt_labels = []
    gt_polygons = []

    label_idx = {label.id: i for i, label in enumerate(labels)}

    for annotation in dataset_item.get_annotations(labels=labels, include_empty=False):

        box = ShapeFactory.shape_as_rectangle(annotation.shape)

        if min(box.width * width, box.height * height) < min_size:
            continue

        class_indices = [
            label_idx[label.id] for label in annotation.get_labels(include_empty=False) if label.domain == domain
        ]

        n = len(class_indices)
        gt_bboxes.extend([[box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height] for _ in range(n)])
        if domain != Domain.DETECTION:
            polygon = ShapeFactory.shape_as_polygon(annotation.shape)
            polygon = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
            gt_polygons.extend([[polygon] for _ in range(n)])
        gt_labels.extend(class_indices)

    if len(gt_bboxes) > 0:
        ann_info = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=int),
            masks=PolygonMasks(gt_polygons, height=height, width=width) if gt_polygons else [],
        )
    else:
        ann_info = dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.array([], dtype=int),
            masks=[],
        )
    return ann_info


@DATASETS.register_module()
class OTXDetDataset(CustomDataset):
    """Wrapper that allows using a OTX dataset to train mmdetection models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.

        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTXDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to otx_dataset and converts the dataset items to the view
        convenient for mmdetection.
        """

        def __init__(self, otx_dataset, labels):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.otx_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            height, width = item.height, item.width

            data_info = dict(
                dataset_item=item,
                width=width,
                height=height,
                index=index,
                ann_info=dict(label_list=self.labels),
                ignored_labels=ignored_labels,
            )

            return data_info

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        domain: Domain,
        test_mode: bool = False,
        **kwargs,
    ):
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        new_classes = dataset_cfg.pop("new_classes", [])
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.CLASSES = list(label.name for label in labels)
        self.domain = domain
        self.test_mode = test_mode

        # Instead of using list data_infos as in CustomDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to otx_dataset.
        # Note that list `data_infos` cannot be used here, since OTX dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_infos = OTXDetDataset._DataInfoProxy(otx_dataset, labels)

        self.proposals = None  # Attribute expected by mmdet but not used for OTX datasets

        if not test_mode:
            self._set_group_flag()
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        """Set flag for grouping images.

        Originally, in Custom dataset, images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        This implementation will set group 0 for every image.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        _ = idx
        return np.random.choice(len(self))

    @check_input_parameters_type()
    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.data_infos[idx])  # Copying dict(), not contents
        self.pre_pipeline(item)
        return self.pipeline(item)

    @check_input_parameters_type()
    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.data_infos[idx])  # Copying dict(), not contents
        self.pre_pipeline(item)
        return self.pipeline(item)

    @staticmethod
    @check_input_parameters_type()
    def pre_pipeline(results: Dict[str, Any]):
        """Prepare results dict for pipeline. Add expected keys to the dict."""
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []

    @check_input_parameters_type()
    def get_ann_info(self, idx: int):
        """This method is used for evaluation of predictions.

        The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.otx_dataset[idx]
        labels = self.labels
        return get_annotation_mmdet_format(dataset_item, labels, self.domain)

    def evaluate(  # pylint: disable=too-many-branches
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["mAP"]
        eval_results = OrderedDict()
        for metric in metrics:  # pylint: disable=redefined-argument-from-local
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            if metric == "mAP":
                assert isinstance(iou_thrs, list)
                mean_aps = []
                for iou_thr in iou_thrs:  # pylint: disable=redefined-argument-from-local
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    if isinstance(results[0], tuple):
                        mean_ap, _ = eval_segm(
                            results,
                            annotations,
                            iou_thr=iou_thr,
                            dataset=self.CLASSES,
                            logger=logger,
                            metric=metric,
                        )
                    else:
                        mean_ap, _ = eval_map(
                            results,
                            annotations,
                            scale_ranges=scale_ranges,
                            iou_thr=iou_thr,
                            dataset=self.CLASSES,
                            logger=logger,
                        )
                    mean_aps.append(mean_ap)
                    eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
                eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
        return eval_results


# pylint: disable=too-many-arguments
@DATASETS.register_module()
class ImageTilingDataset:
    """A wrapper of tiling dataset.

    Suitable for training small object dataset. This wrapper composed of `Tile`
    that crops an image into tiles and merges tile-level predictions to
    image-level prediction for evaluation.

    Args:
        dataset (Config): The dataset to be tiled.
        pipeline (List): Sequence of transform object or
            config dict to be composed.
        tile_size (int): the length of side of each tile
        min_area_ratio (float, optional): The minimum overlap area ratio
            between a tiled image and its annotations. Ground-truth box is
            discarded if the overlap area is less than this value.
            Defaults to 0.8.
        overlap_ratio (float, optional): ratio of each tile to overlap with
            each of the tiles in its 4-neighborhood. Defaults to 0.2.
        iou_threshold (float, optional): IoU threshold to be used to suppress
            boxes in tiles' overlap areas. Defaults to 0.45.
        max_per_img (int, optional): if there are more than max_per_img bboxes
            after NMS, only top max_per_img will be kept. Defaults to 200.
    """

    def __init__(
        self,
        dataset: Config,
        pipeline: List[dict],
        tile_size: int,
        min_area_ratio=0.8,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=200,
        filter_empty_gt=True,
        test_mode=False,
    ):
        self.dataset = build_dataset(dataset)
        self.CLASSES = self.dataset.CLASSES
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

        self.tile_dataset = Tile(
            self.dataset,
            pipeline,
            tmp_dir=self.tmp_dir,
            tile_size=tile_size,
            overlap=overlap_ratio,
            min_area_ratio=min_area_ratio,
            iou_threshold=iou_threshold,
            max_per_img=max_per_img,
            filter_empty_gt=False if test_mode else filter_empty_gt,
        )
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.num_samples = len(self.dataset)  # number of original samples
        self.merged_results: Union[List[Tuple[np.ndarray, list]], List[np.ndarray]] = []

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.tile_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get training/test tile.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
            True).
        """
        return self.pipeline(self.tile_dataset[idx])

    def evaluate(self, results, **kwargs) -> Dict[str, float]:
        """Evaluation on Tile dataset.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            **kwargs: Addition keyword arguments.

        Returns:
            dict[str, float]: evaluation metric.
        """
        self.merged_results = self.tile_dataset.merge(results)
        return self.dataset.evaluate(self.merged_results, **kwargs)

    def merge(self, results) -> Union[List[Tuple[np.ndarray, list]], List[np.ndarray]]:
        """Merge tile-level results to image-level results.

        Args:
            results: tile-level results.

        Returns:
            merged_results (list[list | tuple]): Merged results of the dataset.
        """
        self.merged_results = self.tile_dataset.merge(results)
        return self.merged_results

    def __del__(self):
        """Delete the temporary directory when the object is deleted."""
        if getattr(self, "tmp_dir", False):
            self.tmp_dir.cleanup()
