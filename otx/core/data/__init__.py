"""OTX Core Data."""

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

# pylint: disable=too-many-return-statements
import os
import datumaro

from otx.api.entities.model_template import TaskType


def get_dataset_adapter(task_type):
    """Returns a dataset class by task type.

    Args:
        task_type: A task type such as ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION,
        CLASSIFICATION, INSTANCE_SEGMENTATION, DETECTION, CLASSIFICATION, ROTATED_DETECTION, SEGMENTATION.
    """
    if task_type == TaskType.CLASSIFICATION:
        from .classification_dataset_adapter import ClassificationDatasetAdapter

        return ClassificationDatasetAdapter(task_type=task_type)

    if task_type in [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION]:
        from .detection_dataset_adapter import DetectionDatasetAdapter

        return DetectionDatasetAdapter(task_type=task_type)

    if task_type == TaskType.SEGMENTATION:
        from .segmentation_dataset_adapter import SegmentationDatasetAdapter

        return SegmentationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ACTION_CLASSIFICATION:
        from .action_dataset_adapter import ActionClassificationDatasetAdapter

        return ActionClassificationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        from .anomaly_dataset_adapter import AnomalyClassificationDatasetAdapter

        return AnomalyClassificationDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_DETECTION:
        from .anomaly_dataset_adapter import AnomalyDetectionDatasetAdapter

        return AnomalyDetectionDatasetAdapter(task_type=task_type)

    if task_type == TaskType.ANOMALY_SEGMENTATION:
        from .anomaly_dataset_adapter import AnomalySegmentationDatasetAdapter

        return AnomalySegmentationDatasetAdapter(task_type=task_type)

    # TODO: Need to implement
    # if task_type == TaskType.ACTION_DETECTION:
    #    from .action_dataset_adapter import ActionDetectionDatasetAdapter
    #
    #    return ActionDetectionDatasetAdapter(task_type=task_type)
    # if task_type == TaskType.ROTATED_DETECTION:
    #    from .rotated_detection.dataset import RotatedDetectionDataset
    #
    #    return RotatedDetectionDataset

    raise ValueError(f"Invalid task type: {task_type}")

class TaskTypeFinder:
    """To find task type by using dataset root."""
    def __init__(self):
        self.task_data_dict = {
            "classification": ['imagenet'],
            "detection": ['coco', 'voc', 'yolo'],
            "instance_segmentation": ['coco', 'voc'],
            "semantic_segmentation": [
                'common_semantic_segmentation',
                'voc',
                'cityscapes',
                'ade20k2017',
                'ade20k2020'
            ]
        }
    @classmethod
    def find_task_type(cls, data_root):
        """Detect task type."""
        task_type_candidates = []
        data_format = datumaro.Environment().detect_dataset(data_root)
        for task in cls.task_data_dict:
            if data_format in cls.task_data_dict[task]:
                task_type_candidates.append(task)
    
    def _is_cvat_format(self, path):
        """Detect whether data path is CVAT format or not."""
        #TODO: Will be supported by dautmaro detect_dataset. 
        pass


    def _is_mvtec(self, path):
        """Detect whether data path is MVTec format or not."""

        # condition 1: 'ground_truth', 'train', 'test' folder are located
        mvtec_folders = sorted(['ground_truth', 'train', 'test'])
        folder_list = []
        for sub in os.listdir(path):
            if os.isdir(sub):
                folder_list.append(sub)
        return mvtec_folders == sorted(folder_list)
        
        # condition 2: 