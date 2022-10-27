import collections
from copy import deepcopy

import numpy as np
from mmcv.utils import build_from_cfg
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class SelfSLCompose(object):
    """
    Compose pre-processed data for Self-supervised learning (SSL).
    Through interval, how frequently SSL pipeline (pipeline1 + pipeline2) is applied is set.
    """
    def __init__(self, pipeline1, pipeline2):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline1])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline2])
        self.is_supervised = False

    def __call__(self, data):
        if self.is_supervised:
            data = self.pipeline1(data)
            
        else:
            data1 = self.pipeline1(deepcopy(data))
            h1, w1, _ = data1['img_metas'].data['img_shape']
            self.pipeline2.transforms[1].img_scale = [(w1, h1)]
            
            data2 = self.pipeline2(deepcopy(data))        
            
            data = deepcopy(data1)
            data['img'] = (data1['img'], data2['img'])
            data['img_metas'] = (data1['img_metas'], data2['img_metas'])
            data['gt_bboxes'] = (data1['gt_bboxes'], data2['gt_bboxes'])
            data['gt_labels'] = (data1['gt_labels'], data2['gt_labels'])

        data['is_supervised'] = self.is_supervised

        return data


@PIPELINES.register_module()
class ProbCompose(object):
    def __init__(self, transforms, probs):
        assert isinstance(transforms, collections.abc.Sequence)
        assert isinstance(probs, collections.abc.Sequence)
        assert len(transforms) == len(probs)
        assert all(p >= 0.0 for p in probs)

        sum_probs = float(sum(probs))
        assert sum_probs > 0.0
        norm_probs = [float(p) / sum_probs for p in probs]
        self.limits = np.cumsum([0.0] + norm_probs)

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        rand_value = np.random.rand()
        transform_id = np.max(np.where(rand_value > self.limits)[0])

        transform = self.transforms[transform_id]
        data = transform(data)

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string
