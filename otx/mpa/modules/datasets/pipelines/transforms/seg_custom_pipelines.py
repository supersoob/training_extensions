# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines.formating import to_tensor


class BoxMaskGenerator(object):
    def __init__(
        self,
        prop_range=(0.25, 0.4),
        n_boxes=1,
        random_aspect_ratio=True,
        prop_by_area=True,
        within_bounds=True,
        invert=True,
    ):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0) : int(y1), int(x0) : int(x1)] = (
                    1 - masks[i, 0, int(y0) : int(y1), int(x0) : int(x1)]
                )
        return masks


@PIPELINES.register_module()
class Cutmix(object):
    def __init__(
        self,
        prop_range=(0.25, 0.4),
        n_boxes=1,
        random_aspect_ratio=True,
        prop_by_area=True,
        within_bounds=True,
        invert=True,
    ):
        self.mask_generator = BoxMaskGenerator(
            prop_range, n_boxes, random_aspect_ratio, prop_by_area, within_bounds, invert
        )

    def __call__(self, results):
        img0 = results["img"]
        breakpoint()
        if results.get("img1", None):
            img1 = results["img1"]

            mask_size = img0.shape[2:]
            n_masks = img0.shape[0]
            masks = torch.Tensor(self.mask_generator.generate_params(n_masks, mask_size))
            cutmix_img = (1 - masks) * img0 + masks * img1

            results["cutmix.img"] = cutmix_img
            results["cutmix.masks"] = masks

        return results


@PIPELINES.register_module(force=True)
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        for target in ["img", "ul_w_img", "aux_img"]:
            if target in results:
                results[target] = mmcv.imnormalize(results[target], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        for target in ["img", "ul_w_img", "aux_img"]:
            if target not in results:
                continue

            img = results[target]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)

            results[target] = DC(to_tensor(img), stack=True)

        for trg_name in ["gt_semantic_seg", "gt_class_borders", "pixel_weights"]:
            if trg_name not in results:
                continue

            out_type = np.float32 if trg_name == "pixel_weights" else np.int64
            results[trg_name] = DC(to_tensor(results[trg_name][None, ...].astype(out_type)), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class BranchImage(object):
    def __init__(self, key_map={}):
        self.key_map = key_map

    def __call__(self, results):
        for k1, k2 in self.key_map.items():
            if k1 in results:
                results[k2] = results[k1]
            if k1 in results["img_fields"]:
                results["img_fields"].append(k2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
