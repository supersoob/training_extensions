import numpy as np
import torch
from mmseg.core import add_prefix
from mmseg.datasets import DATASETS, CustomDataset, build_dataset
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class UnlabeledSegDataset(CustomDataset):
    """Dataset wrapper for Semi-SL Semantic Seg experiments.
    Input : splits of labeled & unlabeld datasets
    """

    def __init__(self, orig_type=None, **kwargs):
        # Original dataset
        dataset_cfg = kwargs.copy()
        if "cutmix" in dataset_cfg:
            self.cutmix_flag = dataset_cfg.pop("cutmix", False)
        else:
            self.cutmix_flag = False
        dataset_cfg["type"] = orig_type
        self.unlabeled_dataset = build_dataset(dataset_cfg)

        # TODO remove
        self.cutmix_flag = True

        # Subsets
        self.num_unlabeled = len(self.unlabeled_dataset)
        self.unlabeled_index = np.random.permutation(self.num_unlabeled)
        if self.cutmix_flag:
            self.cutmix_unlabeled_index = np.random.permutation(self.num_unlabeled)
            self.mask_generator = BoxMaskGenerator()
        print("----------- #Unlabeled: ", self.num_unlabeled)

    def __len__(self):
        """Total number of samples of data."""
        return self.num_unlabeled

    def _cutmix(self, img0, img1):
        mask_size = img0.shape[1:]
        n_masks = 1
        masks = torch.Tensor(self.mask_generator.generate_params(n_masks, mask_size))
        cutmix_img = (1 - masks) * img0 + masks * img1

        return dict(img=torch.squeeze(cutmix_img), masks=torch.squeeze(masks))

    def __getitem__(self, idx):
        unlabeled_idx = int(self.unlabeled_index[idx])
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]

        if self.cutmix_flag:
            unlabeled_cutmix_idx = int(self.cutmix_unlabeled_index[idx])
            unlabeled_cutmix_data = self.unlabeled_dataset[unlabeled_cutmix_idx]

            cutmixed_data = self._cutmix(unlabeled_data["img"].data, unlabeled_cutmix_data["img"].data)
            unlabeled_data.update(add_prefix(unlabeled_cutmix_data, "img1"))
            unlabeled_data.update(add_prefix(cutmixed_data, "cutmix"))

        return unlabeled_data


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
