import functools
from collections import OrderedDict

import numpy as np
import torch
from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@SEGMENTORS.register_module()
class CutmixSegNaive(BaseSegmentor):
    def __init__(self, orig_type=None, unsup_weight=0.1, warmup_start_iter=0, **kwargs):
        print("CutmixSegNaive init!")
        super(CutmixSegNaive, self).__init__()
        self.test_cfg = kwargs["test_cfg"]
        self.warmup_start_iter = warmup_start_iter
        self.count_iter = 0

        cfg = kwargs.copy()
        if orig_type == "SemiSLSegmentor":
            cfg["type"] = "SemiSLSegmentor"
            self.align_corners = cfg["decode_head"][-1].align_corners
        else:
            cfg["type"] = "EncoderDecoder"
            self.align_corners = cfg["decode_head"].align_corners
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)

        self.unsup_weight = unsup_weight

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    def extract_feat(self, imgs):
        return self.model_s.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model_s.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model_s.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        return self.model_s.forward_dummy(img, **kwargs)

    def save_img(self, img_tensor, gt_tensor=None, filename=""):
        from torchvision.utils import save_image

        image = img_tensor[0].clone().detach().cpu()
        save_image(image, f"/home/soobee/training_extensions/cutmix_images/{filename}_data.png")

        if gt_tensor is not None:
            gt = gt_tensor[0].clone().detach().cpu()
            save_image(gt, f"/home/soobee/training_extensions/cutmix_images/{filename}_gt.png")

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        self.count_iter += 1
        if self.warmup_start_iter > self.count_iter:
            x = self.model_s.extract_feat(img)
            loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            return loss_decode

        ul_data = kwargs["extra_0"]
        ul_img0 = ul_data["img"]
        ul_img0_metas = ul_data["img_metas"]
        ul_img1 = ul_data["img1.img"]
        ul_img1_metas = ul_data["img1.img_metas"]

        ul_cutmix = ul_data["cutmix.img"]
        masks = ul_data["cutmix.masks"]

        # self.save_img(ul_img0, None, "img0")
        # self.save_img(ul_cutmix, None, "cutmix")

        with torch.no_grad():
            ul0_feat = self.model_t.extract_feat(ul_img0)
            ul0_logit = self.model_t._decode_head_forward_test(ul0_feat, ul_img0_metas)
            ul0_logit = resize(
                input=ul0_logit, size=ul_img0.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ul0_conf, ul0_pl = torch.max(torch.softmax(ul0_logit, axis=1), axis=1, keepdim=True)

            ul1_feat = self.model_t.extract_feat(ul_img1)
            ul1_logit = self.model_t._decode_head_forward_test(ul1_feat, ul_img1_metas)
            ul1_logit = resize(
                input=ul1_logit, size=ul_img1.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ul1_conf, ul1_pl = torch.max(torch.softmax(ul1_logit, axis=1), axis=1, keepdim=True)

            pl_cutmixed = (1 - masks) * ul0_pl + masks * ul1_pl
            pl_cutmixed = pl_cutmixed.long()

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u_cutmixed = self.model_s.extract_feat(ul_cutmix)
        loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)

        loss_decode_u, _ = self.model_s._decode_head_forward_train(
            x_u_cutmixed, ul_img0_metas, gt_semantic_seg=pl_cutmixed
        )

        for key in loss_decode_u.keys():
            losses[key] = loss_decode[key] + loss_decode_u[key] * self.unsup_weight

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect student model as output state_dict (teacher as auxilliary)"""
        logger.info("----------------- MeanTeacher.state_dict_hook() called")
        output = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model_s."):
                k = k.replace("model_s.", "")
                output[k] = v
        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to teacher model"""
        logger.info("----------------- MeanTeacher.load_state_dict_pre_hook() called")
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict["model_s." + k] = v
            state_dict["model_t." + k] = v
