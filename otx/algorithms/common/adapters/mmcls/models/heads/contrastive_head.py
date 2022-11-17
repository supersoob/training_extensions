import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import HEADS, build_neck


@HEADS.register_module()
class ConstrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args
        predictor (dict): configurations for predictor.
        size_average (bool): whether averaging loss using batch size. Default value is True.
    """

    def __init__(self, predictor, size_average=True, **kwargs):
        super(ConstrastiveHead, self).__init__()
        self.predictor = build_neck(predictor)
        self.size_average = size_average

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        loss = 2 * input.size(0) - 2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= input.size(0)

        return dict(loss=loss)
