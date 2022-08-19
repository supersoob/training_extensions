from typing import Dict

import torch
import torch.nn as nn


class Backbone(object):
    def __init__(self, module:nn.Module, config: Dict = None):
        self.module = module