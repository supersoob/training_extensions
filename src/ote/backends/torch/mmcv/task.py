from ote.backends.torch.task import TorchTask
from ote.core.config import OTEConfig


class MMTask(TorchTask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
