from ote.backends.torch.task import TorchTask
from ote.core.task import TASK_REGISTRY


@TASK_REGISTRY.register_module()
class TorchOptimizer(TorchTask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec)

    def run(self, model, options, **kwargs):
        logger.info(f"{__name__} run({kwargs})")