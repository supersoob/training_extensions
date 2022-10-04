from ote.backends.torch.task import TorchJob
from ote.core.task import JOB_REGISTRY


@JOB_REGISTRY.register_module()
class TorchOptimizer(TorchJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec)

    def run(self, model, options, **kwargs):
        logger.info(f"{__name__} run({kwargs})")