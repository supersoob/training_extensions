from ote.backends.torch.mmcv.classification.job import MMClsJob
from ote.core.job import JOB_REGISTRY
from ote.logger import get_logger

logger = get_logger()

@JOB_REGISTRY.register_module()
class MMClsInferrer(MMClsJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        logger.info(f"{__name__} __init__({kwargs})")

    def run(self, **kwargs):
        logger.info(f"{__name__} run({kwargs})")