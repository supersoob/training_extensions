from ote.core.task import TASK_REGISTRY
from ote.backends.torch.mmcv.classification.task import MMClsTask
from ote.logger import get_logger

logger = get_logger()

@TASK_REGISTRY.register_module()
class MMClsInferrer(MMClsTask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        logger.info(f"{__name__} __init__({kwargs})")

    def run(self, **kwargs):
        logger.info(f"{__name__} run({kwargs})")