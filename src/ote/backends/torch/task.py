from ote.core.task import OTETask


class TorchTask(OTETask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec)
