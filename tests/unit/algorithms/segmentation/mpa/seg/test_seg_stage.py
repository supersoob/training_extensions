
from otx.mpa.seg.stage import SegStage

class TestOTXSegStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        default_config = "./otx/recipes/stages/segmentation/incremental.py"
        self.stage = SegStage(name="",mode="train", config=default_config, common_cfg=None, index = 0)

    @e2e_pytest_unit
    def configure(self, mocker):
        mocker.patch.object(SegStage, "configure_model")
        mocker.patch.object(SegStage, "configure_ckpt")
        mocker.patch.object(SegStage, "configure_data")
        mocker.patch.object(SegStage, "configure_task")
        mocker.patch.object(SegStage, "configure_hook")