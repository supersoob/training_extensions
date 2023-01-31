import os

import pytest

from otx.mpa import Stage
from otx.mpa.seg.trainer import SegTrainer
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit

DEFAULT_SEG_TEMPLATE_DIR = os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2")
DEFAULT_CONFIG_PATH = "./otx/recipes/stages/segmentation/incremental.py"


class TestOTXSegTrainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_CONFIG_PATH)
        self.trainer = SegTrainer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    def test_seg_trainer_run(self, mocker):
        mocker.patch("mmseg.datasets.builder.build_dataset")
        mocker.patch("mmseg.apis.train_segmentor")
        self.trainer.run(self.model_cfg, "", self.data_cfg)
        