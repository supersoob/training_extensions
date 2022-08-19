import os
from ote import OTEConstants
from ote.core.config import OTEConfig


def test_config():
    yaml_file_path = os.path.join(OTEConstants.RECIPES_PATH, "sample_recipe_cls.yaml")
    config = OTEConfig.fromfile(yaml_file_path)
    assert isinstance(config, OTEConfig)
    print(f"ConfigParams obj = {obj}")
