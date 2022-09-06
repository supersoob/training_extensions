import os

from ote import OTEConstants
from ote.core.config import Config
from ote.recipe import Recipe


def test_recipe():
    yaml_file_path = os.path.join(OTEConstants.RECIPES_PATH, "sample_recipe_cls.yaml")
    recipe = Recipe(yaml_file_path)
    assert isinstance(recipe, Recipe)

    # recipe.run(dataset="dataset")

    result = recipe.train(mode="train")
    assert isinstance(result, dict)
    assert result.get("final_ckpt") is not None
