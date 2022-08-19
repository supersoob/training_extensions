import os

from ote import OTEConstants
from ote.core.recipe import Recipe


class __Registry():
    def __init__(self, recipe_path: str):
        # TODO: retrieve recipe from builtin recipe path + given recipe_path
        self._recipes = dict(
            classification=dict(
                classifier=f"{recipe_path}/recipes/sample_classifier.yaml",
                multilabel="",
                hierachical="",
            )
        )

        # TODO: retrieve models from builtin path
        self._models = dict(
            classification=dict(
                effnet_b0="recipes/classification/models/ote_efficientnet_b0.yaml",
                mnet_small="recipes/classification/models/ote_mobilenet_v3_small.yaml",
            )
        )

        # TODO: define supported mapping between a recipe and models
        self._supported_map = dict()

    @property
    def recipes(self):
        return self._recipes


__registry = __Registry(OTEConstants.RECIPES_PATH)

def get_tasks():
    tasks = []
    for k, _ in __registry.recipes.items():
        tasks.append(k)
    return tasks

def get_recipe_names_for_task(task):
    if task not in __registry.recipes.keys():
        print(f"cannot find recipes in the '{task}' task")
        return None
    recipe_names = []
    for k, _ in __registry.recipes[task].items():
        recipe_names.append(k)
    return recipe_names

def get_recipe(task, name):
    if task not in __registry.recipes.keys():
        print(f"cannot find recipes in the '{task}' task")
        return None
    if name not in __registry.recipes[task].keys():
        print(f"cannot find recipe name '{name}' in the '{task}' task")
        return None
    recipe_yaml = __registry.recipes[task][name]
    return Recipe(recipe_yaml)