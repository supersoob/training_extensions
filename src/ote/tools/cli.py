from ote.registry import get_tasks, get_recipe_names_for_task, get_recipe
from ote.core import Recipe, OTEDataset

def find_tasks():
    return get_tasks()


def find_recipes(task):
    return get_recipe_names_for_task(task)


def get_recipe(name):
    return get_recipe(name)


def train(recipe: Recipe, dataset: OTEDataset):
    if isinstance(recipe, Recipe):
        recipe.train()