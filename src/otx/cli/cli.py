from ote.registry import get_tasks, get_recipe_names_for_task, get_recipe
from ote.core import Recipe

def find_tasks():
    return get_tasks()


def find_recipes(task):
    return get_recipe_names_for_task(task)


def get_recipe(name):
    return get_recipe(name)



def build():
    pass

def build_recipe():
    pass

def build_model():
    pass


def train(recipe_file):
    recipe = Recipe(recipe_file)
    if isinstance(recipe, Recipe):
        recipe.train()

def eval():
    pass

def infer():
    pass

def export():
    pass