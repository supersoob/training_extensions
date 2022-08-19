import pytest
from ote.registry import get_recipe, get_recipe_names_for_task, get_tasks


def test_registry():
    tasks = get_tasks()
    assert tasks is not None
    print(f"list of tasks = {tasks}")

    recipe_names = get_recipe_names_for_task("unknown")
    assert recipe_names is None

    recipe_names = get_recipe_names_for_task(tasks[0])
    assert recipe_names is not None
    print(f"recipe names for {tasks[0]} = {recipe_names}")

    recipe = get_recipe(tasks[0], recipe_names[0])
    assert recipe is not None

    with pytest.raises(Exception) as e:
        recipe.train()

    assert e.type == NotImplementedError
