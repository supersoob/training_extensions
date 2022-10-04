import os

from . import backends

class OTEConstants:
    PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RECIPES_PATH = os.path.join(PACKAGE_ROOT, 'recipes')
    # SAMPLES_PATH = os.path.join(PACKAGE_ROOT, 'samples')
    # MODELS_PATH = os.path.join(PACKAGE_ROOT, 'models')

__all__ = [
    OTEConstants
]