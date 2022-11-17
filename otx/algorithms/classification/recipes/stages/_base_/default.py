_base_ = [
    # './dist/dist.py'
    '../../../venv/lib/python3.8/site-packages/recipes/stages/_base_/dist/dist.py',
]

cudnn_benchmark = True

seed = 5
deterministic = False

hparams = dict(dummy=0)

# task_adapt = dict(op='REPLACE')
