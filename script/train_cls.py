import os
import shutil
import subprocess


TEMPLATE_DICT = {
    'efficientnet_b0': 'template.yaml',
    'efficientnet_v2_s': 'template.yaml',
    'mobilenet_v3_large_1': 'template.yaml',
    'mobilenet_v3_large_075': 'template_experiment.yaml',
    'mobilenet_v3_small': 'template_experiment.yaml'
}


def run_command(command, WORKDIR):
    print(command)
    subprocess.run(command, shell=True)

    # move logs in the tmp dir to the real dir
    with open(f'{WORKDIR}/output.log', 'r') as f:
        logs = f.readlines()
        dir_logs = [d for d in logs if 'work dir = ' in d][0]
        work_dir = dir_logs[dir_logs.rfind('= ')+2:].rstrip()

    files = os.listdir(work_dir)
    for file in files:
        try:
            shutil.copy(os.path.join(work_dir, file), os.path.join(WORKDIR, file))
        except:
            pass


def main(
    GPUS: int,
    MODELS,
    DATASETS,
    MODES,
    NUMDATAS,
    WORKDIR_ROOT: str = 'work_dirs/warmstart/cls/byol'
):
    if isinstance(MODELS, str):
        MODELS = [MODELS]

    if isinstance(DATASETS, str):
        DATASETS = [DATASETS]

    for MODEL in MODELS:
        if MODEL not in TEMPLATE_DICT:
            raise ValueError()

        for DATASET in DATASETS:
            if DATASET in ['cifar10', 'xray', 'svhn']:
                VAL = f'test'
            elif DATASET in ['CIFAR100']:
                VAL = f'val'

            # pretrained
            WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/pretrained')
            if not os.path.isdir(WORKDIR):
                os.makedirs(f'{WORKDIR}', exist_ok=True)

                command = (
                    f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                        f'otx/algorithms/classification/configs/{MODEL}_cls_warmstart/template_experiment.yaml '
                        f'--train-data-roots=dataset/{DATASET}/train '
                        f'--val-data-roots=dataset/{DATASET}/{VAL} '
                        f'--save-model-to={WORKDIR} '
                )

                command += f'2>&1 | tee {WORKDIR}/output.log'
                run_command(command, WORKDIR)

            # finetuning
            for MODE in MODES:
                for NUMDATA in NUMDATAS:
                    WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/{MODE}_{NUMDATA}')
                    os.makedirs(f'{WORKDIR}', exist_ok=True)

                    if NUMDATA == 'full':
                        TRAIN = f'train'
                    else:
                        TRAIN = f'train_{NUMDATA}'

                    command = (
                        f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                            f'otx/algorithms/classification/configs/{MODEL}_cls_incr/{TEMPLATE_DICT[MODEL]} '
                            f'--train-data-roots=dataset/{DATASET}/{TRAIN} '
                            f'--val-data-roots=dataset/{DATASET}/{VAL} '
                            f'--save-model-to={WORKDIR} '
                    )
                    if MODE == 'warmstart':
                        command += f'--load-weights={os.path.join(WORKDIR_ROOT, f"{MODEL}/{DATASET}/pretrained/weights.pth")} '

                    command += f'2>&1 | tee {WORKDIR}/output.log'
                    run_command(command, WORKDIR)

if __name__ == '__main__':
    pass
