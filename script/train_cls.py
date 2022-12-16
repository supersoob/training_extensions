import os
import time
import shutil
import subprocess


TEMPLATE_DICT = {
    'efficientnet_b0': 'template.yaml',
    'efficientnet_v2_s': 'template.yaml',
    'mobilenet_v3_large_1': 'template.yaml',
    'mobilenet_v3_large_075': 'template_experiment.yaml',
    'mobilenet_v3_small': 'template_experiment.yaml'
}

DEFAULT_BATCH = 256
DEFAULT_LR = 0.45


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


def warmstart(
    GPUS: int,
    MODELS,
    DATASETS,
    BATCHSIZES=None,
    LRS=None,
    WORKDIR_ROOT: str = 'work_dirs/warmstart/seg/detcon'
):
    """
    `BATCHSIZES` and `LRS` are parameters just for warmstart.
    """
    if isinstance(MODELS, str):
        MODELS = [MODELS]

    if isinstance(DATASETS, str):
        DATASETS = [DATASETS]

    if BATCHSIZES is None:
        BATCHSIZES = [DEFAULT_BATCH]

    if LRS is None:
        LRS = [DEFAULT_LR]

    for MODEL in MODELS:
        if MODEL not in TEMPLATE_DICT:
            raise ValueError()

        for DATASET in DATASETS:
            for BATCHSIZE in BATCHSIZES:
                for LR in LRS:
                    # pretrained
                    WARMSTART_WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/pretrained_224')
                    if BATCHSIZE != DEFAULT_BATCH or LR != DEFAULT_LR:
                        WARMSTART_WORKDIR += f'_batch{BATCHSIZE}_lr{LR}'

                    if not os.path.isfile(os.path.join(WARMSTART_WORKDIR, 'weights.pth')):
                        # warmstart
                        os.makedirs(f'{WARMSTART_WORKDIR}', exist_ok=True)

                        command = (
                            f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                                f'otx/algorithms/classification/configs/{MODEL}_cls_incr/selfsl/template_experimental.yaml '
                                f'--train-data-roots=dataset/{DATASET}/train '
                                f'--save-model-to={WARMSTART_WORKDIR} '
                        )
                        if BATCHSIZE != DEFAULT_BATCH or LR != DEFAULT_LR:
                            command += (
                                f'params '
                                f'--learning_parameters.batch_size={BATCHSIZE} '
                                f'--learning_parameters.learning_rate={LR} '
                            )

                        command += f'2>&1 | tee {WARMSTART_WORKDIR}/output.log'
                        run_command(command, WARMSTART_WORKDIR)


def finetuning(
    GPUS: int,
    MODELS,
    DATASETS,
    MODES,
    NUMDATAS,
    BATCHSIZES=None,
    LRS=None,
    WORKDIR_ROOT: str = 'work_dirs/warmstart/seg/detcon'
):
    """
    `BATCHSIZES` and `LRS` are parameters just for warmstart.
    """
    if isinstance(MODELS, str):
        MODELS = [MODELS]

    if isinstance(DATASETS, str):
        DATASETS = [DATASETS]

    if BATCHSIZES is None:
        BATCHSIZES = [DEFAULT_BATCH]

    if LRS is None:
        LRS = [DEFAULT_LR]

    for MODEL in MODELS:
        if MODEL not in TEMPLATE_DICT:
            raise ValueError()

        for DATASET in DATASETS:
            if DATASET in ['cifar10', 'xray', 'svhn', 'food-101']:
                VAL = f'test'
            elif DATASET in ['CIFAR100']:
                VAL = f'val'
            else:
                raise ValueError()

            for BATCHSIZE in BATCHSIZES:
                for LR in LRS:

                    for MODE in MODES:
                        assert MODE in ['warmstart', 'sup']
                        if MODE == 'warmstart':                            
                            # pretrained
                            WARMSTART_WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/pretrained_224')
                            if BATCHSIZE != DEFAULT_BATCH or LR != DEFAULT_LR:
                                WARMSTART_WORKDIR += f'_batch{BATCHSIZE}_lr{LR}'

                            while not os.path.isfile(os.path.join(WARMSTART_WORKDIR, 'weights.pth')):
                                # check warmstart was already done every a minute
                                # if not, wait until warmstart is done
                                time.sleep(60)

                        # finetuning
                        for NUMDATA in NUMDATAS:
                            WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/{NUMDATA}/{MODE}_224')
                            if MODE == 'warmstart' and (BATCHSIZE != DEFAULT_BATCH or LR != DEFAULT_LR):
                                WORKDIR += f'_batch{BATCHSIZE}_lr{LR}'
                            
                            if os.path.isfile(os.path.join(WORKDIR, 'weights.pth')):
                                # skip if training was already done before
                                continue

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
                                command += f'--load-weights={os.path.join(WARMSTART_WORKDIR, "weights.pth")} '

                            command += f'2>&1 | tee {WORKDIR}/output.log'
                            run_command(command, WORKDIR)


if __name__ == '__main__':
    pass