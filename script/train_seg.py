import os
import shutil
import subprocess


TEMPLATE_DICT = {
    'ocr_lite_hrnet_s_mod2': 'template.yaml',
    'ocr_lite_hrnet_18_mod2': 'template.yaml',
    'ocr_lite_hrnet_x_mod3': 'template.yaml',
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
    CLASSES,
    SEEDS,
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
        BATCHSIZES = [8]

    if LRS is None:
        LRS = [0.001]

    for MODEL in MODELS:
        if MODEL not in TEMPLATE_DICT:
            raise ValueError()

        for DATASET in DATASETS:
            if DATASET in ['pascal_voc']:
                WARMSTART_TRAIN_IMG = f'dataset/{DATASET}/train_aug/image'
                WARMSTART_TRAIN_MASK = f'dataset/{DATASET}/detcon_mask'
                VAL = f'val_subset100'

            elif DATASET in ['cityscapes']:
                WARMSTART_TRAIN_IMG = f'dataset/{DATASET}/warmstart/image'
                WARMSTART_TRAIN_MASK = f'dataset/{DATASET}/warmstart/label'
                VAL = f'val_subset100'

            else:
                raise ValueError()

            for BATCHSIZE in BATCHSIZES:
                for LR in LRS:

                    for MODE in MODES:
                        assert MODE in ['warmstart', 'sup']
                        if MODE == 'warmstart':
                            # pretrained
                            WARMSTART_WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/pretrained')
                            if BATCHSIZE != 8 or LR != 0.001:
                                WARMSTART_WORKDIR += f'_batch{BATCHSIZE}_lr{LR}'

                            if not os.path.isfile(os.path.join(WARMSTART_WORKDIR, 'weights.pth')):
                                # warmstart
                                os.makedirs(f'{WARMSTART_WORKDIR}', exist_ok=True)

                                command = (
                                    f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                                        f'otx/algorithms/segmentation/configs/{MODEL}_warmstart/template_experiment.yaml '
                                        f'--train-data-roots={WARMSTART_TRAIN_IMG} '
                                        f'--train-ann-files={WARMSTART_TRAIN_MASK} '
                                        f'--save-model-to={WARMSTART_WORKDIR} '
                                )
                                if BATCHSIZE != 8 or LR != 0.001:
                                    command += (
                                        f'params '
                                        f'--learning_parameters.batch_size={BATCHSIZE} '
                                        f'--learning_parameters.learning_rate={LR} '
                                    )

                                command += f'2>&1 | tee {WARMSTART_WORKDIR}/output.log'
                                run_command(command, WARMSTART_WORKDIR)

                        # finetuning
                        for NUMDATA in NUMDATAS:
                            for CLASS in CLASSES:
                                for SEED in SEEDS:
                                    WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/{MODE}_{CLASS}_#{NUMDATA}_seed{SEED}')
                                    if os.path.isfile(os.path.join(WORKDIR, 'weights.pth')):
                                        # skip if training was already done before
                                        continue

                                    os.makedirs(f'{WORKDIR}', exist_ok=True)

                                    command = (
                                        f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                                            f'otx/algorithms/segmentation/configs/{MODEL}/{TEMPLATE_DICT[MODEL]} '
                                            f'--train-ann-files=dataset/{DATASET}/subset_supcon/partial/{CLASS}/{NUMDATA}/seed{SEED}/label '
                                            f'--train-data-root=dataset/{DATASET}/subset_supcon/partial/{CLASS}/{NUMDATA}/seed{SEED}/image '
                                            f'--val-data-roots=dataset/{DATASET}/subset_supcon/partial/{VAL}_{CLASS}/image '
                                            f'--val-ann-files=dataset/{DATASET}/subset_supcon/partial/{VAL}_{CLASS}/label '
                                            f'--save-model-to={WORKDIR} '
                                    )
                                    if MODE == 'warmstart':
                                        command += f'--load-weights={os.path.join(WARMSTART_WORKDIR, "weights.pth")} '

                                    command += f'2>&1 | tee {WORKDIR}/output.log'
                                    run_command(command, WORKDIR)


if __name__ == '__main__':
    pass