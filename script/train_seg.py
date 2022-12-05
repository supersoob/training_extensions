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
    WORKDIR_ROOT: str = 'work_dirs/warmstart/seg/detcon'
):
    if isinstance(MODELS, str):
        MODELS = [MODELS]

    if isinstance(DATASETS, str):
        DATASETS = [DATASETS]

    for MODEL in MODELS:
        if MODEL not in TEMPLATE_DICT:
            raise ValueError()

        for DATASET in DATASETS:
            if DATASET in ['pascal_voc']:
                VAL = f'val_subset100'
            else:
                raise ValueError()

            # pretrained
            WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/pretrained')
            if not os.path.isdir(WORKDIR):
                os.makedirs(f'{WORKDIR}', exist_ok=True)

                command = (
                    f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                        f'otx/algorithms/segmentation/configs/{MODEL}_warmstart/template_experiment.yaml '
                        f'--train-data-roots=dataset/{DATASET}/train_aug/image '
                        f'--train-ann-files=dataset/{DATASET}/detcon_mask '
                        f'--save-model-to={WORKDIR} '
                )

                command += f'2>&1 | tee {WORKDIR}/output.log'
                run_command(command, WORKDIR)

            # finetuning
            for MODE in MODES:
                for NUMDATA in NUMDATAS:
                    for CLASS in CLASSES:
                        for SEED in SEEDS:
                            WORKDIR = os.path.join(WORKDIR_ROOT, f'{MODEL}/{DATASET}/{MODE}_{CLASS}_#{NUMDATA}_seed{SEED}')
                            os.makedirs(f'{WORKDIR}', exist_ok=True)

                            # if NUMDATA == 'full':
                            #     TRAIN = f'train'
                            # else:
                            #     TRAIN = f'train_{NUMDATA}'

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
                                command += f'--load-weights={os.path.join(WORKDIR_ROOT, f"{MODEL}/{DATASET}/pretrained/weights.pth")} '

                            command += f'2>&1 | tee {WORKDIR}/output.log'
                            run_command(command, WORKDIR)

if __name__ == '__main__':
    pass