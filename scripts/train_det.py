import importlib
import os
import shutil
import subprocess
from typing import Dict, List

from mmcv.utils.config import Config


def main(
    GPUS: int = 0,
    EXP: str = 'subset_geti1',
    DATASETS: List[str] = ['bccd', 'fish', 'pothole', 'vitens'],
    MODELS: List[str] = ['mobilenetv2_atss', 'mobilenetv2_ssd', 'cspdarknet_yolox'],
    MODES: List[str] = ['sup', 'sup_detcon', 'detcon_supcon'],
    NUM_DATAS: List[int] = [16, 32, 48],
    # hparams for the model
    INTERVALS: List[int] = [1],
    LAMBDAS: List[int] = [1],
):

    # hparams for training
    BATCHSIZES = {16: [8], 32: [8, 16], 48: [8, 24]}
    LRS = {
        'mobilenetv2_atss': [0.004],
        'mobilenetv2_ssd': [0.01],
        'cspdarknet_yolox': [0.0002]
    }

    cfg = None
    for DATASET in DATASETS:
        # DATAROOT = f'data/{DATASET}'
        if DATASET == 'fish':
            # REAL_CLASSES = '"["fish"]"'
            NUM_CLASSES = 2
        elif DATASET == 'bccd':
            # REAL_CLASSES = '"["Platelets", "RBC", "WBC"]"'
            NUM_CLASSES = 4
        elif DATASET == 'pothole':
            # REAL_CLASSES = '"["pothole"]"'
            NUM_CLASSES = 2
        elif DATASET == 'vitens':
            # REAL_CLASSES = '"["object"]"'
            NUM_CLASSES = 2
        else:
            raise ValueError()

        for MODEL in MODELS:
            for MODE in MODES:
                assert MODE in ['sup', 'sup_detcon', 'detcon_supcon']
                for NUM_DATA in NUM_DATAS:
                    for BATCHSIZE in BATCHSIZES[NUM_DATA]:
                        for LR in LRS[MODEL]:
                            for INTERVAL in INTERVALS:
                                for LAMBDA in LAMBDAS:
                                    for seed in [1, 2, 3, 4, 5]:
                                        # set command
                                        BASELINE_ROOT = f'otx/algorithms/detection/configs/detection/{MODEL}'
                                        SELFSL_ROOT = f'{BASELINE_ROOT}/selfsl'
                                        RECIPE_ROOT = f'{SELFSL_ROOT}/{DATASET}'
                                        if 'supcon' in MODE:
                                            # supcon
                                            OPTIONS = f'batch{BATCHSIZE}_lr{LR}_lambda{LAMBDA}_interval{INTERVAL}'
                                            WORKDIR = f'work_dirs/supcon/detection/{EXP}/{DATASET}/{NUM_DATA}/{OPTIONS}/{MODE}_{MODEL}/seed{seed}'
                                            RECIPE = f'{RECIPE_ROOT}/{OPTIONS}/{MODE}_{MODEL}'
                                        else:
                                            # supervised
                                            if 'detcon' in MODE:
                                                # supervised with self-sl pipeline
                                                assert len(LAMBDAS) == 1 and len(INTERVALS) == 1
                                                OPTIONS = f'batch{BATCHSIZE}_lr{LR}'
                                                WORKDIR = f'work_dirs/supcon/detection/{EXP}/{DATASET}/{NUM_DATA}/{OPTIONS}/{MODEL}_{MODE}/seed{seed}'
                                                RECIPE = f'{RECIPE_ROOT}/{OPTIONS}/{MODEL}_{MODE}'
                                            else:
                                                # supervised baseline
                                                assert len(LAMBDAS) == 1 and len(INTERVALS) == 1
                                                OPTIONS = f'batch{BATCHSIZE}_lr{LR}'
                                                WORKDIR = f'work_dirs/supcon/detection/{EXP}/{DATASET}/{NUM_DATA}/{OPTIONS}/{MODEL}/seed{seed}'
                                                RECIPE = f'{RECIPE_ROOT}/{OPTIONS}/{MODEL}'

                                        os.makedirs(RECIPE, exist_ok=True)


                                        ################## update template.yaml ##################
                                        shutil.copy(os.path.join(BASELINE_ROOT, 'template.yaml'), os.path.join(RECIPE, 'template.yaml'))
                                        template = Config.fromfile(os.path.join(RECIPE, 'template.yaml'))

                                        ## update hparams
                                        template['hyper_parameters']['base_path'] = '../' * 4 + template['hyper_parameters']['base_path']
                                        template['hyper_parameters']['parameter_overrides']['learning_parameters']['batch_size']['default_value'] = BATCHSIZE
                                        template['hyper_parameters']['parameter_overrides']['learning_parameters']['learning_rate']['default_value'] = LR

                                        template.dump(os.path.join(RECIPE, 'template.yaml'))


                                        ######## update model.py & copy data_pipeline.py ########
                                        if 'supcon' in MODE:
                                            shutil.copy(f'{SELFSL_ROOT}/data_pipeline_detcon_supcon.py', os.path.join(RECIPE, 'data_pipeline.py'))

                                            if (cfg is None) or (cfg.__name__ != os.path.join(SELFSL_ROOT, 'model_detcon_supcon').replace('/', '.')):
                                                cfg = importlib.import_module(os.path.join(SELFSL_ROOT, 'model_detcon_supcon').replace('/', '.'))
                                            else:
                                                cfg = importlib.reload(cfg)
                                            
                                            ## update _base_
                                            cfg._base_ = [
                                                '../' * (len(RECIPE.split('/')) - len(SELFSL_ROOT.split('/'))) + b for b in cfg._base_
                                            ] # 3

                                            ## update hparams
                                            cfg.model['num_classes'] = NUM_CLASSES
                                            cfg.model['loss_weights'] = {'detcon': LAMBDA}
                                            for hook in cfg.custom_hooks:
                                                if hook['type'] == 'SwitchPipelineHook':
                                                    hook['interval'] = INTERVAL

                                            new_cfg = Config(
                                                cfg_dict={k: getattr(cfg, k) for k in ['_base_', 'model', 'custom_hooks', 'load_from']}, 
                                                filename=os.path.join(SELFSL_ROOT, 'model_detcon_supcon.py')
                                            )

                                        else:
                                            if 'detcon' in MODE:
                                                shutil.copy(f'{SELFSL_ROOT}/data_pipeline_detcon.py', os.path.join(RECIPE, 'data_pipeline.py'))
                                            else:
                                                shutil.copy(f'{BASELINE_ROOT}/data_pipeline.py', os.path.join(RECIPE, 'data_pipeline.py'))

                                            if (cfg is None) or (cfg.__name__ != os.path.join(BASELINE_ROOT, 'model').replace('/', '.')):
                                                cfg = importlib.import_module(os.path.join(BASELINE_ROOT, 'model').replace('/', '.'))
                                            else:
                                                cfg = importlib.reload(cfg)

                                            ## update _base_
                                            cfg._base_ = [
                                                '../' * (len(RECIPE.split('/')) - len(BASELINE_ROOT.split('/'))) + b for b in cfg._base_
                                            ] # 4

                                            model_cfg_for_update = [k for k in ['_base_', 'model', 'fp16', 'load_from', '__width_mult', 'ignore'] if hasattr(cfg, k)]

                                            new_cfg = Config(
                                                cfg_dict={k: getattr(cfg, k) for k in model_cfg_for_update}, 
                                                filename=os.path.join(BASELINE_ROOT, 'model.py')
                                            )

                                        new_cfg.dump(os.path.join(RECIPE, 'model.py'))

                                        os.makedirs(WORKDIR, exist_ok=True)
                                        command = (
                                            f'CUDA_VISIBLE_DEVICES={GPUS} otx train '
                                                f'{os.path.join(RECIPE, "template.yaml")} '
                                                f'--train-ann-files=dataset/{DATASET}/annotations/instances_train_{NUM_DATA}_{seed}.json '
                                                f'--train-data-roots=dataset/{DATASET}/images/train '
                                                f'--val-ann-files=dataset/{DATASET}/annotations/instances_val_100.json '
                                                f'--val-data-roots=dataset/{DATASET}/images/val '
                                                f'--save-model-to={WORKDIR} '
                                        )
                                        print(command)

                                        command += f'2>&1 | tee {WORKDIR}/output.log'
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

if __name__ == '__main__':
    main(
        GPUS=0,
        EXP='test',
        DATASETS=['bccd', 'fish', 'pothole', 'vitens'],
        MODELS=['mobilenetv2_ssd'],
        MODES=['sup'],
        NUM_DATAS=[16, 32, 48],
        INTERVALS=[1],
        LAMBDAS=[1],
    )
