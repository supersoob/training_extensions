import argparse
import glob
import importlib
import json
import os
import os.path as osp
from datetime import datetime
# from mmcv import Config
from subprocess import run


def load_sc_dataset_cfg(data_cfg_path):
    if osp.isfile(data_cfg_path):
        with open(data_cfg_path) as json_file:
            sc_datasets_cfg = json.load(json_file)
        # print(f'Load SC Dataset CFG from: {data_cfg_path}')
        # spec = importlib.util.spec_from_file_location('sc_datasets_cfg',
        #                                               data_cfg_path)
        # sc_datasets_cfg = spec.loader.load_module()
        return sc_datasets_cfg
    raise Exception(f'Could not load Dataset CFG from: {data_cfg_path}')




def update_test_cfg():
    update_string = 'model.test_cfg.rpn.nms_pre=800 '
    update_string += 'model.test_cfg.rpn.nms_post=500 '
    update_string += 'model.test_cfg.rpn.max_num=500 '
    update_string += 'model.test_cfg.rcnn.max_per_img=500 '
    return update_string


def collect_metric(path, metric):
    """Collects average precision values in log file."""
    line = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   ' \
        'all | maxDets=100 ] = '
    average_precisions = []
    if metric == 'bbox' or metric == 'segm':
        beginning = line
    elif metric == 'f1':
        beginning = 'F1 best score = '
    elif metric == 'mae':
        beginning = 'MAE best score = '
    elif metric == 'mae%':
        beginning = 'Relative MAE best score = '
    else:
        raise RuntimeError()

    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def collect_f1_thres(path):
    beginning = 'F1 conf thres = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                return float(line.replace(beginning, ''))


def collect_training_time_stats(dataset_log):
    
    if not os.path.isfile(dataset_log):
        print(f"{dataset_log} wasn't found")

        return None

    first_line, last_line = '', ''
    last_epoch, max_epoch = 0, 0
    with open(dataset_log, 'r') as log_file:
        for line in log_file:
            if line.startswith('2022-'):
                # line = line[:19]
                if first_line == '':
                    first_line = line
                else:
                    last_line = line
            if line.startswith("[ INFO ] workflow: [('train', 1)], max: "):
                max_epoch = int(line.replace("[ INFO ] workflow: [('train', 1)], max: ", "").replace(" epochs", ""))
            if line.startswith('[ INFO ] Epoch ['):
                last_epoch = line.replace('[ INFO ] Epoch [', '')


    last_epoch = int(last_epoch.split(']')[0])
    FMT = '%Y-%m-%d %H:%M:%S'
    tdelta = (datetime.strptime(last_line, FMT) -
              datetime.strptime(first_line, FMT)).total_seconds() / 60
    return tdelta, last_epoch, max_epoch


# def get_command_eval_line(subset, dataset, work_dir, data_root, metric='bbox', update_nms=False):
    """Returns a command line for evaluation.
    Args:
        subset: train/val/test
        dataset: dataset dictionary including annotation prefix, classes, etc
        work_dir:
        data_root:
        metric: available options -- 'bbox', 'f1', 'mae', 'segm'
    Returns:
    """
    dataset_folder = osp.join(work_dir, dataset['name'])

    if metric == 'mae%':
        metric = 'mae'

    subset_metric = f"{subset}" if metric == 'bbox' else f"{subset}_{metric}"
    if osp.exists(f"{dataset_folder}/{subset_metric}"):
        print(
            f'Skip evaluation as it exist already: {dataset_folder}/{subset_metric}')
        return ''

    if not osp.exists(dataset_folder):
        print(f'get_command_eval_line: {dataset_folder} does not exist')
        return ''

    if ('tiling_ds' in dataset) and (dataset['tiling_ds']):
        cfg_path = osp.join(dataset_folder, 'model_tiling.py')
    else:
        cfg_path = osp.join(dataset_folder, 'model.py')
    # cfg = Config.fromfile(cfg_path)

    if len(glob.glob(osp.join(dataset_folder, 'best_*.pth'))):
        ckpt_path = glob.glob(osp.join(dataset_folder, 'best_*.pth'))[0]
    else:
        ckpt_path = osp.join(dataset_folder, 'latest.pth')

    if not osp.exists(ckpt_path):
        # best model wildcard search
        models = glob.glob(osp.join(dataset_folder, '*.pth'))
        if len(models):
            ckpt_path = sorted(models, key=os.path.getmtime)[-1]
        else:
            return ''

    dataset_flag = '.dataset' if 'dataset' in cfg.data.test else ''
    split_update_config = f'--cfg-options ' \
                          f'data.test{dataset_flag}.ann_file={osp.join(data_root, dataset[f"{subset}-ann-file"])} ' \
                          f'data.test{dataset_flag}.img_prefix={osp.join(data_root, dataset[f"{subset}-data-root"])} '

    if update_nms:
        split_update_config += update_test_cfg()

    # avoid time-concuming validation on test part which is equal to val part
    if subset == 'test' and dataset['name'] in [
            'kbts_fish', 'pcd', 'diopsis', 'vitens-tiled', 'wgisd1', 'wgisd5',
            'weed']:
        if metric != 'bbox':
            return f'cp {work_dir}/{dataset["name"]}/val_{metric} {work_dir}/{dataset["name"]}/test_{metric}'
        return f'cp {work_dir}/{dataset["name"]}/val {work_dir}/{dataset["name"]}/test'

    if metric != 'bbox':
        return f'python tools/test.py {cfg_path} {ckpt_path} --out {dataset_folder}/{subset}.pkl ' \
               f'--eval {metric} {split_update_config} | tee {dataset_folder}/{subset}_{metric}'
    return f'python tools/test.py {cfg_path} {ckpt_path} --out {dataset_folder}/{subset}.pkl ' \
           f'--eval bbox {split_update_config} | tee {dataset_folder}/{subset}'


def collect_epoch(dataset_folder):

    def get_epoch(json_log):
        lines = []
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'mode' not in log:
                    continue
                lines.append(log['epoch'])
        return lines

    json_files = glob.glob(os.path.join(dataset_folder, '*.json'))
    if len(json_files):
        json_path = sorted(json_files, key=os.path.getmtime)[-1]
        epoch = str(get_epoch(json_path)[-1])
        return epoch
    return ''


def print_summarized_statistics(datasets, work_dir, cur_metric):
    names = []
    metrics = []

    for dataset in datasets:
        names.append(dataset['name'])
        dataset_log = f'{work_dir}/results/{dataset["name"]}.log'
        metrics.append('')
        # try:
            # trained_epoch, max_epoch = vb(f'{dataset_work_dir}')
            # metrics.append(epoch)
        training_time, trained_epoch, max_epoch = collect_training_time_stats(dataset_log)
        metrics.append(f'{training_time:.0f}')
        metrics.append(f'{trained_epoch:.0f}')
        metrics.append(f'{max_epoch:.0f}')
        # except Exception as e:
            # metrics.append('')

    print(work_dir)
    print(','.join(names))
    print(','.join(metrics))


# def gen_update_cmd(cfg, dataset):
#     train_dataset_flag = '.dataset' if 'dataset' in cfg.data.train else ''
#     val_dataset_flag = '.dataset' if 'dataset' in cfg.data.val else ''
#     classes = dataset['classes'] if 'classes' in dataset else None
#     label_config = update_labels(cfg, classes)
#     update_config = f'--cfg-options ' \
#         f'{label_config} ' \
#         f'data.samples_per_gpu={args.batch_size} ' \
#         f'data.train{train_dataset_flag}.ann_file={osp.join(args.data_root, dataset["train-ann-file"])} ' \
#         f'data.train{train_dataset_flag}.img_prefix={osp.join(args.data_root, dataset["train-data-root"])} ' \
#         f'data.val{val_dataset_flag}.ann_file={osp.join(args.data_root, dataset["val-ann-file"])} ' \
#         f'data.val{val_dataset_flag}.img_prefix={osp.join(args.data_root, dataset["val-data-root"])} ' \
#         f'data.test{val_dataset_flag}.ann_file={osp.join(args.data_root, dataset["test-ann-file"])} ' \
#         f'data.test{val_dataset_flag}.img_prefix={osp.join(args.data_root, dataset["test-data-root"])} '

#     if cfg.load_from and osp.exists(args.pretrained_root):
#         load_from = osp.join(args.pretrained_root, osp.basename(cfg.load_from))
#         update_config += f'load_from={load_from} '

#     return update_config


def train_datasets(args, datasets, skip=None):
    metric = args.metric
    config_path = 'model.py'
    template_path = osp.join(args.work_dir, [file for file in os.listdir(args.work_dir) if file.endswith('.yaml')][0])
    for dataset in datasets:
        if dataset['name'] in skip:
            continue

        log_dir = osp.join(args.work_dir, 'results', dataset['name'])


        # activate_cmd = f' . det_env/bin/activate ' 
        FMT = "%Y-%m-%d %H:%M:%S"
        start_time = f' echo "Start time" ; date +"{FMT}"'

        end_time = f'echo "End time" ; date +"{FMT}"'

        train_cmd = f' ote train {template_path} --train-ann-files {dataset["train-ann-files"]} --train-data-roots {dataset["train-data-roots"]} '\
            f'--val-ann-files {dataset["val-ann-files"]} --val-data-roots {dataset["val-data-roots"]} '\
            f'--save-model-to {log_dir} '

        to_log_file = f' >> {log_dir}.log '


        print(f'{start_time}  {to_log_file} && {train_cmd}  {to_log_file} && {end_time} {to_log_file} ')
        create_dir = f'mkdir -p {osp.join(args.work_dir, "results")}'
        run(f'{create_dir} && {start_time}  {to_log_file} && {train_cmd}  {to_log_file} && {end_time} {to_log_file} ', shell=True, check=True)

    print_summarized_statistics(datasets, args.work_dir, metric)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run train script multiple times to get the best model')
    parser.add_argument('--work-dir', help='work dir path')
    parser.add_argument('--data-root', type=str)
    parser.add_argument(
        '--data-cfg',
        type=str,
        default='experiments/dataset_cfg/sc_datasets_cfg.py')

    subparsers = parser.add_subparsers(dest='task', help='task parser')
    parser_plt = subparsers.add_parser('train', help='parser for training')
    parser_plt.add_argument('--val-only')
    parser_plt.add_argument('--update-nms', action='store_true')
    parser_plt.add_argument('--gpus', type=int, default=1)
    parser_plt.add_argument('--batch-size', type=int, default=6)
    parser_plt.add_argument(
        '--pretrained-root', type=str, default='/home/yuchunli/_MODELS/mmdet')
    parser_plt.add_argument(
        '--metric', choices=['bbox', 'segm', 'f1', 'mae', 'mae%'], default='bbox')

    parser_plt = subparsers.add_parser(
        'export_images',
        help='parser for generating predicted images with ground-truth')
    parser_plt.add_argument('--font-size', type=int, default=13)
    parser_plt.add_argument('--border-width', type=int, default=2)
    parser_plt.add_argument('--score-thres', type=int, default=0.1)
    parser_plt.add_argument('--save-all', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sc_datasets = load_sc_dataset_cfg(args.data_cfg)
    if args.task == 'train':
        train_datasets(args, sc_datasets, skip=('weed', 'diopsis', )) 
        # 'bbcd', 'pothole', 'wildfire', 'aerial', 'dice', 'minneapple', 'wgisd1', 'wgisd5', 'uno'))
        # 'bbcd', 'pothole', 'wildfire', 'aerial', 'dice', 'minneapple', 'wgisd1', 'wgisd5',
    else:
        print_summarized_statistics(sc_datasets, args.work_dir, '')


