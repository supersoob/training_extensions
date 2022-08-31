from ote.backends.torch.task import TorchTask
from ote.core.config import Config


class MMTask(TorchTask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)

    @staticmethod
    def configure_hook(cfg, **kwargs):
        """Update cfg.custom_hooks based on cfg.custom_hook_options
        """
        def update_hook(opt, custom_hooks, idx, hook):
            """Delete of update a custom hook
            """
            if isinstance(opt, dict):
                if opt.get('_delete_', False):
                    # if option include _delete_=True, remove this hook from custom_hooks
                    logger.info(f"configure_hook: {hook['type']} is deleted")
                    del custom_hooks[idx]
                else:
                    logger.info(f"configure_hook: {hook['type']} is updated with {opt}")
                    hook.update(**opt)

        custom_hook_options = cfg.pop('custom_hook_options', {})
        logger.info(f"configure_hook() {cfg.get('custom_hooks', [])} <- {custom_hook_options}")
        custom_hooks = cfg.get('custom_hooks', [])
        for idx, hook in enumerate(custom_hooks):
            for opt_key, opt in custom_hook_options.items():
                if hook['type'] == opt_key:
                    update_hook(opt, custom_hooks, idx, hook)

    @staticmethod
    def get_model_meta(cfg):
        ckpt_path = cfg.get('load_from', None)
        meta = {}
        if ckpt_path:
            ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location='cpu')
            meta = ckpt.get('meta', {})
        return meta

    @staticmethod
    def get_train_data_cfg(cfg):
        if 'dataset' in cfg.data.train:  # Concat|RepeatDataset
            dataset = cfg.data.train.dataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            return dataset
        else:
            return cfg.data.train

    @staticmethod
    def get_data_classes(cfg):
        data_classes = []
        train_cfg = MMTask.get_train_data_cfg(cfg)
        if 'data_classes' in train_cfg:
            data_classes = list(train_cfg.pop('data_classes', []))
        elif 'classes' in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_model_classes(cfg):
        """Extract trained classes info from checkpoint file.
        MMCV-based models would save class info in ckpt['meta']['CLASSES']
        For other cases, try to get the info from cfg.model.classes (with pop())
        - Which means that model classes should be specified in model-cfg for
          non-MMCV models (e.g. OMZ models)
        """
        classes = []
        meta = MMTask.get_model_meta(cfg)
        # for MPA classification legacy compatibility
        classes = meta.get('CLASSES', [])
        classes = meta.get('classes', classes)

        if len(classes) == 0:
            ckpt_path = cfg.get('load_from', None)
            if ckpt_path:
                classes = MMTask.read_label_schema(ckpt_path)
        if len(classes) == 0:
            classes = cfg.model.pop('classes', cfg.pop('model_classes', []))
        return classes

    @staticmethod
    def get_model_ckpt(ckpt_path, new_path=None):
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
            if not new_path:
                new_path = ckpt_path[:-3] + 'converted.pth'
            torch.save(ckpt, new_path)
            return new_path
        else:
            return ckpt_path

    @staticmethod
    def read_label_schema(ckpt_path, name_only=True, file_name='label_schema.json'):
        serialized_label_schema = []
        if any(ckpt_path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
            label_schema_path = osp.join(osp.dirname(ckpt_path), file_name)
            if osp.exists(label_schema_path):
                with open(label_schema_path, encoding="UTF-8") as read_file:
                    serialized_label_schema = json.load(read_file)
        if serialized_label_schema:
            if name_only:
                all_classes = [labels['name'] for labels in serialized_label_schema['all_labels'].values()]
            else:
                all_classes = serialized_label_schema
        else:
            all_classes = []
        return all_classes

    @staticmethod
    def set_inference_progress_callback(model, cfg):
        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get('custom_hooks', None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == 'OTEProgressHook']
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:
            def pre_hook(module, input):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, input, output):
                time_monitor.on_test_batch_end(None, None)
            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

    @staticmethod
    def register_checkpoint_hook(checkpoint_config):
        if checkpoint_config.get('type', False):
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        return hook

from mmcv.parallel import MMDataParallel

# Temporary copy from mmdet
# TODO: refactor
from mmcv.parallel.data_container import DataContainer


def scatter_cpu(inputs):
    """Scatter inputs to cpu.
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, DataContainer):
            return obj.data
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


class MMDataCPU(MMDataParallel):
    """Implementation of MMDataParallel to use CPU for training"""

    def scatter(self, inputs, kwargs):
        return TorchTask.scatter_kwargs(inputs, kwargs)

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.val_step(*inputs[0], **kwargs[0])

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module(*inputs[0], **kwargs[0])