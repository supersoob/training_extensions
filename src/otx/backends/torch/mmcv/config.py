from mmcv.utils.config import ConfigDict

from ote.core.config import Config


def convert_config(ote_config):
    return ote_config


def update_or_add_custom_hook(cfg: Config, hook_cfg: ConfigDict):
    """Update hook cfg if same type is in custom_hook or append it
    """
    custom_hooks = cfg.get('custom_hooks', [])
    custom_hooks_updated = False
    for custom_hook in custom_hooks:
        if custom_hook['type'] == hook_cfg['type']:
            custom_hook.update(hook_cfg)
            custom_hooks_updated = True
            break
    if not custom_hooks_updated:
        custom_hooks.append(hook_cfg)
    cfg['custom_hooks'] = custom_hooks