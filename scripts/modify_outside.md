- otx/algorithms/classification/venv/lib/python3.8/site-packages/mpa/cls/trainer.py L120  
    To disenable validation

    ```python
    # Save config
    # cfg.dump(osp.join(cfg.work_dir, 'config.yaml')) # FIXME bug to save
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # TODO (sungchul): check other condition for vaildation
    validate = False if cfg.model.type in ['BYOL'] else True
    if distributed:
        os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
        os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')

        mp.spawn(ClsTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                    args=(datasets, cfg, distributed, validate, timestamp, meta))
    else:
        ClsTrainer.train_worker(None, datasets, cfg,
                                distributed,
                                validate,
                                timestamp,
                                meta)
    ```

- otx/algorithms/classification/venv/lib/python3.8/site-packages/recipes/stages/_base_/data/pipelines/semisl_pipeline.py L3
    To use cifar/svhn, changing target_size 224 -> 32 is required.

    ```python
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    __resize_target_size = 32
    ```