- otx/algorithms/detection/venv/lib/python3.8/site-packages/mmdet/datasets/custom.py L298  
    To obtain results with mAP

    ```python
    def evaluate(self,
                results,
                metric='mAP',
                logger=None,
                proposal_nums=(100, 300, 1000),
                #  iou_thr=0.5,
                iou_thr=np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True).tolist(),
                scale_ranges=None):
    ```

- otx/algorithms/detection/venv/lib/python3.8/site-packages/mpa/det/trainer.py L53  
    To check final cfgs

    ```python
    # # Work directory
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg_for_save = cfg.copy()
    for i, k in enumerate(cfg_for_save['custom_hooks']):
        if k['type'] == 'OTXProgressHook':
            cfg_for_save['custom_hooks'].pop(i)
            break
        
    from mmcv import Config
    cfg_for_save = Config(cfg_dict=cfg_for_save, filename=cfg.filename)
    cfg_for_save.dump(osp.join(osp.abspath(cfg.work_dir), 'hparams.py'))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    ```