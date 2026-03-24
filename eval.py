import argparse
import os
import torch.distributed as dist

from mmengine.config import Config
from mmengine.runner import Runner

import custom_datasets
import pearl_ovss


def parse_args():
    parser = argparse.ArgumentParser(description='PEARL evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--backbone', default='')
    parser.add_argument('--attn', default='pearl')
    parser.add_argument('--prop', default="off")
    parser.add_argument('--work-dir', default='',
                        help='work directory to save logs/checkpoints. '
                             'If empty, will be auto-set to ./work_dirs/<dataset_name>')
    parser.add_argument('--show-dir', default='',
                        help='directory to save visualization images; '
                             'if empty, disable visualization hook')
    parser.add_argument(
        '--launcher',
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help="Launch mode for multi-GPU inference. "
             "Use 'pytorch' together with torchrun --nproc_per_node=<N>."
    )

    args = parser.parse_args()
    return args


def visualization_hook(cfg, show_dir):
    if show_dir == '':
        cfg.default_hooks.pop('visualization', None)
        return

    if 'visualization' not in cfg.default_hooks:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks, '
            'see base_config.py'
        )
    else:
        hook = cfg.default_hooks['visualization']
        hook['draw'] = True

    visualizer = cfg.visualizer
    visualizer['save_dir'] = show_dir
    # visualizer['alpha'] = 1


def safe_set_arg(cfg, arg, name, func=lambda x: x):
    if arg != '':
        cfg.model[name] = func(arg)


def infer_dataset_name_from_config(config_path: str) -> str:
    base = os.path.basename(config_path)
    stem = os.path.splitext(base)[0]
    if stem.startswith('cfg_'):
        stem = stem[len('cfg_'):]
    return stem


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir == '' or args.work_dir is None:
        dataset_name = infer_dataset_name_from_config(args.config)
        cfg.work_dir = os.path.join('./work_dirs', dataset_name)
    else:
        cfg.work_dir = args.work_dir
    
    safe_set_arg(cfg, args.backbone, 'clip_path')
    safe_set_arg(cfg, args.attn, 'attn_strategy')
    safe_set_arg(cfg, args.prop, 'use_prop')

    # visualization_hook(cfg, args.show_dir)

    cfg.launcher = args.launcher

    runner = Runner.from_cfg(cfg)
    runner.test()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
