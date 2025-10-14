#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from .utils import *


CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 4096,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}


@job(array=3, cpus=4, gpus=1, ram='16GB', time='24:00:00')
def train(i: int):
    # Create descriptive run name
    lr = CONFIG['learning_rate']
    bs = CONFIG['batch_size']
    wd = CONFIG['weight_decay']
    window = CONFIG['window']
    run_name = f"kolmogorov_w{window}_lr{lr:.0e}_bs{bs}_wd{wd:.0e}_seed{i}"

    run = wandb.init(
        project='sda-kolmogorov',
        name=run_name,
        group='kolmogorov_flow_baseline',
        tags=['kolmogorov', 'periodic-bc', f'seed{i}', f'lr{lr:.0e}', f'window{window}'],
        notes=f'Kolmogorov flow with forcing, run {i+1}/3. Training with window={window}, lr={lr:.0e}',
        config=CONFIG,
    )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    window = CONFIG['window']
    score = make_score(**CONFIG)
    shape = torch.Size((window * 2, 64, 64))
    sde = VPSDE(score.kernel, shape=shape).cuda()

    # Data
    trainset = TrajectoryDataset(PATH / 'data/train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(PATH / 'data/valid.h5', window=window, flatten=True)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / 'state.pth',
    )

    # Evaluation
    x = sde.sample(torch.Size([2]), steps=64).cpu()
    x = x.unflatten(1, (-1, 2))
    w = KolmogorovFlow.vorticity(x)

    run.log({'samples': wandb.Image(draw(w))})
    run.finish()


if __name__ == '__main__':
    schedule(
        train, # type: ignore
        name='Training',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
