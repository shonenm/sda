#!/usr/bin/env python

"""
IBPM円柱周り流れの時系列データに対する拡散モデルの学習スクリプト

Immersed Boundary Projection Method (IBPM)で計算されたRe=100の円柱周り流れ
（カルマン渦列）のデータから2D速度場の時系列を生成するモデルを学習

- 幾何条件（円柱マスク + 流入プロファイル）を条件付け
- VPSDE.loss()による標準的なデノイジングスコアマッチング
- reflect padding（非周期境界条件）
"""

import torch
import wandb

from dawgz import job, schedule
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import trange

from sda.data import IBPMDataset
from sda.score import VPSDE
from sda.utils import save_config, slack_on_complete

# Import from experiments.ibpm.utils with absolute path
from experiments.ibpm.utils import make_score, PATH


# データパス設定
import os
if 'SCRATCH' in os.environ:
    DATA_PATH = Path(os.environ['SCRATCH']) / 'sda/ibpm'
else:
    DATA_PATH = Path('/workspace/data/ibpm_h5_wide_perturbed')

# 学習設定
# IBPM円柱流れ（399×199グリッド）の時系列を処理
CONFIG = {
    # アーキテクチャ
    'window': 16,                         # 時間窓のサイズ（データセットに合わせる）
    'cond_channels': 2,                   # 条件チャネル数（mask + inflow）
    'embedding': 64,                      # 時刻埋め込みの次元数
    'hidden_channels': (64, 128, 256),    # U-Netの各深さでのチャネル数
    'hidden_blocks': (2, 2, 2),           # 各深さでの残差ブロック数
    'kernel_size': 3,                     # 畳み込みカーネルのサイズ
    'activation': 'SiLU',                 # 活性化関数
    # 学習設定
    'epochs': 2000,                       # エポック数（大規模学習）
    'batch_size': 2,                      # バッチサイズ（高解像度のためメモリ節約）
    'optimizer': 'AdamW',                 # オプティマイザ
    'learning_rate': 1e-4,                # 学習率
    'weight_decay': 1e-3,                 # 重み減衰
    'scheduler': 'cosine',                # 学習率スケジューラ（長時間学習向け）
}


@job(array=1, cpus=4, gpus=1, ram='16GB', time='72:00:00')
@slack_on_complete(success_msg="🎉 IBPM Training completed!")
def train(i: int):
    """IBPM円柱流れモデルの学習ジョブ

    Re=100の円柱周り流れ（カルマン渦列）のシミュレーションデータから
    時系列生成モデルを学習。幾何条件（円柱マスク＋流入）を明示的に条件付け

    Args:
        i: ジョブ配列のインデックス
    """
    import math

    # WandB実行名を生成
    lr = CONFIG['learning_rate']
    bs = CONFIG['batch_size']
    wd = CONFIG['weight_decay']
    window = CONFIG['window']
    run_name = f"ibpm_vpsde_w{window}_lr{lr:.0e}_bs{bs}_wd{wd:.0e}_seed{i}"

    # WandBで実験管理
    run = wandb.init(
        project='sda-ibpm',
        name=run_name,
        group='ibpm_cylinder_vpsde',
        tags=['ibpm', 'cylinder', 'vpsde', f'seed{i}', f'lr{lr:.0e}'],
        notes=f'IBPM with VPSDE (Kolmogorov-style), run {i+1}',
        config=CONFIG,
    )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # データセットの準備
    train_dataset = IBPMDataset(
        DATA_PATH / 'train.h5',
        time_window=window,
        use_sdf=False,  # 2チャネル（mask + inflow）
    )
    valid_dataset = IBPMDataset(
        DATA_PATH / 'valid.h5',
        time_window=window,
        use_sdf=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # データ形状を取得（最初のバッチから）
    sample_x, sample_cond, _ = train_dataset[0]
    T, C, H, W = sample_x.shape  # (window, 2, H, W)

    # ネットワークの構築
    score_net = make_score(**CONFIG).cuda()

    # VPSDEを構築（Kolmogorov流と同様）
    # 入力形状: (window * 2, H, W) - 時間とチャネルをフラット化
    # eta=0.01: デフォルトの0.001より大きくすることでサンプリングの数値安定性を向上
    shape = torch.Size((window * C, H, W))
    sde = VPSDE(score_net.kernel, shape=shape, eta=0.01).cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(
        sde.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    # スケジューラ（線形減衰）
    epochs = CONFIG['epochs']
    if CONFIG['scheduler'] == 'linear':
        lr_lambda = lambda t: 1 - (t / epochs)
    elif CONFIG['scheduler'] == 'cosine':
        lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    else:
        lr_lambda = lambda t: 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 学習ループ
    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train = []
        losses_valid = []

        # 訓練フェーズ
        sde.train()
        for batch in train_loader:
            x, cond, mask = batch  # x: (B, T, C, H, W), cond: (B, C_cond, H, W)

            # GPUに転送し、時間とチャネルをフラット化
            x = x.cuda().flatten(1, 2)  # (B, T*C, H, W)
            cond = cond.cuda()  # (B, C_cond, H, W)

            # VPSDE.loss()を使用（条件cを渡す）
            loss = sde.loss(x, c=cond)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            losses_train.append(loss.detach())

        # 検証フェーズ
        sde.eval()
        with torch.no_grad():
            for batch in valid_loader:
                x, cond, mask = batch
                x = x.cuda().flatten(1, 2)
                cond = cond.cuda()
                losses_valid.append(sde.loss(x, c=cond))

        # 統計情報
        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        current_lr = optimizer.param_groups[0]['lr']

        # WandBにログ
        run.log({
            'epoch': epoch + 1,
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': current_lr,
        })

        # プログレスバーに表示
        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=current_lr)

        # スケジューラ更新
        scheduler.step()

        # モデル保存（50エポックごと）
        if (epoch + 1) % 50 == 0:
            torch.save(
                score_net.state_dict(),
                runpath / f'state_epoch{epoch+1}.pth',
            )

    # 最終モデルの保存
    torch.save(
        score_net.state_dict(),
        runpath / 'state_final.pth',
    )

    run.finish()


if __name__ == '__main__':
    import os
    from sda.utils import load_env_for_slurm

    # .envから環境変数を読み込んでSLURM用にexport文を生成
    env_exports = load_env_for_slurm(['SLACK_WEBHOOK_URL', 'WANDB_API_KEY'])
    env_exports.append('export WANDB_SILENT=true')

    # SLURMバックエンドでジョブをスケジュール
    schedule(
        train, # type: ignore
        name='IBPM_Training',
        backend='slurm',
        export='ALL',
        interpreter='/workspace/sda/.venv/bin/python',  # 共有venv内のPythonを使用
        env=env_exports,
    )
