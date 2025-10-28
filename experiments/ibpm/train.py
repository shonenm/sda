#!/usr/bin/env python

"""
IBPM円柱周り流れの時系列データに対する拡散モデルの学習スクリプト

Immersed Boundary Projection Method (IBPM)で計算されたRe=100の円柱周り流れ
（カルマン渦列）のデータから2D速度場の時系列を生成するモデルを学習
"""

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from .utils import *


# 学習設定
# IBPM円柱流れ（64×64グリッド）の時系列を処理
CONFIG = {
    # アーキテクチャ
    'window': 5,                          # 時間窓のサイズ（前後2ステップ+現在）
    'embedding': 64,                      # 時刻埋め込みの次元数
    'hidden_channels': (96, 192, 384),    # U-Netの各深さでのチャネル数（3段階）
    'hidden_blocks': (3, 3, 3),           # 各深さでの残差ブロック数
    'kernel_size': 3,                     # 畳み込みカーネルのサイズ
    'activation': 'SiLU',                 # 活性化関数
    # 学習設定
    'epochs': 1024,                       # エポック数（Kolmogorovより短い）
    'batch_size': 32,                     # バッチサイズ
    'optimizer': 'AdamW',                 # オプティマイザ
    'learning_rate': 2e-4,                # 学習率
    'weight_decay': 1e-3,                 # 重み減衰
    'scheduler': 'linear',                # 学習率スケジューラ
}


@job(array=3, cpus=4, gpus=1, ram='16GB', time='24:00:00')
def train(i: int):
    """IBPM円柱流れモデルの学習ジョブ

    Re=100の円柱周り流れ（カルマン渦列）のシミュレーションデータから
    時系列生成モデルを学習。円柱は境界条件として扱い、周期的な渦放出をモデル化

    Args:
        i: ジョブ配列のインデックス（0-2、3回の独立実行）
    """
    # WandB実行名を生成（ハイパーパラメータを含む）
    lr = CONFIG['learning_rate']
    bs = CONFIG['batch_size']
    wd = CONFIG['weight_decay']
    window = CONFIG['window']
    run_name = f"ibpm_w{window}_lr{lr:.0e}_bs{bs}_wd{wd:.0e}_seed{i}"

    # WandBで実験管理（タグとメタデータ付き）
    run = wandb.init(
        project='sda-ibpm',
        name=run_name,
        group='ibpm_cylinder_flow',
        tags=['ibpm', 'cylinder', f'seed{i}', f'lr{lr:.0e}', f'window{window}'],
        notes=f'IBPM cylinder flow (Re=100), run {i+1}/3. Training with window={window}, lr={lr:.0e}',
        config=CONFIG,
    )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # ネットワークの構築
    window = CONFIG['window']
    score = make_score(**CONFIG)  # MCScoreNet + LocalScoreUNet
    # window * 2チャネル（5時刻×2成分=10チャネル）、64×64の2D画像
    shape = torch.Size((window * 2, 64, 64))
    sde = VPSDE(score.kernel, shape=shape).cuda()

    # データの準備（IBPMシミュレーション結果、flatten=Trueで時間×チャネルをフラット化）
    trainset = TrajectoryDataset(Path("/workspace/data/ibpm_h5/train.h5"), window=window, flatten=True)
    validset = TrajectoryDataset(Path("/workspace/data/ibpm_h5/valid.h5"), window=window, flatten=True)

    # 学習ループ
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG,
    )

    # エポックごとにログを記録
    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # モデルの保存
    torch.save(
        score.state_dict(),
        runpath / 'state.pth',
    )

    # 評価：サンプル生成と可視化
    x = sde.sample(torch.Size([2]), steps=64).cpu()  # 2つのサンプルを生成
    x = x.unflatten(1, (-1, 2))  # (2, 10, 64, 64) -> (2, 5, 2, 64, 64)
    w = KolmogorovFlow.vorticity(x)  # 速度場から渦度を計算（カルマン渦列の可視化）

    # 渦度場の画像をWandBにログ
    run.log({'samples': wandb.Image(draw(w))})
    run.finish()


if __name__ == '__main__':
    # SLURMバックエンドでジョブをスケジュール（3回の独立実行）
    schedule(
        train, # type: ignore
        name='Training',
        backend='slurm',
        export='ALL',                     # すべての環境変数をエクスポート
        env=['export WANDB_SILENT=true'], # WandBの出力を抑制
    )
