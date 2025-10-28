#!/usr/bin/env python

"""
Lorenz 63システムの時系列データに対する拡散モデルの学習スクリプト

グローバルモデル（U-Net）とローカルモデル（MLP）の2種類を訓練
3次元カオス力学系の時系列生成とデータ同化への応用
"""

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


# グローバルモデルの学習設定
# 時系列全体を一度に処理するU-Netベース
GLOBAL_CONFIG = {
    # アーキテクチャ
    'embedding': 32,           # 時刻埋め込みの次元数
    'hidden_channels': (64,),  # U-Netの隠れ層チャネル数
    'hidden_blocks': (3,),     # 各深さでの残差ブロック数
    'activation': 'SiLU',      # 活性化関数
    # 学習設定
    'epochs': 4096,            # エポック数（長期学習）
    'batch_size': 64,          # バッチサイズ（大きめ）
    'optimizer': 'AdamW',      # オプティマイザ
    'learning_rate': 1e-3,     # 学習率
    'weight_decay': 1e-3,      # 重み減衰
    'scheduler': 'linear',     # 学習率スケジューラ
}

# ローカルモデルの学習設定
# 時間窓内の局所依存のみを考慮するMLPベース
LOCAL_CONFIG = {
    # アーキテクチャ
    'window': 5,               # 時間窓のサイズ（前後2ステップ+現在）
    'embedding': 32,           # 時刻埋め込みの次元数
    'width': 256,              # 隠れ層の幅
    'depth': 5,                # 隠れ層の深さ
    'activation': 'SiLU',      # 活性化関数
    # 学習設定
    'epochs': 4096,            # エポック数
    'batch_size': 64,          # バッチサイズ
    'optimizer': 'AdamW',      # オプティマイザ
    'learning_rate': 1e-3,     # 学習率
    'weight_decay': 1e-3,      # 重み減衰
    'scheduler': 'linear',     # 学習率スケジューラ
}


@job(array=3, cpus=4, gpus=1, ram='8GB', time='06:00:00')
def train_global(i: int):
    """グローバルモデル（U-Net）の学習ジョブ

    時系列全体を一度に処理するU-Netベースのスコアネットワークを訓練
    長期依存関係を捉えるが、計算コストが高い

    Args:
        i: ジョブ配列のインデックス（0-2、3回の独立実行）
    """
    # WandBで実験管理
    run = wandb.init(project='sda-lorenz', group='global', config=GLOBAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(GLOBAL_CONFIG, runpath)

    # ネットワークの構築（32時刻 × 3次元状態）
    score = make_global_score(**GLOBAL_CONFIG)
    sde = VPSDE(score, shape=(32, 3)).cuda()

    # データの準備（32ステップの時間窓）
    trainset = TrajectoryDataset(PATH / 'data/train.h5', window=32)
    validset = TrajectoryDataset(PATH / 'data/valid.h5', window=32)

    # 学習ループ
    generator = loop(
        sde,
        trainset,
        validset,
        **GLOBAL_CONFIG,
        device='cuda',
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
        runpath / f'state.pth',
    )

    # 評価：サンプル生成と尤度計算
    chain = make_chain()

    # 1024個のサンプルを生成
    x = sde.sample((1024,), steps=64).cpu()
    x = chain.postprocess(x)  # 後処理（正規化の逆変換など）

    # 遷移確率の対数平均（モデルがLorenz力学をどれだけ学習したか）
    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


@job(array=3, cpus=4, gpus=1, ram='8GB', time='06:00:00')
def train_local(i: int):
    """ローカルモデル（MLP）の学習ジョブ

    時間窓内の局所依存のみを考慮するMLPベースのスコアネットワークを訓練
    グローバルモデルより軽量で、長い時系列に対応可能

    Args:
        i: ジョブ配列のインデックス（0-2、3回の独立実行）
    """
    # WandBで実験管理
    run = wandb.init(project='sda-lorenz', group='local', config=LOCAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(LOCAL_CONFIG, runpath)

    # ネットワークの構築（window時刻 × 3次元状態、フラット化）
    window = LOCAL_CONFIG['window']
    score = make_local_score(**LOCAL_CONFIG)
    sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()  # 5×3=15次元

    # データの準備（window時刻の時間窓、flatten=Trueで1次元化）
    trainset = TrajectoryDataset(PATH / 'data/train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(PATH / 'data/valid.h5', window=window, flatten=True)

    # 学習ループ
    generator = loop(
        sde,
        trainset,
        validset,
        **LOCAL_CONFIG,
        device='cuda',
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
        runpath / f'state.pth',
    )

    # 評価：サンプル生成と尤度計算
    chain = make_chain()

    # 4096個のサンプルを生成（ローカルモデルは軽量なのでより多く）
    x = sde.sample((4096,), steps=64).cpu()
    x = x.unflatten(-1, (-1, 3))  # (4096, 15) -> (4096, 5, 3) に復元
    x = chain.postprocess(x)

    # 遷移確率の対数平均
    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


if __name__ == '__main__':
    # SLURMバックエンドで両方のモデルタイプをスケジュール
    # train_global: グローバルモデル（U-Net）× 3回
    # train_local: ローカルモデル（MLP）× 3回
    schedule(
        train_global,
        train_local,
        name='Training',
        backend='slurm',
        export='ALL',                     # すべての環境変数をエクスポート
        env=['export WANDB_SILENT=true'], # WandBの出力を抑制
    )
