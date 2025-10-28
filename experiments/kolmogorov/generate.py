#!/usr/bin/env python

"""
Kolmogorov流の訓練データ生成スクリプト

SLURMジョブ配列を使って1024個の独立したシミュレーションを並列実行し、
その後HDF5形式で訓練/検証/テストデータセットに集約
"""

import h5py
import numpy as np
import random

from dawgz import job, after, ensure, schedule
from typing import *

from sda.mcs import KolmogorovFlow

from .utils import *


@ensure(lambda i: (PATH / f'data/x_{i:06d}.npy').exists())
@job(array=1024, cpus=1, ram='1GB', time='00:05:00')
def simulate(i: int):
    """Kolmogorov流の単一軌道をシミュレート

    事前分布からサンプリングした初期条件から128ステップの時系列を生成
    最初の64ステップはバーンイン期間として破棄し、残りの64ステップを保存

    Args:
        i: シミュレーション番号（0-1023）、乱数シードとしても使用
    """
    chain = make_chain()

    random.seed(i)  # 再現性のための乱数シード設定

    # 事前分布（ガウス分布）からサンプリング
    x = chain.prior()
    # 128ステップの時間発展（定常状態に到達させる）
    x = chain.trajectory(x, length=128)
    # バーンイン期間を除外（定常状態の64ステップのみ保存）
    x = x[64:]

    np.save(PATH / f'data/x_{i:06d}.npy', x)


@after(simulate) # type: ignore
@job(cpus=1, ram='1GB', time='00:15:00')
def aggregate():
    """シミュレーション結果を集約してHDF5データセットを作成

    1024個の.npyファイルを読み込み、空間解像度を256→64に粗視化し、
    訓練/検証/テストデータ（8:1:1の比率）に分割してHDF5形式で保存
    """
    files = sorted(PATH.glob('data/x_*.npy'))
    length = len(files)

    # データ分割: 訓練80%、検証10%、テスト10%
    i = int(0.8 * length)  # 819個
    j = int(0.9 * length)  # 922個

    splits = {
        'train': files[:i],
        'valid': files[i:j],
        'test': files[j:],
    }

    for name, files in splits.items():
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            # データセット作成: (サンプル数, 時間, チャネル, 高さ, 幅)
            dset = f.create_dataset(
                'x',
                shape=(len(files), 64, 2, 64, 64),  # 64時刻 × 2成分(u,v) × 64×64空間
                dtype=np.float32,
            )

            # 各ファイルを読み込んで粗視化し、データセットに書き込む
            for i, x in enumerate(map(np.load, files)):
                # 空間解像度を256×256から64×64に縮小（4×4平均プーリング）
                arr = KolmogorovFlow.coarsen(torch.from_numpy(x), 4) \
                                    .detach().cpu().numpy().astype(np.float32)
                dset[i, ...] = arr


if __name__ == '__main__':
    # データディレクトリの作成
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    # SLURMバックエンドでジョブをスケジュール
    # simulate: 1024個のジョブ配列（各5分）
    # aggregate: simulateの後に実行（15分）
    schedule(
        aggregate, # type: ignore
        name='Data generation',
        backend='slurm',
        prune=True,   # 既に完了したジョブはスキップ
        export='ALL', # すべての環境変数をエクスポート
    )
