#!/usr/bin/env python

"""
Lorenz 63システムの訓練データ生成スクリプト

1024個の独立した軌道をシミュレートし、バーンイン後の長時系列を生成
訓練/検証/テストデータに分割してHDF5形式で保存
"""

import h5py

from dawgz import job, after, schedule
from typing import *

from utils import *


@job(cpus=1, ram='1GB', time='00:05:00')
def simulate():
    """Lorenz 63システムの軌道をシミュレート

    1024個の独立した軌道を生成し、定常状態の1024ステップを保存
    各軌道は異なる初期条件から開始し、カオス的挙動の多様性を確保
    """
    chain = make_chain()

    # 事前分布（ガウス分布）から1024個の初期条件をサンプリング
    x = chain.prior((1024,))
    # バーンイン期間：1024ステップで定常状態（アトラクタ）に到達
    x = chain.trajectory(x, length=1024, last=True)
    # メインのシミュレーション：1024ステップの時系列を生成
    x = chain.trajectory(x, length=1024)
    # 前処理（正規化など）
    x = chain.preprocess(x)
    # 転置: (1024軌道, 1024時刻, 3次元) -> (1024時刻, 1024軌道, 3次元)
    x = x.transpose(0, 1)

    # データ分割: 訓練80%、検証10%、テスト10%
    i = int(0.8 * len(x))  # 819時刻
    j = int(0.9 * len(x))  # 922時刻

    splits = {
        'train': x[:i],
        'valid': x[i:j],
        'test': x[j:],
    }

    # HDF5形式で保存
    for name, x in splits.items():
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            f.create_dataset('x', data=x, dtype=np.float32)


if __name__ == '__main__':
    # データディレクトリの作成
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    # SLURMバックエンドでジョブをスケジュール（単一ジョブ、5分）
    schedule(
        simulate,
        name='Data generation',
        backend='slurm',
        export='ALL',  # すべての環境変数をエクスポート
    )
