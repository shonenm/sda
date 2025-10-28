#!/usr/bin/env python

"""
Lorenz 63システムのデータ同化評価スクリプト

学習済みスコアモデルを使用してデータ同化性能を評価
- 合成観測データの生成（低頻度/低ノイズ、高頻度/高ノイズ）
- 粒子フィルタによる真の事後分布の推定
- スコアベース同化手法の性能評価（補正ステップ数を変化）
- Earth Mover's Distance (EMD) による精度評価
"""

import h5py
import numpy as np

from dawgz import job, after, context, ensure, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


@ensure(lambda: (PATH / f'results/obs.h5').exists())
@job(cpus=1, ram='1GB', time='00:05:00')
def observations():
    """合成観測データの生成

    テストデータから2種類の観測シナリオを作成:
    - lo: 低頻度（8ステップごと）、低ノイズ（σ=0.05）、x座標のみ観測
    - hi: 高頻度（全時刻）、高ノイズ（σ=0.25）、x座標のみ観測
    """
    with h5py.File(PATH / 'data/test.h5', mode='r') as f:
        x = f['x'][:, :65]  # 最初の65時刻を使用

    # 低頻度・低ノイズ観測（8ステップごと、σ=0.05）
    y_lo = np.random.normal(x[:, ::8, :1], 0.05)
    # 高頻度・高ノイズ観測（全時刻、σ=0.25）
    y_hi = np.random.normal(x[:, :, :1], 0.25)

    with h5py.File(PATH / 'results/obs.h5', mode='w') as f:
        f.create_dataset('lo', data=y_lo)
        f.create_dataset('hi', data=y_hi)


jobs = []

# 評価対象のモデルリスト（異なるorder kのモデル）
# k: 時間窓のorder（前後k時刻を考慮）
for name, local in [
    ('polar-capybara-13_y1g6w4jm', True),  # k=1 (window=3)
    ('snowy-leaf-29_711r6as1', True),      # k=2 (window=5)
    ('ruby-serenity-42_nbhxlnf9', True),   # k=3 (window=7)
    ('light-moon-51_09a36gw8', True),      # k=4 (window=9)
    ('lilac-bush-61_7f0sioiw', False),     # k≈8 (グローバルモデル、window=32)
]:
    # 2種類の観測シナリオで評価
    for freq in ['lo', 'hi']:
        @after(observations)
        @context(name=name, local=local, freq=freq)
        @job(name=f'{name}_{freq}', array=64, cpus=2, gpus=1, ram='8GB', time='01:00:00')
        def evaluation(i: int):
            """データ同化性能の評価ジョブ

            各観測データに対して:
            1. 粒子フィルタで真の事後分布を推定（ベースライン）
            2. スコアモデルで事後分布を推定（補正ステップ数を変化）
            3. 各手法の精度を対数尤度とEMDで評価

            Args:
                i: 評価データのインデックス（0-63）
            """
            chain = make_chain()

            # 観測データの読み込み
            with h5py.File(PATH / 'results/obs.h5', mode='r') as f:
                y = torch.from_numpy(f[freq][i])

            # 観測演算子: 状態からx座標のみを抽出
            A = lambda x: chain.preprocess(x)[..., :1]

            # 観測シナリオのパラメータ
            if freq == 'lo':  # 低頻度・低ノイズ
                sigma, step = 0.05, 8
            else:             # 高頻度・高ノイズ
                sigma, step = 0.25, 1

            # 真の事後分布（粒子フィルタで推定、ベースライン）
            # 2回独立に実行してEMD計算用のリファレンスとする
            x = posterior(y, A=A, sigma=sigma, step=step)[:1024]
            x_ = posterior(y, A=A, sigma=sigma, step=step)[:1024]

            # 粒子フィルタの性能評価
            log_px = log_prior(x).mean().item()  # 事前分布との整合性
            log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()  # 観測との整合性
            w1 = emd(x, x_).item()  # 推定の安定性（2回の推定間のEMD）

            # 結果をCSVに記録
            with open(PATH / f'results/stats_{freq}.csv', mode='a') as f:
                f.write(f'{i},ground-truth,,{log_px},{log_py},{w1}\n')

            print('GT:', log_px, log_py, w1, flush=True)

            # スコアモデルによる事後分布推定
            score = load_score(PATH / f'runs/{name}/state.pth', local=local)
            # ガウス尤度でスコアを条件付ける
            sde = VPSDE(
                GaussianScore(
                    y=y,                        # 観測データ
                    A=lambda x: x[..., ::step, :1],  # 観測演算子
                    std=sigma,                  # 観測ノイズ
                    sde=VPSDE(score, shape=()),  # 事前分布のスコア
                    gamma=3e-2,                 # 尤度の重み
                ),
                shape=(65, 3),  # 65時刻 × 3次元状態
            ).cuda()

            # 補正ステップ数を変化させて評価
            for C in (0, 1, 2, 4, 8, 16):
                # 事後分布からサンプリング（補正ステップ数C）
                x = sde.sample((1024,), steps=256, corrections=C, tau=0.25).cpu()
                x = chain.postprocess(x)

                # 性能評価指標
                log_px = log_prior(x).mean().item()
                log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
                w1 = emd(x, x_).item()  # 真の事後分布（粒子フィルタ）とのEMD

                # 結果をCSVに記録
                with open(PATH / f'results/stats_{freq}.csv', mode='a') as f:
                    f.write(f'{i},{name},{C},{log_px},{log_py},{w1}\n')

                print(f'{C:02d}:', log_px, log_py, w1, flush=True)

        jobs.append(evaluation)


if __name__ == '__main__':
    # 結果ディレクトリの作成
    (PATH / 'results').mkdir(parents=True, exist_ok=True)

    # SLURMバックエンドで全評価ジョブをスケジュール
    # 5モデル × 2観測シナリオ × 64データ = 640ジョブ
    schedule(
        *jobs,
        name='Evaluation',
        backend='slurm',
        prune=True,   # 既に完了したジョブはスキップ
        export='ALL', # すべての環境変数をエクスポート
    )
