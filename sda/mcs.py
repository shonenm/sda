r"""Markov chains"""

import abc
import jax
import jax.numpy as jnp
import jax.random as rng
import numpy as np
import math
import random
import torch

try:
    import jax_cfd.base as cfd
except:
    pass

from torch import Tensor, Size
from torch.distributions import Normal, MultivariateNormal
from typing import *


class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    一次の時不変マルコフ連鎖を表す抽象基底クラス
    確率過程の遷移モデルを定義するための基本インターフェース

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0)

        初期状態の分布からサンプリング
        Args:
            shape: サンプルのバッチ形状
        Returns:
            初期状態のサンプル
        """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1})

        遷移確率に基づいて次の状態を生成
        Args:
            x: 現在の状態
        Returns:
            次の時刻の状態
        """

        pass

    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1})

        マルコフ連鎖の軌跡を生成
        Args:
            x: 初期状態
            length: 軌跡の長さ（ステップ数）
            last: Trueの場合は最終状態のみを返す
        Returns:
            軌跡全体または最終状態
        """

        if last:
            # 最終状態のみが必要な場合は、メモリ効率的に計算
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            # 全軌跡を保存
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class DampedSpring(MarkovChain):
    r"""Linearized dynamics of a mass attached to a spring, subject to wind and drag.

    バネに接続された質量の線形化された動力学モデル
    風と抵抗の影響を受けるシステムのシミュレーション
    """

    def __init__(self, dt: float = 0.01):
        super().__init__()

        # 初期状態の平均と共分散行列
        self.mu_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # [位置, 速度, 加速度, 風]
        self.Sigma_0 = torch.tensor([1.0, 1.0, 1.0, 1.0]).diag()

        # 状態遷移行列（線形システムの動力学）
        self.A = torch.tensor([
            [1.0, dt, dt**2 / 2, 0.0],  # 位置の更新
            [0.0, 1.0, dt, 0.0],         # 速度の更新
            [-0.5, -0.1, 0.0, 0.2],      # 加速度の更新（バネの復元力、抵抗、風の影響）
            [0.0, 0.0, 0.0, 0.99],       # 風の減衰
        ])
        self.b = torch.tensor([0.0, 0.0, 0.0, 0.0])  # バイアス項
        self.Sigma_x = torch.tensor([0.1, 0.1, 0.1, 1.0]).diag() * dt  # プロセスノイズの共分散

    def prior(self, shape: Size = ()) -> Tensor:
        # 初期状態を多変量正規分布からサンプリング
        return MultivariateNormal(self.mu_0, self.Sigma_0).sample(shape)

    def transition(self, x: Tensor) -> Tensor:
        # 線形遷移とガウシアンノイズによる次状態の生成
        return MultivariateNormal(x @ self.A.T + self.b, self.Sigma_x).sample()


class DiscreteODE(MarkovChain):
    r"""Discretized ordinary differential equation (ODE)

    常微分方程式を離散化したマルコフ連鎖
    連続時間の動力学システムを離散時間ステップで近似

    Wikipedia:
        https://wikipedia.org/wiki/Ordinary_differential_equation
    """

    def __init__(self, dt: float = 0.01, steps: int = 1):
        super().__init__()

        self.dt, self.steps = dt, steps  # 時間ステップと内部積分ステップ数

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r"""Performs a step of the fourth-order Runge-Kutta integration scheme.

        4次のルンゲ=クッタ法による数値積分
        常微分方程式の高精度な数値解法

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        # 4つの傾きを計算
        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        # 重み付き平均で次の状態を計算
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @abc.abstractmethod
    def f(self, x: Tensor) -> Tensor:
        r""" f(x) = \frac{dx}{dt}

        微分方程式の右辺（時間微分）を定義
        各具体的なシステムで実装される
        """

        pass

    def transition(self, x: Tensor) -> Tensor:
        # 複数のサブステップに分けて積分（数値安定性向上）
        for _ in range(self.steps):
            x = self.rk4(self.f, x, self.dt / self.steps)

        return x


class Lorenz63(DiscreteODE):
    r"""Lorenz 1963 dynamics

    ローレンツ63システム：カオス的挙動を示す3次元動力学系
    気象モデルの単純化として提案された有名なカオスシステム

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_system
    """

    def __init__(
        self,
        sigma: float = 10.0,  # プラントル数：流体の粘性と熱拡散の比 [9, 13]
        rho: float = 28.0,    # レイリー数：浮力と粘性の比 [28, 40]
        beta: float = 8 / 3,  # 幾何学的なパラメータ [1, 3]
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sigma, self.rho, self.beta = sigma, rho, beta

    def prior(self, shape: Size = ()) -> Tensor:
        # アトラクタ付近の初期分布から開始
        mu = torch.tensor([0.0, 0.0, 25.0])  # アトラクタの中心付近
        sigma = torch.tensor([
            [64.0, 50.0,  0.0],
            [50.0, 81.0,  0.0],
            [ 0.0,  0.0, 75.0],
        ])

        return MultivariateNormal(mu, sigma).sample(shape)

    def f(self, x: Tensor) -> Tensor:
        # ローレンツ方程式の時間微分
        return torch.stack((
            self.sigma * (x[..., 1] - x[..., 0]),                      # dx/dt
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],            # dy/dt
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],             # dz/dt
        ), dim=-1)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        # データを正規化（ニューラルネットワーク学習用）
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return (x - mu) / sigma

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        # 正規化を逆変換
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return mu + sigma * x


class NoisyLorenz63(Lorenz63):
    r"""Noisy Lorenz 1963 dynamics

    ノイズを含むローレンツ63システム
    決定論的なローレンツ系にガウシアンノイズを追加
    """

    def moments(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # 遷移の平均（決定論的な遷移）と標準偏差（時間ステップに依存）
        return super().transition(x), self.dt ** 0.5

    def transition(self, x: Tensor) -> Tensor:
        # 正規分布に基づく確率的遷移
        return Normal(*self.moments(x)).sample()

    def log_prob(self, x1: Tensor, x2: Tensor) -> Tensor:
        # 遷移確率の対数尤度（粒子フィルタなどで使用）
        return Normal(*self.moments(x1)).log_prob(x2).sum(dim=-1)


class Lorenz96(DiscreteODE):
    r"""Lorenz 1996 dynamics

    ローレンツ96モデル：大気の一般循環を模した周期境界条件を持つモデル
    任意の次元数nで動作し、空間的な相互作用を表現

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_96_model
    """

    def __init__(
        self,
        n: int = 32,       # 状態空間の次元数（緯度方向の格子点数）
        F: float = 16.0,   # 外部強制項（大気加熱を表現）
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n, self.F = n, F

    def prior(self, shape: Size = ()) -> Tensor:
        # ランダムな初期状態から開始
        return torch.randn(*shape, self.n)

    def f(self, x: Tensor) -> Tensor:
        # 周期的に隣接する要素をシフト
        x1, x2, x3 = [torch.roll(x, i, dims=-1) for i in (1, -2, -1)]

        # ローレンツ96の時間微分：移流項、散逸項、強制項
        return (x1 - x2) * x3 - x + self.F


class LotkaVolterra(DiscreteODE):
    r"""Lotka-Volterra dynamics

    ロトカ=ヴォルテラ方程式：捕食者-被食者モデル
    生態系における個体数の時間発展を記述

    Wikipedia:
        https://wikipedia.org/wiki/Lotka-Volterra_equations
    """

    def __init__(
        self,
        alpha: float = 1.0,   # 被食者の増殖率
        beta: float = 1.0,    # 捕食率
        delta: float = 1.0,   # 捕食者の増殖効率
        gamma: float = 1.0,   # 捕食者の死亡率
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.alpha, self.beta = alpha, beta
        self.delta, self.gamma = delta, gamma

    def prior(self, shape: Size = ()) -> Tensor:
        # [0, 1]の範囲でランダムな初期個体数
        return torch.rand(*shape, 2)

    def f(self, x: Tensor) -> Tensor:
        # 対数空間での動力学（数値安定性向上）
        return torch.stack((
            self.alpha - self.beta * x[..., 1].exp(),      # 被食者の増減
            self.delta * x[..., 0].exp() - self.gamma,     # 捕食者の増減
        ), dim=-1)


class KolmogorovFlow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    コルモゴロフ強制を伴う2次元流体力学シミュレーション
    ナビエ・ストークス方程式に基づく乱流シミュレーション
    JAXを使用した高速な数値計算

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,      # グリッドサイズ（解像度）
        dt: float = 0.01,     # 時間ステップ
        reynolds: int = 1e3,  # レイノルズ数（慣性力と粘性力の比）
    ):
        super().__init__()

        # 周期境界条件を持つ2次元グリッドの設定
        grid = cfd.grids.Grid(
            shape=(size, size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),  # [0, 2π] × [0, 2π]の領域
        )

        # 周期境界条件（流体がドメインの端で反対側に現れる。標(2π, y)の流体は(0, y)と同じ、(x, 2π)は(x, 0)と同じ）
        bc = cfd.boundaries.periodic_boundary_conditions(2)

        # コルモゴロフ強制：特定の波数での外部エネルギー注入
        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,     # 強制の強さ
            constant_wavenumber=4.0,    # エネルギー注入する波数
            linear_coefficient=-0.1,    # 線形減衰
            forcing_type='kolmogorov',
        )

        # 数値安定性できない最小時間ステップ計算（CFL条件: 時間ステップΔtの間に、流体が1グリッド以上移動してはいけない）
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )

        # 安定性を保つために必要なサブステップ数を計算
        if dt_min > dt:
            steps = 1
        else:
            steps = math.ceil(dt / dt_min)

        # 半陰的ナビエ・ストークス方程式のソルバー
        step = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps, # サブステップごとの時間ステップ
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=steps, # サブステップ数
        )

        def prior(key: rng.PRNGKey) -> jax.Array:
            # フィルタされた初期速度場を生成（物理的に妥当な初期条件）
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )

            return jnp.stack((u.data, v.data))  # (2, H, W): u成分とv成分

        def transition(uv: jax.Array) -> jax.Array:
            # 速度場を境界条件付きの変数にラップ
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )

            # 時間発展（ナビエ・ストークス方程式の数値積分）
            u, v = step((u, v))

            return jnp.stack((u.data, v.data))

        # JAXのJITコンパイルとベクトル化で高速化
        self._prior = jax.jit(jnp.vectorize(prior, signature='(K)->(C,H,W)'))
        self._transition = jax.jit(jnp.vectorize(transition, signature='(C,H,W)->(C,H,W)'))

    def prior(self, shape: Size = ()) -> Tensor:
        # ランダムシードを生成してJAXの乱数キーに変換
        seed = random.randrange(2**32)

        # バッチサイズ分の乱数キーを生成
        key = rng.PRNGKey(seed)
        keys = rng.split(key, Size(shape).numel())
        keys = keys.reshape(*shape, -1)

        # JAXで初期速度場を生成し、PyTorch Tensorに変換
        x = self._prior(keys)
        x = torch.tensor(np.asarray(x))

        return x

    def transition(self, x: Tensor) -> Tensor:
        # PyTorch TensorをJAX arrayに変換
        x = x.detach().cpu().numpy()
        # JAXで時間発展を計算
        x = self._transition(x)
        # 結果をPyTorch Tensorに戻す
        x = torch.tensor(np.asarray(x))

        return x

    @staticmethod
    def coarsen(x: Tensor, r: int = 2) -> Tensor:
        # 画像の粗視化（ダウンサンプリング）：r×rブロックの平均を取る
        *batch, h, w = x.shape

        x = x.reshape(*batch, h // r, r, w // r, r)
        x = x.mean(dim=(-3, -1))  # ブロック内で平均

        return x

    @staticmethod
    def upsample(x: Tensor, r: int = 2, mode: str = 'bilinear') -> Tensor:
        # 画像のアップサンプリング（周期境界条件を考慮）
        *batch, h, w = x.shape

        x = x.reshape(-1, 1, h, w)
        x = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode='circular')  # 周期境界でパディング
        x = torch.nn.functional.interpolate(x, scale_factor=(r, r), mode=mode)  # 補間
        x = x[..., r:-r, r:-r]  # パディング部分を削除
        x = x.reshape(*batch, r * h, r * w)

        return x

    @staticmethod
    def vorticity(x: Tensor) -> Tensor:
        # 速度場から渦度（vorticity）を計算：ω = ∂v/∂x - ∂u/∂y
        # 渦度は流体の回転の度合いを表す
        *batch, _, h, w = x.shape

        y = x.reshape(-1, 2, h, w)
        y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='circular')  # 周期境界

        # 速度成分の空間微分を計算
        du, = torch.gradient(y[:, 0], dim=-1)  # u成分のx方向微分
        dv, = torch.gradient(y[:, 1], dim=-2)  # v成分のy方向微分

        # 渦度 = dv/dx - du/dy (回転の測度)
        y = du - dv
        y = y[:, 1:-1, 1:-1]  # パディング部分を除去
        y = y.reshape(*batch, h, w)

        return y
