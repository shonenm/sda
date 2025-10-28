r"""Neural networks"""

import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.nn import LayerNorm


class ResidualBlock(nn.Sequential):
    r"""Creates a residual block.

    残差ブロック：入力と出力を加算することで勾配消失問題を軽減
    出力を入力に直接足し戻す構造を「残差接続」と呼ぶ
    ResNetなどで使用される基本的なアーキテクチャ
    """

    def forward(self, x: Tensor) -> Tensor:
        # 残差接続: output = input + F(input)
        return x + super().forward(x)


class ModResidualBlock(nn.Module):
    r"""Creates a residual block with modulation.

    変調付き残差ブロック：条件付き生成モデルで使用
    変調とは、外部の条件信号（y）を使ってネットワーク内部の処理を調整すること
    外部信号yによって処理をモジュレート（調整）する
    """

    def __init__(self, project: nn.Module, residue: nn.Module):
        super().__init__()

        self.project = project  # 変調信号yの射影
        self.residue = residue  # 残差処理

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # 変調付き残差: output = x + residue(x + project(y))
        # yの情報をxに注入してから残差処理を適用
        return x + self.residue(x + self.project(y))


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    残差多層パーセプトロン：残差接続を持つ全結合ニューラルネットワーク
    各層で残差ブロックを使用することで、深いネットワークの学習を安定化

    Arguments:
        in_features: 入力特徴の次元数
        out_features: 出力特徴の次元数
        hidden_features: 隠れ層の特徴次元数のシーケンス
        activation: 活性化関数のコンストラクタ
        kwargs: :class:`nn.Linear`に渡されるキーワード引数
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        **kwargs,
    ):
        blocks = []

        # 各層間で次元変換と残差ブロックを配置
        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            # 次元が変わる場合は線形変換を追加
            if after != before:
                blocks.append(nn.Linear(before, after, **kwargs))

            # 残差ブロック: LayerNorm -> Linear -> Activation -> Linear
            blocks.append(
                ResidualBlock(
                    LayerNorm(),                          # 正規化
                    nn.Linear(after, after, **kwargs),    # 第1線形層
                    activation(),                         # 活性化関数
                    nn.Linear(after, after, **kwargs),    # 第2線形層
                )
            )

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features


class UNet(nn.Module):
    r"""Creates a U-Net with modulation.

    変調付きU-Net：エンコーダ・デコーダ構造を持つ畳み込みニューラルネットワーク
    医用画像セグメンテーションや画像生成タスクで広く使用される
    スキップ接続により高解像度情報を保持し、変調信号で生成を制御

    References:
        | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        | https://arxiv.org/abs/1505.04597

    Arguments:
        in_channels: 入力チャネル数
        out_channels: 出力チャネル数
        mod_features: 変調特徴の次元数（条件付け信号の次元）
        hidden_channels: 各深さでの隠れチャネル数
        hidden_blocks: 各深さでの残差ブロック数
        kernel_size: 畳み込みカーネルのサイズ
        stride: ダウンサンプリング畳み込みのストライド
        activation: 活性化関数のコンストラクタ
        spatial: 空間次元数（1, 2, 3のいずれか）
        kwargs: :class:`nn.Conv2d`に渡されるキーワード引数
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        hidden_blocks: Sequence[int] = (2, 3, 5),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        activation: Callable[[], nn.Module] = nn.ReLU,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

        # 空間次元に応じた畳み込み層を選択（1D, 2D, 3D）
        convolution = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }.get(spatial)

        # カーネルサイズとストライドを空間次元に合わせて調整
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * spatial

        if type(stride) is int:
            stride = [stride] * spatial

        # 畳み込みのパラメータを設定（same padding）
        kwargs.update(
            kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size],
        )

        # 変調付き残差ブロックを生成するヘルパー関数
        block = lambda channels: ModResidualBlock(
            project=nn.Sequential(
                # 変調信号を射影してチャネル次元に変換
                nn.Linear(mod_features, channels),
                nn.Unflatten(-1, (-1,) + (1,) * spatial),  # 空間次元を追加
            ),
            residue=nn.Sequential(
                # 残差処理パス
                LayerNorm(-(spatial + 1)),                # チャネル正規化
                convolution(channels, channels, **kwargs),
                activation(),
                convolution(channels, channels, **kwargs),
            ),
        )

        # U-Netの各レベルの層を構築
        heads, tails = [], []  # エンコーダの入口、デコーダの出口
        descent, ascent = [], []  # エンコーダ、デコーダの処理ブロック

        for i, blocks in enumerate(hidden_blocks):
            if i > 0:
                # 中間層：ダウンサンプリング（エンコーダ）
                heads.append(
                    nn.Sequential(
                        convolution(
                            hidden_channels[i - 1],
                            hidden_channels[i],
                            stride=stride,  # ストライド畳み込みで解像度を下げる
                            **kwargs,
                        ),
                    )
                )

                # アップサンプリング（デコーダ）
                tails.append(
                    nn.Sequential(
                        LayerNorm(-(spatial + 1)),
                        nn.Upsample(scale_factor=tuple(stride), mode='nearest'),  # 解像度を上げる
                        convolution(
                            hidden_channels[i],
                            hidden_channels[i - 1],
                            **kwargs,
                        ),
                    )
                )
            else:
                # 最初の層と最後の層
                heads.append(convolution(in_channels, hidden_channels[i], **kwargs))
                tails.append(convolution(hidden_channels[i], out_channels, **kwargs))

            # 各深さでの処理ブロック
            descent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))
            ascent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))

        # モジュールリストとして保存（デコーダは逆順）
        self.heads = nn.ModuleList(heads)
        self.tails = nn.ModuleList(reversed(tails))
        self.descent = nn.ModuleList(descent)
        self.ascent = nn.ModuleList(reversed(ascent))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # エンコーダ・デコーダの順伝播
        # x: 入力データ（画像など）
        # y: 変調信号（時刻、条件など）
        memory = []  # スキップ接続用の中間特徴を保存
                     # U-Net のようなEncoder-Decoder構造の間で、同じ階層の特徴を飛び越えて伝える接続
                     # エンコーダ側で得た「高解像度・局所的な情報」をデコーダ側でも使えるようにする

        # エンコーダパス：解像度を下げながら特徴抽出
        for head, blocks in zip(self.heads, self.descent):
            x = head(x)  # ダウンサンプリング

            for block in blocks:
                x = block(x, y)  # 変調付き処理

            memory.append(x)  # スキップ接続のために保存

        memory.pop()  # 最後の特徴はスキップ不要（ボトルネック）

        # デコーダパス：解像度を上げながら再構成
        for blocks, tail in zip(self.ascent, self.tails):
            for block in blocks:
                x = block(x, y)  # 変調付き処理

            if memory:
                # スキップ接続：エンコーダの対応する層からの特徴を加算
                x = tail(x) + memory.pop()
            else:
                x = tail(x)

        return x
