"""Pydantic-based configuration management for SDA.

型安全な設定管理：
- バリデーション付きの設定スキーマ
- 環境変数からの読み込みサポート
- ネストされた設定構造
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """スコアネットワークの設定"""

    # ScoreNet / ScoreUNet 共通
    embedding: int = Field(default=64, ge=1, description="時刻埋め込みの次元数")
    context: int = Field(default=0, ge=0, description="文脈特徴の次元数")

    # UNet固有
    channels: list[int] = Field(
        default=[64, 128, 256],
        min_length=1,
        description="UNetの各レベルのチャネル数",
    )
    kernel_size: int = Field(default=3, ge=1, description="畳み込みカーネルサイズ")
    spatial: int = Field(default=2, ge=0, description="空間次元数 (0=MLP, 2=2D画像)")

    # MCScoreNet固有
    order: int = Field(default=1, ge=1, description="マルコフ連鎖の次数")


class SDEConfig(BaseModel):
    """SDE (確率微分方程式) の設定"""

    alpha: Literal["lin", "cos", "exp"] = Field(
        default="cos",
        description="ノイズスケジュール: lin=線形, cos=コサイン, exp=指数",
    )
    eta: float = Field(
        default=1e-3,
        gt=0,
        description="数値安定性のための小さな定数",
    )


class SamplingConfig(BaseModel):
    """サンプリングの設定"""

    steps: int = Field(default=64, ge=1, description="離散時間ステップ数")
    corrections: int = Field(default=0, ge=0, description="Langevin補正回数")
    tau: float = Field(default=1.0, gt=0, description="Langevinステップの振幅")


class DataConfig(BaseModel):
    """データセットの設定"""

    window: int | None = Field(
        default=None,
        ge=1,
        description="時間窓のサイズ (Noneで全系列)",
    )
    train_ratio: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="訓練データの割合",
    )


class OptimizerConfig(BaseModel):
    """オプティマイザの設定"""

    name: Literal["adam", "adamw", "sgd"] = Field(
        default="adamw",
        description="オプティマイザの種類",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="学習率",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="重み減衰 (L2正則化)",
    )
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam/AdamWのbeta係数",
    )


class SchedulerConfig(BaseModel):
    """学習率スケジューラの設定"""

    name: Literal["linear", "cosine", "exponential", "none"] = Field(
        default="cosine",
        description="スケジューラの種類",
    )
    warmup_epochs: int = Field(
        default=0,
        ge=0,
        description="ウォームアップのエポック数",
    )


class TrainingConfig(BaseModel):
    """学習の設定"""

    epochs: int = Field(default=256, ge=1, description="エポック数")
    batch_size: int = Field(default=64, ge=1, description="バッチサイズ")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # ロギング
    log_interval: int = Field(default=10, ge=1, description="ログ出力間隔 (バッチ数)")
    save_interval: int = Field(default=50, ge=1, description="チェックポイント保存間隔 (エポック数)")

    # 早期終了
    early_stopping: bool = Field(default=False, description="早期終了を有効化")
    patience: int = Field(default=20, ge=1, description="早期終了の忍耐エポック数")


class WandbConfig(BaseModel):
    """Weights & Biases の設定"""

    enabled: bool = Field(default=True, description="W&Bを有効化")
    project: str = Field(default="sda", description="プロジェクト名")
    entity: str | None = Field(default=None, description="組織/ユーザー名")
    name: str | None = Field(default=None, description="実験名")
    tags: list[str] = Field(default_factory=list, description="タグ")


class ExperimentConfig(BaseModel):
    """実験全体の設定

    使用例:
        >>> from sda.config import ExperimentConfig
        >>> config = ExperimentConfig()
        >>> config.training.epochs
        256
        >>> config = ExperimentConfig(training={"epochs": 100, "batch_size": 32})
        >>> config.training.epochs
        100
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    sde: SDEConfig = Field(default_factory=SDEConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # 実験メタデータ
    seed: int = Field(default=42, description="乱数シード")
    device: str = Field(default="cuda", description="計算デバイス")
    debug: bool = Field(default=False, description="デバッグモード")

    model_config = {"extra": "forbid"}  # 未知のフィールドを禁止


# 環境変数からの読み込みをサポートするSettings
try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        """環境変数から読み込む設定

        環境変数の例:
            SDA_DEBUG=true
            SDA_TRAINING__EPOCHS=100
            SDA_WANDB__PROJECT=my_project

        使用例:
            >>> from sda.config import Settings
            >>> settings = Settings()
            >>> settings.debug
            False
        """

        debug: bool = Field(default=False, description="デバッグモード")
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
            default="INFO",
            description="ログレベル",
        )
        json_logs: bool = Field(default=False, description="JSON形式のログ出力")

        model_config = {
            "env_prefix": "SDA_",
            "env_nested_delimiter": "__",
        }

except ImportError:
    # pydantic-settingsがない場合はスキップ
    Settings = None  # type: ignore[misc, assignment]
