# Kolmogorov Flow vs IBPM 円柱流れ 比較ドキュメント

Score-based Data Assimilation (SDA) フレームワークにおける2つの実験の詳細比較

---

## Executive Summary

### 全項目比較表

| カテゴリ | 項目 | Kolmogorov | IBPM | 差異 |
|----------|------|------------|------|------|
| **データ** | 解像度 | 64×64 | 199×399 | IBPM 19倍大 |
| | 訓練サンプル | 819軌道 | 42サンプル | Kolmo 20倍多 |
| | 時系列長 | 64 steps | 16 steps | Kolmo 4倍長 |
| | 総ウィンドウ数 | ~52,000 | ~500 | **Kolmo 100倍** |
| | 値域 | [-6, 5] | [-0.8, 1.7] | - |
| | 平均/std | 0.0 / 0.97 | 0.5 / 0.54 | Kolmoはゼロ中心 |
| | Re数 | 1000 | 100 | Kolmo 10倍高 |
| | 境界条件 | 周期 | 非周期 | **重要** |
| **ネットワーク** | hidden_channels | (96,192,384) | (96,192,384) | 同一 |
| | hidden_blocks | (3,3,3) | (3,3,3) | 同一 |
| | padding_mode | **circular** | **reflect** | **重要** |
| | 条件チャネル | 1 (forcing) | 2 (mask+inflow) | - |
| | 条件の扱い | **c無視**(内部固定) | **c必須**(外部) | **最重要** |
| **学習** | epochs | **4096** | **100** | **Kolmo 41倍** |
| | batch_size | 32 | 4 | Kolmo 8倍 |
| | learning_rate | 2e-4 | 1e-4 | Kolmo 2倍 |
| | weight_decay | 1e-3 | 1e-3 | 同一 |
| **SDE** | type | VPSDE | VPSDE | 同一 |
| | alpha | cos | cos | 同一 |
| | shape | (10,64,64) | (10,199,399) | - |
| | 損失関数 | `sde.loss(x)` | `sde.loss(x,c=cond)` | 条件有無 |
| **サンプリング** | steps | 64-512 | 256 | - |
| | corrections | 0-1 | **1必須** | **重要** |
| | tau | 0.5-1.0 | 0.5 | - |
| | 条件渡し | 不要 | 必要 | **重要** |

### 最重要な5つの違い

| # | 違い | Kolmogorov | IBPM | 影響度 |
|---|------|------------|------|--------|
| 1 | **データ量** | 52,000 windows | 500 windows | ★★★ |
| 2 | **学習エポック** | 4,096 | 100 | ★★★ |
| 3 | **条件付け方式** | 内部forcing(c無視) | 外部条件(c必須) | ★★★ |
| 4 | **境界条件** | circular | reflect | ★★ |
| 5 | **corrections** | 0でも動作 | 1が必須 | ★★ |

### 実装ファイル対応

| 役割 | Kolmogorov | IBPM |
|------|------------|------|
| 学習 | `experiments/kolmogorov/train.py` | `experiments/ibpm/train.py` |
| Utils | `experiments/kolmogorov/utils.py` | `experiments/ibpm/utils.py` |
| 可視化 | `experiments/kolmogorov/figures.ipynb` | `experiments/ibpm/figures.ipynb` |
| Dataset | `sda/utils.py` (TrajectoryDataset) | `sda/data/ibpm_dataset.py` |

### IBPMの課題と推奨改善

| 優先度 | 課題 | 推奨改善 |
|--------|------|----------|
| ★★★ | データ量100倍不足 | データ拡張 or 追加シミュレーション |
| ★★★ | エポック41倍不足 | 100→300以上に増加 |
| ★★ | corrections=0で発散 | 常にcorrections=1を使用 |
| ★ | 非ゼロ中心データ | 正規化を検討 |

### 学習損失の推移比較

WandBから取得した実際の学習ログデータ:

**IBPM** (Run: `ibpm_w5_lr2e-04_bs32_wd1e-03_seed0`, 1024 epochs)

| Epoch | Train Loss | Valid Loss |
|-------|------------|------------|
| 0 | 1.2432 | 2.2983 |
| 10 | 0.8394 | 0.7674 |
| 50 | 0.1914 | 0.1081 |
| 100 | 0.0408 | 0.0370 |
| 200 | 0.0294 | 0.0306 |
| 500 | 0.0227 | 0.0192 |
| 1000 | 0.0191 | 0.0206 |
| **1023 (final)** | **0.0166** | **0.0506** |

**Kolmogorov** (Run: `treasured-durian-3`, 4096 epochs)

| Epoch | Train Loss | Valid Loss |
|-------|------------|------------|
| 0 | 0.8082 | 0.4119 |
| 10 | 0.0556 | 0.0513 |
| 50 | 0.0334 | 0.0208 |
| 100 | 0.0296 | 0.0225 |
| 500 | 0.0151 | 0.0159 |
| 1000 | 0.0125 | 0.0144 |
| 2000 | 0.0107 | 0.0091 |
| 4000 | 0.0066 | 0.0076 |
| **4095 (final)** | **0.0074** | **0.0059** |

**比較分析**:

| 観点 | Kolmogorov | IBPM | 考察 |
|------|------------|------|------|
| 最終損失 | 0.0059 | 0.0166 | Kolmoの方が2.8倍低い |
| 収束速度 | 10 epochで0.05台 | 100 epochで0.04台 | Kolmoが10倍速く収束 |
| 過学習兆候 | なし | あり (valid↑ at final) | IBPMはデータ量不足の可能性 |
| 学習安定性 | 安定 | 終盤でvalid loss上昇 | - |

---

## 1. 概要

### 1.1 両実験の目的

| 実験 | 目的 |
|------|------|
| **Kolmogorov** | 2D強制乱流の時系列生成とデータ同化 |
| **IBPM** | 円柱周り流れ（カルマン渦）の時系列生成とデータ同化 |

### 1.2 物理現象の違い

#### Kolmogorov Flow（強制2D乱流）
- **現象**: エネルギーが大スケールに逆カスケードする2D乱流
- **ドメイン**: [0, 2π] × [0, 2π] の周期領域（トーラス）
- **強制項**: sin(4x) パターンでエネルギー注入
- **特徴**: 統計的に定常な乱流状態

#### IBPM 円柱流れ（カルマン渦列）
- **現象**: 円柱後流に形成される周期的な渦放出
- **ドメイン**: x ∈ [-4, 12], y ∈ [-4, 4] の有界領域
- **境界**: 流入（左）、流出（右）、円柱（中央）
- **特徴**: ストローハル数 St ≈ 0.16 @ Re=100

---

## 2. データセット比較

### 2.1 基本仕様

| 項目 | Kolmogorov | IBPM |
|------|------------|------|
| **空間解像度** | 64 × 64 | 199 × 399 |
| **訓練サンプル数** | 819 軌道 | 42 サンプル |
| **時系列長** | 64 ステップ/軌道 | 16 ステップ/サンプル |
| **チャネル数** | 2 (u, v) | 2 (u, v) |
| **総ウィンドウ数** | ~52,000 | ~500 |

### 2.2 データ統計

| 統計量 | Kolmogorov | IBPM |
|--------|------------|------|
| **値域** | [-6.2, 5.5] | [-0.8, 1.7] |
| **平均値** | ≈ 0.0 | ≈ 0.5 |
| **標準偏差** | ≈ 0.97 | ≈ 0.54 |
| **正規化** | ゼロ中心 | 非ゼロ中心 |

### 2.3 物理パラメータ

| パラメータ | Kolmogorov | IBPM |
|------------|------------|------|
| **レイノルズ数** | Re = 1000 | Re = 100 |
| **境界条件** | 周期的 | 非周期（流入/流出） |
| **外力** | sin(4x) 強制 | なし（流入条件のみ） |
| **時間刻み** | dt = 0.2 | シミュレーション依存 |

### 2.4 データセットクラス実装

#### Kolmogorov: `TrajectoryDataset`
```python
# ファイル: sda/utils.py
class TrajectoryDataset(Dataset):
    def __init__(self, file, window=None, flatten=False):
        self.data = h5py.File(file)['x'][:]  # 全データをメモリロード

    def __getitem__(self, i):
        x = self.data[i]
        if self.window:
            start = randint(0, len(x) - self.window)  # ランダム抽出
            x = x[start:start+self.window]
        if self.flatten:
            x = x.flatten(0, 1)  # (T, C, H, W) → (T*C, H, W)
        return x, {}  # 条件なし
```

#### IBPM: `IBPMDataset`
```python
# ファイル: sda/data/ibpm_dataset.py
class IBPMDataset(Dataset):
    def __init__(self, h5_path, time_window=8, use_sdf=False):
        # 幾何条件を初期化時に生成
        self.cylinder_mask = build_cylinder_mask(H, W, center, radius)
        self.inflow_profile = build_inflow_profile(H, W, U=1.0)

    def __getitem__(self, idx):
        x = self.data[t:t+window, sample, ...]  # スライディングウィンドウ
        cond = torch.stack([self.cylinder_mask, self.inflow_profile])
        return x, cond, mask  # 条件付き
```

**主な違い**:
- Kolmogorov: ランダムウィンドウ抽出、条件なし
- IBPM: スライディングウィンドウ、幾何条件付き

---

## 3. ネットワーク構造比較

### 3.1 共通アーキテクチャ: MCScoreNet + LocalScoreUNet

```
MCScoreNet (order = window // 2)
├── unfold(): 時間窓を展開 [t-2, t-1, t, t+1, t+2]
├── kernel: LocalScoreUNet
│   └── ScoreUNet (U-Net + 時刻埋め込み)
└── fold(): 出力を時系列に再構成
```

### 3.2 U-Net設定比較

| パラメータ | Kolmogorov | IBPM |
|------------|------------|------|
| hidden_channels | (96, 192, 384) | (96, 192, 384) |
| hidden_blocks | (3, 3, 3) | (3, 3, 3) |
| embedding | 64 | 64 |
| kernel_size | 3 | 3 |
| activation | SiLU | SiLU |
| **padding_mode** | **circular** | **reflect** |
| 入力チャネル | 10 (5×2) | 10 (5×2) |
| **条件チャネル** | **1** | **2** |

### 3.3 条件付けの違い（重要）

#### Kolmogorov: 内部forcing（暗黙的条件付け）

```python
# experiments/kolmogorov/utils.py
class LocalScoreUNet(ScoreUNet):
    def __init__(self, channels, size=64, **kwargs):
        super().__init__(channels, 1, **kwargs)  # 条件1チャネル

        # sin(4x)強制項を固定で生成
        domain = 2 * np.pi / size * (np.arange(size) + 0.5)
        forcing = torch.sin(4 * domain).expand(1, size, size)
        self.register_buffer('forcing', forcing)

    def forward(self, x, t, c=None):
        # 外部条件cを無視し、内部forcingを使用
        return super().forward(x, t, self.forcing)
```

#### IBPM: 外部条件（明示的条件付け）

```python
# experiments/ibpm/utils.py
class LocalScoreUNet(ScoreUNet):
    def __init__(self, channels, cond_channels=2, **kwargs):
        super().__init__(channels, cond_channels, **kwargs)

    def forward(self, x, t, c):
        # 外部条件cを必須として使用
        return super().forward(x, t, c)
```

**影響**:
- Kolmogorov: サンプリング時に `c=None` で動作
- IBPM: サンプリング時に `c=cond` を正しく渡す必要あり

---

## 4. 学習設定比較

### 4.1 ハイパーパラメータ

| パラメータ | Kolmogorov | IBPM | 比率 |
|------------|------------|------|------|
| **epochs** | **4096** | **100** | **41倍** |
| batch_size | 32 | 4 | 8倍 |
| learning_rate | 2e-4 | 1e-4 | 2倍 |
| weight_decay | 1e-3 | 1e-3 | 同じ |
| scheduler | linear | linear | 同じ |
| optimizer | AdamW | AdamW | 同じ |

### 4.2 学習ループの違い

#### Kolmogorov
```python
# experiments/kolmogorov/train.py
trainset = TrajectoryDataset(path, window=5, flatten=True)
sde = VPSDE(score.kernel, shape=(10, 64, 64))

for epoch in range(4096):
    for x, _ in train_loader:
        x = x.cuda()
        loss = sde.loss(x)  # 条件なし
        loss.backward()
```

#### IBPM
```python
# experiments/ibpm/train.py
trainset = IBPMDataset(path, time_window=5)
sde = VPSDE(score.kernel, shape=(10, 199, 399))

for epoch in range(100):
    for x, cond, mask in train_loader:
        x = x.cuda().flatten(1, 2)
        cond = cond.cuda()
        loss = sde.loss(x, c=cond)  # 条件付き
        loss.backward()
```

### 4.3 実際の学習結果

| 指標 | Kolmogorov | IBPM |
|------|------------|------|
| 最終訓練損失 | ~0.01 (推定) | 0.00412 |
| 最終検証損失 | ~0.01 (推定) | 0.00416 |
| 学習時間 | 数時間〜1日 | ~3時間 |

---

## 5. SDE設定比較

### 5.1 VPSDE共通設定

```python
# sda/score.py
class VPSDE:
    def __init__(self, eps, shape, alpha='cos', eta=1e-3):
        self.eps = eps      # スコアネットワーク
        self.shape = shape  # データ形状

        # ノイズスケジュール: α(t) = cos(arccos(√η) * t)²
        if alpha == 'cos':
            self.alpha = lambda t: cos(acos(sqrt(eta)) * t) ** 2
```

### 5.2 形状設定

| 設定 | Kolmogorov | IBPM |
|------|------------|------|
| shape | (10, 64, 64) | (10, 199, 399) |
| 総パラメータ数 | 40,960 | 793,810 |
| メモリ使用量 | 低 | 高（約20倍） |

### 5.3 損失関数

両者とも標準的なDenoising Score Matching損失を使用：

```python
L = E_t[ ||ε_φ(x(t), t, c) - ε||² ]

# x(t) = μ(t)·x + σ(t)·ε  （前向き拡散）
# ε ~ N(0, I)  （真のノイズ）
```

---

## 6. サンプリング設定比較

### 6.1 パラメータ比較

| パラメータ | Kolmogorov | IBPM |
|------------|------------|------|
| steps | 64〜512 | 256 |
| **corrections** | **0〜1** | **1（必須）** |
| tau | 0.5〜1.0 | 0.5 |
| 条件渡し | 不要 | 必要 |

### 6.2 サンプリングコード

#### Kolmogorov
```python
# experiments/kolmogorov/figures.ipynb
sde = VPSDE(score, shape=(L, C, H, W))
x = sde.sample(steps=256, corrections=1, tau=0.5)  # c=None
```

#### IBPM
```python
# experiments/ibpm/figures.ipynb
sde = VPSDE(score.kernel, shape=(T*C, H, W))
x = sde.sample(torch.Size([1]), c=cond, steps=256, corrections=1)
```

### 6.3 Predictor-Correctorスキーム

```python
# VPSDE.sample()
for t in time[:-1]:
    # Predictor（決定論的）
    r = μ(t-dt) / μ(t)
    x = r*x + (σ(t-dt) - r*σ(t)) * ε_φ(x, t, c)

    # Corrector（Langevin MCMC）← correctionsが重要
    for _ in range(corrections):
        z ~ N(0, I)
        eps = ε_φ(x, t-dt, c)
        delta = tau / ||eps||²
        x = x - (delta*eps + sqrt(2*delta)*z) * σ(t-dt)
```

**IBPMでcorrections=1が必須な理由**:
- データ量が少なく、スコア推定の精度が低い
- Correctorステップで分布を修正する必要がある

---

## 7. 実装ファイル対応表

| 役割 | Kolmogorov | IBPM |
|------|------------|------|
| **学習スクリプト** | `experiments/kolmogorov/train.py` | `experiments/ibpm/train.py` |
| **ユーティリティ** | `experiments/kolmogorov/utils.py` | `experiments/ibpm/utils.py` |
| **可視化** | `experiments/kolmogorov/figures.ipynb` | `experiments/ibpm/figures.ipynb` |
| **データセット** | `sda/utils.py` | `sda/data/ibpm_dataset.py` |
| **共通モジュール** | `sda/score.py`, `sda/nn.py`, `sda/mcs.py` | 同左 |

---

## 8. 主要な差異のまとめ

### 8.1 最も重要な違い

| # | 項目 | Kolmogorov | IBPM | 影響 |
|---|------|------------|------|------|
| 1 | **データ量** | 52,000 windows | 500 windows | **100倍差** |
| 2 | **学習エポック** | 4,096 | 100 | **41倍差** |
| 3 | **境界条件** | circular | reflect | パディング方式 |
| 4 | **条件付け** | 暗黙的(forcing固定) | 明示的(外部条件) | サンプリング方式 |
| 5 | **corrections** | 0でも動作 | 1が必須 | サンプリング安定性 |

### 8.2 設計上の違い

```
Kolmogorov設計思想:
  - 大量データ × 長時間学習 = 高精度スコア推定
  - 周期境界で物理的に一貫
  - 強制項は固定（学習不要）

IBPM設計思想:
  - 限られたデータ × 短時間学習 = 実用的アプローチ
  - 非周期境界で複雑な幾何を表現
  - 幾何条件を明示的に条件付け
```

---

## 9. IBPMの課題と推奨改善

### 9.1 現状の課題

1. **データ量の不足**: 42サンプル × 16ステップ ≈ 500ウィンドウ
2. **学習エポックの不足**: 100エポックではスコア推定が不十分
3. **サンプリングの不安定性**: corrections=0では発散

### 9.2 推奨改善策

| 優先度 | 改善策 | 期待効果 |
|--------|--------|----------|
| 高 | エポック数を300〜500に増加 | スコア推定精度向上 |
| 高 | corrections=1を常に使用 | サンプリング安定化 |
| 中 | データ拡張（回転、反転など） | 汎化性能向上 |
| 中 | データ正規化（ゼロ中心化） | 学習安定化 |
| 低 | ネットワーク容量増加 | 表現力向上 |

### 9.3 Kolmogorovとの公平な比較のために

Kolmogorovと同等の条件にするには：
- エポック数: 100 → 4096（41倍）
- または データ量: 500 → 52,000（100倍）

---

## 10. 付録: コード参照

### 10.1 LocalScoreUNet比較

```python
# ============ Kolmogorov ============
# experiments/kolmogorov/utils.py:43-82
class LocalScoreUNet(ScoreUNet):
    def __init__(self, channels, size=64, **kwargs):
        super().__init__(channels, 1, **kwargs)
        domain = 2 * np.pi / size * (np.arange(size) + 0.5)
        forcing = torch.sin(4 * domain).expand(1, size, size)
        self.register_buffer('forcing', forcing)

    def forward(self, x, t, c=None):
        return super().forward(x, t, self.forcing)  # c無視

# ============ IBPM ============
# experiments/ibpm/utils.py:75-88
class LocalScoreUNet(ScoreUNet):
    def __init__(self, channels, cond_channels=2, **kwargs):
        super().__init__(channels, cond_channels, **kwargs)

    def forward(self, x, t, c):
        return super().forward(x, t, c)  # c必須
```

### 10.2 CONFIG比較

```python
# ============ Kolmogorov ============
# experiments/kolmogorov/train.py:25-40
CONFIG = {
    'window': 5,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    'epochs': 4096,
    'batch_size': 32,
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}

# ============ IBPM ============
# experiments/ibpm/train.py:39-55
CONFIG = {
    'window': 5,
    'cond_channels': 2,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    'epochs': 100,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}
```

---

*作成日: 2024-12-09*
*最終更新: 2024-12-09*
