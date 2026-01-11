# IBPM サンプリング問題の調査報告書

## 概要

IBPM（Immersed Boundary Projection Method）円柱流れデータに対するScore-based Data Assimilation実験において、学習完了後のサンプリングが「真っ白」（構造なし）になる問題を調査した。

---

## 問題の経緯

### Phase 1: 初期学習（SimpleSDE）

**状況**:
- `train.py`でSimpleSDEを使用して100エポック学習
- 損失は収束したように見えた

**結果**:
- 無条件サンプリング → ノイズのみ
- スパース観測再構成 → ノイズのみ

**原因特定**:
- `train.py`は`SimpleSDE`（beta_min/beta_max方式）を使用
- `figures.ipynb`のサンプリングは`VPSDE`（alpha schedule方式）を使用
- **学習とサンプリングでSDE定義が異なっていた**

### Phase 2: VPSDE方式に修正

**修正内容** (`train.py`):
```python
# 修正前（問題あり）
sde = SimpleSDE(beta_min=0.1, beta_max=20.0)
loss = composite_loss(...)

# 修正後（Kolmogorov準拠）
from sda.score import VPSDE
sde = VPSDE(score_net.kernel, shape=(T*C, H, W))
loss = sde.loss(x, c=cond)
```

**学習結果**:
- 100エポック完了
- 最終損失: `lt=0.00412, lv=0.00416`

**結果**:
- 無条件サンプリング → **真っ白**
- スパース観測再構成 → **真っ白**

---

## 詳細調査

### 1. 学習とサンプリングのアーキテクチャ比較

| 項目 | 学習時 | サンプリング時 |
|------|--------|----------------|
| SDE | `VPSDE(score.kernel, shape=(T*C,H,W))` | `VPSDE(score, shape=(L,C,H,W))` |
| ネットワーク | `LocalScoreUNet` | `MCScoreNet` |
| データ形状 | `(B, T*C, H, W)` フラット化 | `(B, L, C, H, W)` 非フラット化 |
| 条件 | `c=cond` | `c=cond` |

**理論的検証**:
- MCScoreNetの`unfold`操作により、内部的にkernelは`(B, T*C, H, W)`を受け取る
- 形式は互換性あり

### 2. Kolmogorov flow vs IBPM の決定的な違い

**Kolmogorov** (`experiments/kolmogorov/utils.py`):
```python
class LocalScoreUNet(ScoreUNet):
    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)  # ← cを無視、内部forcingを使用
```

**IBPM** (`experiments/ibpm/utils.py`):
```python
class LocalScoreUNet(ScoreUNet):
    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        return super().forward(x, t, c)  # ← 外部cが必須
```

**重要な違い**:
- Kolmogorovは外部条件`c`を**無視**し、内部`self.forcing`を使用
- IBPMは外部条件`c`を**必須**とする
- Kolmogorovのサンプリングでは`c=None`でも動作するが、IBPMでは`c`を正しく渡す必要がある

### 3. 損失値の異常

VPSDE学習後の損失: `lt=0.00412`

**仮説**: モード崩壊（トリビアル解）
- ネットワークが`ε(x, t, c) ≈ 0`を常に出力
- 損失 = MSE(ε_pred, ε_true) が低くなる（ε_trueの平均は0）
- サンプリング時はノイズが除去されず、白いままになる

### 4. データ正規化の確認

**IBPMデータの統計**:
```
u-velocity: mean=0.500, std=0.540
v-velocity: mean=0.000, std=0.307
Data range: [-0.806, 1.697]
```

データはゼロ中心ではない（mean ≈ 0.5）が、これ自体は問題ではない。

---

## 検証用デバッグコード

`figures.ipynb` cell-22に追加したデバッグ出力:

```python
# ネットワーク出力の統計を確認
x_test = torch.randn(1, window, 2, H, W).cuda()
t_test = torch.tensor(0.5).cuda()
with torch.no_grad():
    eps_out = score(x_test, t_test, cond)
print(f"score output shape: {eps_out.shape}, nan={eps_out.isnan().any()}")
```

**期待値**:
- `std ≈ 1`（入力がガウシアンノイズなら）

**もし `std ≈ 0` なら**: モデルが崩壊している → 再学習が必要

---

## 考えられる原因と対策

### 原因候補

1. **モード崩壊**: ネットワークがほぼゼロを出力
2. **条件伝播の問題**: `c`がMCScoreNet経由で正しく伝わっていない
3. **学習データの正規化不足**: データがゼロ中心でない
4. **学習パラメータの問題**: 学習率、バッチサイズなど

### 対策案

#### 案1: score.kernelを直接使用してサンプリング

学習時と同じ形式でサンプリング:
```python
# 学習時と同じ: フラット化された形状でサンプリング
shape = torch.Size((window * 2, H, W))  # (10, 199, 399)
sde = VPSDE(score.kernel, shape=shape).cuda()

# サンプリング
x_flat = sde.sample(torch.Size([1]), c=cond, steps=64)  # (1, 10, 199, 399)

# unflatten して可視化用に変換
x = x_flat.unflatten(1, (window, 2))  # (1, 5, 2, 199, 399)
```

#### 案2: データ正規化を追加して再学習

```python
# train.py で
data_mean = train_data.mean()
data_std = train_data.std()
x = (x - data_mean) / data_std

# サンプリング時に逆変換
x_sample = x_sample * data_std + data_mean
```

#### 案3: 学習パラメータの調整

- エポック数増加: 100 → 300
- バッチサイズ増加: 4 → 8
- 学習率調整: 1e-4 → 5e-5

---

## ファイル構成

| ファイル | 役割 | 状態 |
|---------|------|------|
| `experiments/ibpm/train.py` | 学習スクリプト | VPSDE方式に修正済み |
| `experiments/ibpm/figures.ipynb` | 可視化・サンプリング | VPSDEモデルロード対応済み |
| `experiments/ibpm/utils.py` | LocalScoreUNet定義 | 外部条件c必須 |
| `sda/score.py` | VPSDE, MCScoreNet定義 | ライブラリコード |

---

## 学習ログ

### VPSDE学習（最新）

- Run ID: `ibpm_vpsde_w5_lr1e-04_bs4_wd1e-03_seed0_63l6siyf`
- エポック: 100
- 最終損失: `lt=0.00412, lv=0.00416`
- 所要時間: 約2時間43分

```
100%|██████████████| 100/100 [2:43:09<00:00, 97.90s/it, lr=1e-6, lt=0.00412, lv=0.00416]
```

---

## 次のステップ

1. **デバッグ出力の確認**: `score.kernel`の出力統計を確認
2. **案1の実装**: `score.kernel`を直接使用してサンプリング
3. **必要に応じて再学習**: データ正規化を追加

---

## 参考: Kolmogorovのサンプリングパターン

`experiments/kolmogorov/figures.ipynb`より:

```python
sde = VPSDE(
    GaussianScore(
        y_star,
        A=A,
        std=0.1,
        sde=VPSDE(score, shape=()),  # MCScoreNet, shape=()
    ),
    shape=x_star.shape,  # (8, 2, 64, 64)
).cuda()

x = sde.sample(steps=256, corrections=1, tau=0.5)  # c=None（不要）
```

Kolmogorovでは`c`を渡さずにサンプリングしている（内部forcingを使用するため不要）。

---

---

## 追加調査結果 (2024-12-09)

### デバッグ実行結果

#### 1. ネットワーク出力統計

| テスト | 入力 | 出力 std | 判定 |
|--------|------|----------|------|
| MCScoreNet + ノイズ | std=1.0 | std=1.19 | ✅ 正常 |
| kernel + ノイズ | std=1.0 | std=1.19 | ✅ 正常 |
| kernel + 実データ | std=0.51 | std=0.09 | ⚠️ 低い |

**結論**: モデル自体は崩壊していない。ノイズ予測は正常。

#### 2. サンプリング結果

| 設定 | mean | std | range |
|------|------|-----|-------|
| steps=64 | 0.89 | 2.56 | [-271, 206] |
| steps=256 | 0.45 | 1.93 | [-118, 232] |
| **steps=256, corrections=1** | **0.51** | **0.52** | **[-0.28, 1.57]** |
| **元データ** | **0.50** | **0.51** | **[-0.74, 1.49]** |

**重要発見**: `corrections=1`が必須！Langevin補正なしではサンプリングが発散。

#### 3. チャネル別統計の問題

| チャネル | サンプル std | 訓練データ std | 比率 |
|----------|-------------|----------------|------|
| u (水平速度) | 0.0164 | 0.1604 | **10%** |
| v (垂直速度) | 0.0123 | 0.0670 | **18%** |

**問題**: サンプルの変動が訓練データの10-18%しかない。モデルは「平均場」を出力している。

#### 4. デノイジングテスト

**訓練データからのデノイジング** (t=0.5):
- u: denoised std=0.17, train std=0.16 ✅
- v: denoised std=0.09, train std=0.07 ✅

**純粋ノイズからのデノイジング** (t=0.5):
- u: denoised std=**0.12**, train std=0.16 (25%低い)
- v: denoised std=**0.11**, train std=0.07 (予想より高い)

**根本原因**: モデルは「ノイズ混合データのデノイズ」は学習できているが、
「純粋ノイズからの構造生成」が不十分。

### 考えられる原因

1. **データ量不足**: 42サンプル × 16時間ステップ = 672ウィンドウは少ない可能性
2. **学習エポック不足**: 100エポックでは収束が不十分
3. **条件付けの問題**: 幾何条件（円柱マスク）の影響が弱い
4. **ネットワーク容量**: hidden_channels=(96, 192, 384)が不十分

### 推奨対策

1. **短期対策**: `corrections=1`以上を必須で使用
2. **中期対策**: 学習エポック数を300-500に増加
3. **長期対策**: データ拡張、ネットワーク容量増加を検討

---

## 更新履歴

- 2024-12-09: 初版作成、VPSDE学習完了後の調査結果をまとめ
- 2024-12-09: デバッグ結果追加、corrections=1必須の発見
