# 拡散モデルによる逆問題における勾配スケーリング問題

## 概要

拡散モデルを用いた逆問題（Inverse Problems）において、観測点数が増加すると
尤度勾配が比例して大きくなり、事前スコアとのバランスが崩壊する問題。

これは **"Gradient Dominance"** または **"Guidance Strength Scaling"** と呼ばれる
一般的な課題であり、高解像度データを扱う際に顕在化する。

---

## 1. 理論的背景

### 1.1 ベイズ的スコア更新

拡散モデルによる事後分布サンプリングでは、以下のスコア更新が行われる：

```
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)
                 ─────────────   ─────────────
                   事前スコア      尤度勾配
                 (Prior Score)  (Likelihood Gradient)
```

### 1.2 次元依存性の問題

高次元空間における **測度の集中（Concentration of Measure）** により：

- **事前スコアのノルム**: データ次元数 D に依存し、概ね √D のオーダー
  （拡散モデルの学習時に正規化されるため比較的安定）

- **尤度勾配のノルム**: 観測点数 N による二乗誤差の総和（`.sum()`）であるため、
  N または √N に比例して増大

**問題の本質**: N（観測点数）が変動する場合、尤度勾配の「強さ」が劇的に変わり、
事前スコアとのバランスが崩れる。

### 1.3 現象の分類

| 状態 | 条件 | 結果 |
|------|------|------|
| **Likelihood Dominance** | N が大きすぎる | 数値爆発、観測への過剰適合 |
| **Balanced** | 適切なバランス | 正常な再構成 |
| **Prior Dominance** | N が小さすぎる | 観測が無視される、フラット出力 |

---

## 2. 本問題での発現

### 修正前の結果画像

修正前（IBPM sub=4基準のスケーリング）の結果は以下に保存:

```
results/ibpm/evaluate/sparse_before_fix/
├── sparse_ground_truth.png
├── sparse_sub2_reconstructed.png   # 爆発（±20000）
├── sparse_sub4_reconstructed.png   # 成功
├── sparse_sub8_reconstructed.png   # 動作
└── sparse_sub16_reconstructed.png  # 動作
```

- **sub=2**: 値が±20000に爆発（Likelihood Dominance）
- **sub=4**: カルマン渦が見える（Balanced - 偶然）
- **sub=8, 16**: 動作するがPrior Dominanceの可能性

### 2.1 GaussianScoreの実装 (`sda/score.py:549`)

```python
log_p = -(err ** 2 / var).sum() / 2  # 対数尤度
s, = torch.autograd.grad(log_p, x)   # 勾配
return eps - sigma * s               # 事前スコア - σ × 尤度勾配
```

`.sum()` により **勾配の大きさが観測点数に比例** する：

```
|s| ∝ N / std²
```

### 2.2 Kolmogorov vs IBPM

**Kolmogorov（参照実装）**: `(8, 2, 64, 64)` = 8,192 要素

| subsample | 観測点数 | std | 動作 |
|-----------|---------|-----|------|
| 2 | 16,384 | 0.1 | ✓ |
| 4 | 4,096 | 0.1 | ✓ |
| 8 | 1,024 | 0.1 | ✓ |
| 16 | 256 | 0.1 | ✓ |

**IBPM**: `(32, 199, 399)` ≈ 2,500,000 要素

| subsample | 観測点数 | Kolmo比 | 結果 | 状態 |
|-----------|---------|---------|------|------|
| 2 | 630,000 | **38x** | 爆発 | Likelihood Dominance |
| 4 | 160,000 | 10x | ギリギリ | Balanced (偶然) |
| 8 | 40,000 | 2.4x | フラット | Prior Dominance |
| 16 | 9,600 | 0.6x | フラット | Prior Dominance |

---

## 3. 解決策の分類

### 3.1 Likelihood Weighting（尤度重み付け）- 現在の対応

観測点数に応じて `std` をスケーリングし、勾配の大きさを正規化する。

```python
n_obs_kolmo_ref = 16384  # Kolmogorov sub=2 を基準
std_scaled = 0.1 * math.sqrt(n_obs / n_obs_kolmo_ref)
```

**数学的根拠**:
```
勾配バランス: N_ref / std_ref² = N_target / std_target²
→ std_target = std_ref × √(N_target / N_ref)
```

| 評価項目 | 結果 |
|----------|------|
| 数学的妥当性 | ✓ 勾配バランス式から導出 |
| 実装容易性 | ✓ パラメータ変更のみ |
| 後方互換性 | ✓ 既存コード変更なし |
| 汎用性 | △ 問題ごとに調整が必要 |
| 根本解決 | ✗ `.sum()`の問題は残る |

### 3.2 Adaptive Gradient Normalization（適応的勾配正規化）

尤度勾配を事前スコアのノルムに合わせて自動正規化する。

```python
g_total = s_prior + ξ * (||s_prior|| / ||g_lik||) * g_lik
```

**利点**:
- 観測点数に依存しない自動調整
- ハイパーパラメータ（ξ）が1つで済む

**欠点**:
- 実装が複雑
- 勾配のノルム計算のオーバーヘッド

**参考文献**: DPS派生手法、Manifold Constrained Gradient (MCG)

### 3.3 Projection / Decomposition Methods（射影法）

観測空間と非観測空間を分離し、観測点は直接置換する。

```
観測点:   x_i = y_i + noise  (データで強制置換)
非観測点: x_i = diffusion output
```

**利点**:
- 勾配爆発の概念自体がなくなる
- 数学的に安定

**欠点**:
- 観測ノイズがある場合、不連続面が生じる可能性
- 流体力学的整合性（連続の式など）を壊すリスク

**参考文献**: DDRM (Denoising Diffusion Restoration Models)

### 3.4 Proximal Methods（近接法）

正則化項（拡散モデル）とデータ忠実項（観測）を分離して交互に最適化。

**利点**:
- 数学的に収束が保証されやすい
- 各ステップが明確に分離

**欠点**:
- 実装が複雑
- 計算コストが高い

**参考文献**: PnP-ADMM, RED (Regularization by Denoising)

---

## 4. 各解決策の評価（IBPM向け）

| アプローチ | 推奨度 | 理由 |
|-----------|--------|------|
| **3.1 Likelihood Weighting** | ★★★ 推奨 | 実装容易、既存コードとの整合性高い |
| 3.2 Adaptive Normalization | ★★☆ 有力 | 観測点数が動的に変わる場合に有効 |
| 3.3 DDRM (射影法) | ★☆☆ 注意 | 流体データでは物理的整合性を壊すリスク |
| 3.4 Proximal Methods | ★☆☆ 複雑 | 実装コストに見合うメリットが不明 |

---

## 5. 長期的な改善案

### 5.1 GaussianScoreへの正規化オプション追加

```python
class GaussianScore(nn.Module):
    def __init__(self, ..., normalize: bool = False):
        self.normalize = normalize

    def forward(self, x, t, c=None):
        ...
        if self.normalize:
            log_p = -(err ** 2 / var).mean() / 2  # 正規化
        else:
            log_p = -(err ** 2 / var).sum() / 2   # 後方互換
```

### 5.2 Adaptive Normalizationの組み込み

```python
class AdaptiveGaussianScore(nn.Module):
    def forward(self, x, t, c=None):
        eps = self.sde.eps(x, t, c)  # 事前スコア
        g_lik = self._likelihood_grad(x, t, c)  # 尤度勾配

        # 勾配ノルムを自動調整
        scale = eps.norm() / (g_lik.norm() + 1e-8)
        return eps - self.sigma(t) * scale * g_lik
```

---

## 6. 結論

本問題は **「高次元逆問題における勾配支配（Gradient Dominance）」** という
一般的な課題の具体例である。

**短期対応**: Likelihood Weighting（stdスケーリング）で実用上は解決可能

**長期対応**: GaussianScoreに正規化オプションを追加し、
高解像度実験では `normalize=True` をデフォルトにすることを推奨

---

## 参考文献

- Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (DPS)
- Kawar et al., "Denoising Diffusion Restoration Models" (DDRM)
- Song et al., "Score-Based Generative Modeling through SDEs"
- Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction" (PnP)
