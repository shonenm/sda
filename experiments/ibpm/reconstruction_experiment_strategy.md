# IBPM 復元実験スクリプト作成戦略（改訂版 v2）

## 概要

IBPM（Immersed Boundary Projection Method）による円柱周り流れ（Re=100）データに対して、スコアベースデータ同化による復元実験を行う。

**重要変更**: Kolmogorov学習済みモデルの転用ではなく、**学習時点からIBPMの幾何制約・境界条件を反映した専用モデル**を構築する。

---

## 変更方針の転換

### 旧方針（v1）からの変更点

| 項目 | 旧方針（転用ベース） | 新方針（専用学習） |
|-----|-----------------|----------------|
| 学習データ | Kolmogorov流 | IBPM円柱流れ |
| 条件チャネル | sin(4x) 外力 | 円柱マスク + 流入プロファイル |
| 境界条件 | circular padding | reflect padding |
| 物理制約 | 推論時のみマスク適用 | 学習時から損失関数に組込 |
| 損失関数 | Score matching のみ | Score + マスク付きMSE + 物理罰則 |
| 推論時制約 | 事後的マスク適用 | 拡散ステップ毎の投影 + ガイダンス |

---

## Kolmogorov流との物理的違い

### 境界条件と外力

| 項目 | Kolmogorov流 | IBPM円柱流れ |
|-----|-------------|-------------|
| 境界条件 | 周期境界（トーラス） | 非周期（流入・流出・壁面） |
| 外力 | sin(4x)の定常強制 | なし（円柱による乱れ） |
| 物理制約 | なし | 円柱内部で速度=0 + no-slip境界 |
| 支配的構造 | 大スケール渦 | カルマン渦列（周期的渦放出） |
| 特徴的長さ | 強制波長 λ=π/2 | 円柱直径 D≈15ピクセル |
| Re数 | 暗黙的に高Re | Re=100（明示的） |
| Padding方式 | circular（周期） | reflect（非周期） |

### 数学的定式化の違い

#### Kolmogorov流の支配方程式

非圧縮性Navier-Stokes方程式 + 外力項:

```
∂u/∂t + (u·∇)u = -∇p + ν∇²u + f(x)
∇·u = 0

境界条件: 周期境界 (トーラス上)
外力: f(x) = f₀ sin(4x) ŷ  (y方向の定常強制)
```

**物理的意味**:
- 強制項f(x)がエネルギーを注入し、大スケール渦を維持
- 周期境界により全方向で空間的に繰り返される
- 定常解は存在しないがアトラクタ上の準周期運動

#### IBPM円柱流れの支配方程式

同じNavier-Stokes方程式だが境界条件と制約が異なる:

```
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0

境界条件:
  - 流入境界 (x=0):    u = U∞, v = 0
  - 流出境界 (x=Lx):   ∂u/∂x = 0 (自由流出)
  - 壁面境界 (y=0,Ly): u = v = 0 (no-slip)
  - 円柱表面:           u = v = 0 (no-slip)

物理制約: Ω_cylinder内で u = 0 (剛体内部)
Reynolds数: Re = U∞D/ν = 100
```

**物理的意味**:
- 外力なし：上流からの流入がエネルギー源
- 円柱による流れの剥離がカルマン渦列を生成
- Re=100で周期的な渦放出（Strouhal数 St≈0.16）

#### スコアモデルが学習する対象

スコアベース生成モデルは以下を学習:

```
∇_x log p(x_t | c)  ≈  s_θ(x_t, t, c)

ここで:
  x_t: 拡散過程の状態 (時刻tでノイズが混入)
  c: 条件付け情報
     - Kolmogorov: sin(4x) の外力チャネル
     - IBPM: [円柱マスク, 流入プロファイル] の幾何条件
  p(x): データ分布（Navier-Stokes方程式の解の分布）
```

**重要な洞察**:
1. **モデルはNavier-Stokes方程式そのものを解くわけではない**
   - データ駆動的に解の分布 p(x) を学習
   - 支配方程式は暗黙的にデータに埋め込まれている

2. **条件付け c の役割（新方針）**:
   - **幾何制約を学習時から明示的に伝達**
   - 円柱マスク: 物体領域（=0）vs 流体領域（=1）
   - 流入プロファイル: 左端境界での速度条件
   - U-Netに結合することで、条件に応じた速度場生成を学習

3. **専用学習の利点**:
   - 境界条件を学習時から反映（reflect padding）
   - 物理制約を損失関数に組込（no-slip, divergence-free）
   - 円柱周りの流れパターンを直接学習
   - 推論時の制約適用がより自然

### IBPM専用モデルの構築方針

**新しいアプローチ**:
```python
# IBPM学習用の条件生成
def build_conditions(H=64, W=64):
    # 円柱マスク（物体=0, 流体=1）
    cylinder_mask = build_cylinder_mask(H, W, center=(32, 37), radius=7.5)

    # 流入プロファイル（左端 x=0 で u=U∞, v=0）
    inflow_profile = build_inflow_profile(H, W, U=1.0)

    # 条件チャネル (C_cond=2)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0)
    return cond

# モデル定義
score_net = LocalScoreUNet(
    in_channels=2,           # u, v
    cond_channels=2,         # mask, inflow (旧: 1 for sin(4x))
    padding_mode='reflect',  # 旧: 'circular'
)

# 学習時
x_t, t = sde.forward_diffuse(x_ibpm)
cond = build_conditions()
score = score_net(x_t, t, cond)

# 損失関数（物理制約付き）
loss = score_matching_loss(score, x_t, t, x_ibpm) \
     + λ_mask * masked_mse(score, x_ibpm, cylinder_mask) \
     + λ_wall * no_slip_penalty(score, cylinder_mask) \
     + λ_div * divergence_penalty(score) \
     + λ_out * outflow_grad_penalty(score)
```

**アーキテクチャの変更点**:
1. **padding_mode**: 'circular' → 'reflect'（全Conv2D層）
2. **条件チャネル**: 外力1ch → 幾何2ch（マスク＋流入）
3. **損失関数**: Score matching + 4種の物理罰則
4. **時間窓**: window=5を維持（時間発展の学習）

---

## データ構造の理解

### HDF5データの形状

```python
# /workspace/data/ibpm_h5/train.h5
shape: (16, 64, 2, 64, 64)
#      (時刻, サンプル, チャネル, H, W)

# /workspace/data/ibpm_h5/test.h5
shape: (5, 64, 2, 64, 64)
#      (時刻, サンプル, チャネル, H, W)
```

**注意**: Kolmogorovとは軸の順序が異なる
- Kolmogorov: (サンプル, 時刻, チャネル, H, W)
- IBPM: (時刻, サンプル, チャネル, H, W)

### 円柱の位置とマスク

データから推定された円柱領域:
- 中心: (x, y) = (32, 37)
- 半径: 約7.5ピクセル
- 領域: 全体の約0.8%（32/4096ピクセル）

```python
def create_cylinder_mask(size=64):
    """円柱外部がTrue、内部がFalse"""
    center_x, center_y = 32.0, 37.0
    radius = 7.5
    # ... 距離計算 ...
    return dist > radius
```

### 物理的制約

**硬制約**: 円柱内部で速度は厳密に0
```python
def apply_cylinder_constraint(x, mask):
    """円柱内部の速度を0に設定"""
    return x * mask[None, None, :, :]
```

---

## 実装変更の詳細

### ファイル別変更指示

#### 2.1 sda/models/local_score_unet.py

**変更内容**:
```python
# 旧: padding_mode='circular'
# 新: padding_mode='reflect'
class LocalScoreUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        cond_channels=2,  # 旧: 1 (sin(4x))
        padding_mode='reflect',  # ← 変更
        ...
    ):
        # 全Conv2D層でreflect paddingを使用
        self.conv1 = nn.Conv2d(..., padding=1, padding_mode='reflect')
        ...

    def forward(self, x, t, cond):
        # cond: (B, 2, H, W) = [cylinder_mask, inflow_profile]
        x = torch.cat([x, cond], dim=1)  # (B, 4, H, W)
        ...
```

#### 2.2 sda/data/ibpm_dataset.py（新規）

**目的**: 幾何条件を含むIBPMデータセットの構築

```python
def build_cylinder_mask(H=64, W=64, center=(32, 37), radius=7.5):
    """円柱マスク: 流体=1, 物体=0"""
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    return (dist > radius).float()

def build_inflow_profile(H=64, W=64, U=1.0):
    """流入プロファイル: x=0でu=U, 他は0"""
    profile = torch.zeros(H, W)
    profile[:, 0] = U  # 左端列のみ非ゼロ
    return profile

class IBPMDataset(Dataset):
    def __init__(self, h5_path, time_window=8):
        self.data = h5py.File(h5_path, 'r')['velocity'][:]  # (T, N, 2, H, W)
        self.cylinder_mask = build_cylinder_mask()
        self.inflow = build_inflow_profile()

    def __getitem__(self, idx):
        # 時間窓の切り出し
        x = self.data[idx:idx+self.time_window]  # (window, 2, 64, 64)

        # 条件チャネル
        cond = torch.stack([self.cylinder_mask, self.inflow], dim=0)

        # 流体マスク（評価用）
        mask = self.cylinder_mask

        return x, cond, mask
```

#### 2.3 sda/losses.py

**重要**: スコアネットの出力は **∇x log p_t（スコア）** であり、速度場そのものではない。
物理損失はTweedie推定量 **x̂** に対して適用する。

**Tweedie推定量の計算**:
```python
def tweedie_estimator(x_t, t, score, sde):
    """Tweedie推定量: x̂ = x_t + σ(t)² * ∇ log p_t

    Args:
        x_t: 拡散状態 (B, C, H, W)
        t: 時刻 (B,)
        score: スコアネット出力 ∇ log p_t (B, C, H, W)
        sde: SDEインスタンス（σ(t)の計算用）

    Returns:
        x̂: クリーンデータの推定 (B, C, H, W)
    """
    # VP-SDEの場合（実装に応じて調整）
    # σ(t)² = β(t) or (1 - α(t)²) など
    sigma_sq = sde.get_sigma_squared(t)  # (B,)
    sigma_sq = sigma_sq.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

    x_hat = x_t + sigma_sq * score
    return x_hat
```

**追加する物理損失**:

```python
def masked_mse(x_hat, x_true, fluid_mask):
    """流体領域のみでMSE計算

    Args:
        x_hat: Tweedie推定量 (B, 2, H, W)
        x_true: 真値 (B, 2, H, W)
        fluid_mask: 流体領域マスク (H, W), 1=流体, 0=物体
    """
    mask_expanded = fluid_mask[None, None, :, :].expand_as(x_hat)
    diff = (x_hat - x_true) * mask_expanded
    n_fluid = mask_expanded.sum()
    return diff.pow(2).sum() / (n_fluid + 1e-12)

def compute_distance_field(cylinder_mask, max_dist=5):
    """円柱境界からの距離場を計算（境界付近の重み付け用）"""
    from scipy.ndimage import distance_transform_edt

    # 流体領域での距離（物体境界からの距離）
    fluid_region = cylinder_mask.cpu().numpy()
    dist = distance_transform_edt(fluid_region)
    dist = torch.from_numpy(dist).float()

    # 距離が小さいほど重みを大きく（境界近傍のペナルティ）
    weight = torch.exp(-dist / max_dist)
    return weight

def no_slip_penalty(x_hat, fluid_mask, distance_field=None):
    """壁・円柱境界近傍での速度罰則

    Args:
        x_hat: Tweedie推定量 (B, 2, H, W)
        fluid_mask: 流体領域マスク (H, W)
        distance_field: 境界からの距離場 (H, W)（オプション）
    """
    # 境界付近のバンド領域を抽出
    if distance_field is None:
        # 簡易版：境界から2px以内
        from scipy.ndimage import binary_erosion
        eroded = torch.from_numpy(
            binary_erosion(fluid_mask.cpu().numpy(), iterations=2)
        ).float()
        boundary_band = fluid_mask - eroded
    else:
        # 距離場を使った重み付け（より精密）
        boundary_band = distance_field

    boundary_band = boundary_band.to(x_hat.device)

    # 境界近傍での速度の大きさをペナルティ
    velocity_mag = x_hat.abs().sum(dim=1)  # (B, H, W)
    penalty = (velocity_mag * boundary_band[None, :, :]).mean()

    return penalty

def divergence_penalty(x_hat, fluid_mask):
    """非圧縮性制約: ∇·u = 0（流体領域のみ）

    中心差分＋境界では片側差分を使用

    注意:
    - グリッド間隔 Δx = Δy = 1 を仮定
    - 実スケールの場合は du_dx / dx_physical などで正規化
    """
    u, v = x_hat[:, 0], x_hat[:, 1]  # (B, H, W)

    # x方向微分（中心差分、Δx = 1）
    du_dx = torch.zeros_like(u)
    du_dx[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / 2.0  # 中心差分 / (2Δx)
    du_dx[:, :, 0] = u[:, :, 1] - u[:, :, 0]         # 左端：前進差分 / Δx
    du_dx[:, :, -1] = u[:, :, -1] - u[:, :, -2]      # 右端：後退差分 / Δx

    # y方向微分（中心差分、Δy = 1）
    dv_dy = torch.zeros_like(v)
    dv_dy[:, 1:-1, :] = (v[:, 2:, :] - v[:, :-2, :]) / 2.0  # 中心差分 / (2Δy)
    dv_dy[:, 0, :] = v[:, 1, :] - v[:, 0, :]         # 上端：前進差分 / Δy
    dv_dy[:, -1, :] = v[:, -1, :] - v[:, -2, :]      # 下端：後退差分 / Δy

    # 発散
    div = du_dx + dv_dy  # (B, H, W)

    # 流体領域のみで評価（物体内は無意味）
    fluid_mask_expanded = fluid_mask[None, :, :].expand_as(div)
    div_masked = div * fluid_mask_expanded
    n_fluid = fluid_mask_expanded.sum()

    return div_masked.pow(2).sum() / (n_fluid + 1e-12)

def outflow_grad_penalty(x_hat):
    """流出境界での勾配抑制: ∂u/∂x ≈ 0, ∂v/∂x ≈ 0（右端境界）

    注意:
    - u, v 両成分に同じペナルティを適用
    - 流出境界では速度の x 方向勾配が小さいことを仮定
    """
    # 右端列とその前列の差分（u, v 両方）
    u_out = x_hat[:, :, :, -1]   # (B, 2, H)
    u_prev = x_hat[:, :, :, -2]
    grad = u_out - u_prev         # ∂u/∂x ≈ (u[x] - u[x-1]) / Δx, Δx=1
    return grad.pow(2).mean()

# 統合損失関数
def composite_loss(score, x_t, t, x_true, cond, fluid_mask, sde, weights):
    """
    Args:
        score: スコアネット出力 (B, 2, H, W)
        x_t: 拡散状態 (B, 2, H, W)
        t: 時刻 (B,)
        x_true: 真値 (B, 2, H, W)
        cond: 条件 (B, 2, H, W)
        fluid_mask: 流体マスク (H, W)
        sde: SDEインスタンス
        weights: dict with keys ['score', 'mask', 'wall', 'div', 'out']
    """
    # スコアマッチング損失
    loss_score = score_matching_loss(score, x_t, t)

    # Tweedie推定量（クリーンデータの推定）
    x_hat = tweedie_estimator(x_t, t, score, sde)

    # 物理損失（x̂に対して適用）
    loss_mask = masked_mse(x_hat, x_true, fluid_mask)
    loss_wall = no_slip_penalty(x_hat, fluid_mask)
    loss_div = divergence_penalty(x_hat, fluid_mask)
    loss_out = outflow_grad_penalty(x_hat)

    total = (
        weights['score'] * loss_score +
        weights['mask'] * loss_mask +
        weights['wall'] * loss_wall +
        weights['div'] * loss_div +
        weights['out'] * loss_out
    )

    return total, {
        'score': loss_score.item(),
        'mask': loss_mask.item(),
        'wall': loss_wall.item(),
        'div': loss_div.item(),
        'out': loss_out.item(),
    }
```

**初期係数**:
```python
loss_weights = {
    'score': 1.0,
    'mask': 1.0,
    'wall': 0.1,
    'div': 0.05,
    'out': 0.05,
}
```

#### 2.4 sda/train_ibpm.py（新規）

**目的**: IBPM専用の学習スクリプト

```python
from sda.data.ibpm_dataset import IBPMDataset
from sda.models.local_score_unet import LocalScoreUNet
from sda.losses import composite_loss

# データローダー
train_dataset = IBPMDataset('/workspace/data/ibpm_h5/train.h5', time_window=8)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# モデル（reflect padding）
score_net = LocalScoreUNet(
    in_channels=2,
    cond_channels=2,
    padding_mode='reflect',
).cuda()

# Optimizer with gradient clipping
optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-4)

# SDE（係数計算のみ、スコアネットは外部）
sde = VPSDE(beta_min=0.1, beta_max=20.0, T=1.0)

# 学習ループ
for epoch in range(num_epochs):
    for batch in train_loader:
        x, cond, mask = batch  # x: (B, T, 2, H, W), cond: (2, H, W), mask: (H, W)

        # 形状整形: (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W).cuda()

        # 条件チャネルをバッチ分拡張
        cond = cond.unsqueeze(0).expand(B * T, -1, H, W).cuda()

        # マスクもバッチ分拡張
        mask = mask.unsqueeze(0).expand(B * T, H, W).cuda()

        # 拡散過程でノイズ付加
        t = torch.rand(B * T).cuda() * (sde.T - sde.eps) + sde.eps
        x_t = sde.forward_diffuse(x, t)

        # スコア計算
        score = score_net(x_t, t, cond)

        # 統合損失（Tweedie推定量を使用）
        loss, loss_dict = composite_loss(
            score, x_t, t, x, cond, mask, sde, loss_weights
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping（NaN防止）
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)

        optimizer.step()

        # ログ記録（W&B等）
        wandb.log({
            'loss/total': loss.item(),
            **{f'loss/{k}': v for k, v in loss_dict.items()}
        })
```

**重要な追加点**:
1. **形状整形**: `(B, T, C, H, W) -> (B*T, C, H, W)` で時系列を展開
2. **条件拡張**: `cond.unsqueeze(0).expand(B*T, ...)` でバッチサイズに合わせる
3. **Gradient clipping**: `max_norm=1.0` でNaN発生を防止
4. **SDEの責務分離**: 係数計算のみ、スコアネットは外部から呼ぶ

#### 2.5 sda/mcs.py

**推論時の制約投影**:

```python
def sample_with_constraints(
    sde, score_net, cond, mask,
    likelihood_score=None,  # GaussianScore(Masked)インスタンス
    **kwargs
):
    """円柱制約付きサンプリング

    Args:
        sde: SDEインスタンス（係数計算のみ、スコアネットは外部）
        score_net: スコアネット
        cond: 条件チャネル (2, H, W)
        mask: 流体マスク (H, W)
        likelihood_score: GaussianScore または GaussianScoreMasked インスタンス（オプション）
        **kwargs: eta_wall, eta_div, max_grad_norm, steps, tau など

    Returns:
        x_0: サンプル (T, 2, H, W)
    """
    # 初期化
    x_t = torch.randn(*sde.shape).to(mask.device) * sde.get_sigma(sde.T)

    # ガイダンス係数（小さめに設定）
    eta_wall = kwargs.get('eta_wall', 1e-4)
    eta_div = kwargs.get('eta_div', 1e-4)
    max_grad_norm = kwargs.get('max_grad_norm', 10.0)
    steps = kwargs.get('steps', 256)
    tau = kwargs.get('tau', 0.5)

    timesteps = torch.linspace(sde.T, sde.eps, steps)

    for t in timesteps:
        # 事前スコア
        score_prior = score_net(x_t, t, cond)

        # 尤度スコア（GaussianScore(Masked)が __call__ を実装）
        if likelihood_score is not None:
            score_lik = likelihood_score(x_t, t)
        else:
            score_lik = 0

        # 制約ガイダンス勾配（弱め、数値安定性に注意）
        if eta_wall > 0 or eta_div > 0:
            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)
                penalty_wall = no_slip_penalty(x_t_grad, mask)
                penalty_div = divergence_penalty(x_t_grad, mask)
                penalty_total = eta_wall * penalty_wall + eta_div * penalty_div

                grad_constraint = torch.autograd.grad(
                    penalty_total, x_t_grad, create_graph=False
                )[0]

                # Gradient clipping（数値安定性）
                grad_norm = grad_constraint.norm()
                if grad_norm > max_grad_norm:
                    grad_constraint = grad_constraint * (max_grad_norm / grad_norm)
        else:
            grad_constraint = 0

        # 拡散ステップ（Euler-Maruyama or PC sampler）
        drift = score_prior + score_lik - grad_constraint
        x_t = euler_maruyama_step(x_t, drift, t, tau, sde)

        # 硬制約投影（毎ステップ、物体内=0）
        x_t = x_t * mask[None, None, :, :]

    return x_t


def euler_maruyama_step(x_t, drift, t, tau, sde):
    """Euler-Maruyama積分ステップ"""
    dt = -tau  # 逆向き拡散
    sigma_t = sde.get_sigma(t)
    x_t = x_t + drift * sigma_t**2 * dt
    return x_t
```

**重要な改善点**:
1. **責務分離**: 尤度計算は `GaussianScore(Masked).__call__` に閉じ込める
2. **引数明確化**: `likelihood_score` インスタンスを外部から渡す
3. **SDEの役割**: 係数計算のみ（`get_sigma`, `get_alpha`）
4. ガイダンス係数を小さく（1e-4）
5. Gradient clippingで数値安定性を確保
6. 硬制約投影（マスク適用）を毎ステップ実行

**GaussianScoreMaskedの__call__実装**（上記2.1節参照）:
```python
class GaussianScoreMasked:
    def __call__(self, x_t, t):
        """尤度スコア ∇ log p(y|x_t)（観測点のみ）"""
        # Tweedie推定でx_0を近似
        score_prior = self.sde.score_net(x_t, t, self.cond)  # condはクラス内保持
        x_0_hat = tweedie_estimator(x_t, t, score_prior, self.sde)

        # 観測との残差（観測点のみ）
        residual = (self.A(x_0_hat) - self.y_obs) * self.obs_mask[None, None, :, :]

        # 尤度勾配（観測点のみで作用）
        grad = -self.A_adjoint(residual) / (self.sigma ** 2)

        # 正規化（観測点数で割る）
        n_obs = self.obs_mask.sum()
        grad = grad * (self.obs_mask.numel() / (n_obs + 1e-12))

        # SDEの係数で調整
        alpha_t = self.sde.get_alpha(t)
        likelihood_score = self.gamma * grad * alpha_t

        return likelihood_score
```

#### 2.6 configs/config_ibpm.yaml（新規）

```yaml
model:
  padding_mode: reflect
  condition_channels: 2

data:
  train_path: /workspace/data/ibpm_h5/train.h5
  test_path: /workspace/data/ibpm_h5/test.h5
  time_window: 8

loss:
  weights:
    score: 1.0
    mask: 1.0
    wall: 0.1
    div: 0.05
    out: 0.05

inference:
  guidance:
    eta_wall: 1e-4  # 小さめに（数値安定性）
    eta_div: 1e-4
    max_grad_norm: 10.0  # Gradient clipping
  steps: 256
  corrections: 2
  tau: 0.5

evaluation:
  boundary_crop: 4  # 境界4pxを除外するオプション
  metrics_both: true  # full + interior の2系統評価
```

#### 2.7 sda/eval/metrics.py

**流体領域限定評価**:

```python
def compute_metrics_fluid_only(x_recon, x_star, mask, crop=0):
    """
    Args:
        crop: 境界から除外するピクセル数
    """
    if crop > 0:
        mask = mask[crop:-crop, crop:-crop]
        x_recon = x_recon[..., crop:-crop, crop:-crop]
        x_star = x_star[..., crop:-crop, crop:-crop]

    mask_expanded = mask[None, None, :, :].expand_as(x_star)
    fluid_diff = (x_recon - x_star) * mask_expanded
    n_fluid = mask_expanded.sum()

    metrics = {
        'rmse': fluid_diff.pow(2).sum().div(n_fluid).sqrt(),
        'relative_error': fluid_diff.pow(2).sum() / (x_star * mask_expanded).pow(2).sum(),
        'vorticity_rmse': compute_vorticity_error(x_recon, x_star, mask),
        'kinetic_energy_error': compute_ke_error(x_recon, x_star, mask),
    }

    return metrics

# 2系統で評価
metrics_full = compute_metrics_fluid_only(x_recon, x_star, mask, crop=0)
metrics_interior = compute_metrics_fluid_only(x_recon, x_star, mask, crop=4)
```

---

## 実験1: 荒い観測からの復元（Coarse Observation）

### 目的

空間的に粗視化（サブサンプリング）された観測から、高解像度の速度場を復元する。

### 1.1 観測演算子

**重要**: エイリアシング防止のため、ローパスフィルタ（平均プーリング）→間引きを使用

```python
import torch.nn.functional as F

def make_coarse_observation_operator(subsample_rate: int):
    """空間サブサンプリング演算子（アンチエイリアス付き）

    Args:
        subsample_rate: 2, 4, 8 (16は情報不足のため除外)

    Returns:
        A: 観測演算子
        A_adjoint: 随伴演算子（尤度計算用）
    """
    def A(x):
        """ローパスフィルタ→間引き"""
        # 平均プーリング（アンチエイリアス）
        x_coarse = F.avg_pool2d(
            x,
            kernel_size=subsample_rate,
            stride=subsample_rate
        )
        return x_coarse

    def A_adjoint(y):
        """随伴演算子: 上向きサンプリング（最近傍補間）"""
        # 尤度勾配を元の解像度に戻す
        y_up = F.interpolate(
            y,
            scale_factor=subsample_rate,
            mode='nearest'
        )
        # 正規化（演算子の正しい随伴）
        y_up = y_up / (subsample_rate ** 2)
        return y_up

    return A, A_adjoint

# 代替: Gaussianブラーを使う場合（より強いローパス、厳密な随伴）
def make_coarse_observation_operator_gaussian(subsample_rate: int, sigma=1.0):
    """Gaussianブラー→間引き（conv2d版、厳密な随伴）

    Args:
        subsample_rate: ダウンサンプリング率
        sigma: ガウシアンカーネルの標準偏差

    Returns:
        A: 観測演算子（conv2d）
        A_adjoint: 随伴演算子（conv_transpose2d）
    """
    # ガウシアンカーネル生成（固定重み）
    kernel_size = subsample_rate * 2 - 1
    kernel = _make_gaussian_kernel_2d(kernel_size, sigma)  # (1, 1, K, K)
    kernel = kernel.repeat(2, 1, 1, 1)  # (2, 1, K, K) for groups=2

    # パディング
    pad = kernel_size // 2

    def A(x):
        """Gaussian blur + stride sampling"""
        # x: (B, 2, H, W)
        x_blur = F.conv2d(x, kernel, stride=subsample_rate, padding=pad, groups=2)
        return x_blur

    def A_adjoint(y):
        """厳密な随伴: conv_transpose2d"""
        # y: (B, 2, H//r, W//r)
        y_up = F.conv_transpose2d(y, kernel, stride=subsample_rate, padding=pad, groups=2)
        return y_up

    return A, A_adjoint


def _make_gaussian_kernel_2d(kernel_size: int, sigma: float):
    """2次元ガウシアンカーネル生成

    Args:
        kernel_size: カーネルサイズ（奇数）
        sigma: 標準偏差

    Returns:
        kernel: (1, 1, K, K) テンソル
    """
    import math

    # 1D ガウシアン
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 2D = outer product
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (K, K)
    gauss_2d = gauss_2d / gauss_2d.sum()

    return gauss_2d.view(1, 1, kernel_size, kernel_size)
```

**注意**:
- `conv_transpose2d`は`conv2d`の厳密な随伴演算子（転置畳み込み）
- `groups=2`で u, v 成分を独立に畳み込み
- カーネルは学習せず固定（`requires_grad=False`）

**観測点数の変化**:
- rate=2: 64×64 → 32×32 (25%の情報量)
- rate=4: 64×64 → 16×16 (6.25%)
- rate=8: 64×64 → 8×8 (1.56%)

### 1.2 観測データ生成

```python
# 真値（テストデータから）
x_star = test_data[sample_idx, :time_window]  # (8, 2, 64, 64)

# ノイズ付き観測
sigma_obs = 0.1
A = make_coarse_observation_operator(subsample_rate)
y_star = A(x_star) + torch.normal(0, sigma_obs, size=A(x_star).shape)
```

### 1.3 スコアベース復元

```python
# ガウス尤度で条件付け
sde = VPSDE(
    GaussianScore(
        y_star,
        A=A,
        std=sigma_obs,
        sde=VPSDE(score, shape=()),
        gamma=1.0,  # IBPM用に調整（Lorenzは3e-2）
    ),
    shape=x_star.shape,
).cuda()

# サンプリング
x_recon = sde.sample(
    steps=256,
    corrections=corrections,  # [0, 1, 2, 4]
    tau=0.5,
).cpu()

# 物理制約の適用（重要！）
x_recon = apply_cylinder_constraint(x_recon, cylinder_mask)
```

### 1.4 評価指標（円柱領域を除外）

```python
def compute_metrics(x_recon, x_star, y_star, A, mask, boundary_crop=0):
    """流体領域限定の評価指標

    Args:
        x_recon: 復元結果 (T, 2, H, W)
        x_star: 真値 (T, 2, H, W)
        y_star: 観測データ（coarse: 小サイズ、sparse: 同サイズ）
        A: 観測演算子
        mask: 円柱マスク (H, W), 1=流体, 0=物体
        boundary_crop: 境界から除外するピクセル数（デフォルト0）

    Returns:
        metrics: 評価指標の辞書
    """
    # 境界クロップ（オプション）
    if boundary_crop > 0:
        c = boundary_crop
        x_recon = x_recon[..., c:-c, c:-c]
        x_star = x_star[..., c:-c, c:-c]
        mask = mask[c:-c, c:-c]

    # マスク拡張
    mask_expanded = mask[None, None, :, :].expand_as(x_star)
    n_fluid = mask_expanded.sum()

    # 1. 観測整合性（データフィッティング）
    obs_residual = A(x_recon) - y_star
    obs_error = obs_residual.pow(2).mean().sqrt()
    # 期待値: ≈ sigma_obs = 0.1

    # 2. 流体領域での復元RMSE
    fluid_diff = (x_recon - x_star) * mask_expanded
    rmse = fluid_diff.pow(2).sum().div(n_fluid + 1e-12).sqrt()

    # 3. 相対誤差（正しい定義: ||diff||_2 / ||x||_2）
    numerator = fluid_diff.pow(2).sum().sqrt()
    denominator = ((x_star * mask_expanded).pow(2).sum().sqrt() + 1e-12)
    relative_error = numerator / denominator

    # 4. 渦度場誤差（物理的妥当性）
    w_star = compute_vorticity(x_star, mask)      # (T, H, W)
    w_recon = compute_vorticity(x_recon, mask)
    w_diff = (w_recon - w_star) * mask[None, :, :]
    n_fluid_2d = mask.sum()
    vorticity_rmse = w_diff.pow(2).sum().div(n_fluid_2d + 1e-12).sqrt()

    # 5. 運動エネルギー誤差
    ke_star = (x_star.pow(2).sum(dim=-3) / 2) * mask   # (T, H, W)
    ke_recon = (x_recon.pow(2).sum(dim=-3) / 2) * mask
    ke_error = (ke_recon - ke_star).abs().sum() / (n_fluid_2d + 1e-12)

    return {
        'obs_error': obs_error.item(),           # RMSE(A(x_recon) - y_obs)
        'rmse': rmse.item(),                     # RMSE in fluid region
        'relative_error': relative_error.item(), # ||x_recon - x_star||_2 / ||x_star||_2
        'vorticity_rmse': vorticity_rmse.item(), # RMSE(ω_recon - ω_star)
        'kinetic_energy_error': ke_error.item(), # Mean absolute error in KE
        'n_fluid_points': n_fluid.item(),        # Number of fluid grid points
    }


def compute_vorticity(u, mask):
    """渦度計算: ω = ∂v/∂x - ∂u/∂y（流体領域のみ）

    Args:
        u: 速度場 (T, 2, H, W) or (B, 2, H, W)
        mask: 流体マスク (H, W)

    Returns:
        vorticity: (T, H, W) or (B, H, W)
    """
    u_vel, v_vel = u[:, 0], u[:, 1]  # (T, H, W)

    # 中心差分＋境界は片側差分
    dv_dx = torch.zeros_like(v_vel)
    dv_dx[:, :, 1:-1] = (v_vel[:, :, 2:] - v_vel[:, :, :-2]) / 2.0
    dv_dx[:, :, 0] = v_vel[:, :, 1] - v_vel[:, :, 0]
    dv_dx[:, :, -1] = v_vel[:, :, -1] - v_vel[:, :, -2]

    du_dy = torch.zeros_like(u_vel)
    du_dy[:, 1:-1, :] = (u_vel[:, 2:, :] - u_vel[:, :-2, :]) / 2.0
    du_dy[:, 0, :] = u_vel[:, 1, :] - u_vel[:, 0, :]
    du_dy[:, -1, :] = u_vel[:, -1, :] - u_vel[:, -2, :]

    vorticity = dv_dx - du_dy

    # 流体領域のみ
    vorticity = vorticity * mask[None, :, :]

    return vorticity


# 2系統で評価
metrics_full = compute_metrics(x_recon, x_star, y_star, A, cylinder_mask, boundary_crop=0)
metrics_interior = compute_metrics(x_recon, x_star, y_star, A, cylinder_mask, boundary_crop=4)

print(f"Full fluid region: RMSE={metrics_full['rmse']:.4f}, Rel={metrics_full['relative_error']:.3f}")
print(f"Interior region: RMSE={metrics_interior['rmse']:.4f}, Rel={metrics_interior['relative_error']:.3f}")
```

**修正点**:
1. **相対誤差**: ||diff||_2 / ||x||_2 に修正（次元整合）
2. **境界クロップ**: 評価時に境界帯を除外するオプション
3. **渦度計算**: 中心差分＋片側差分、流体領域のみ
4. **KE誤差**: 正規化を追加

### 1.5 実験パラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| サブサンプリング率 | [2, 4, 8] | 16は除外（情報不足） |
| 観測ノイズ σ | 0.1 | 速度場の標準的なノイズレベル |
| 時間窓 | 8時刻 | カルマン渦の1-2周期分 |
| テストサンプル数 | 5個 | 全テストデータ使用 |
| サンプリングステップ | 256 | 十分な拡散ステップ |
| 補正ステップ | [0, 1, 2, 4] | 消融実験 |
| tau | 0.5 | ステップサイズ |
| gamma | 1.0 | 尤度重み（Lorenzの3e-2より大） |

**gamma値の根拠**:
- Lorenz (3次元): 観測が少ないため小さいgamma
- IBPM (2D画像): 観測点が多いため大きいgamma
- 画像復元では通常gamma=1.0程度

---

## 実験2: スパース観測からの復元（Sparse Observation）

### 目的

ランダムな位置でのみ観測された速度場から、完全な場を復元する。

### 2.1 観測演算子（マスク適用）

**重要**: 観測点のみで尤度を計算し、未観測点は無視する

```python
def make_sparse_observation_operator(observation_rate: float, cylinder_mask, seed: int = 42):
    """ランダムマスク観測演算子

    Args:
        observation_rate: 観測率 (0.5 = 50%の点で観測)
        cylinder_mask: 円柱マスク (H, W), 1=流体, 0=物体
        seed: 再現性のための乱数シード

    Returns:
        A: 観測演算子（恒等写像）
        A_adjoint: 随伴演算子（恒等写像）
        obs_mask: 観測マスク (H, W), 1=観測点, 0=未観測
    """
    torch.manual_seed(seed)
    H, W = 64, 64

    # ランダムマスク生成（流体領域のみ）
    random_mask = torch.rand(H, W) < observation_rate
    obs_mask = random_mask * cylinder_mask  # 論理積→float

    # 観測演算子は恒等（マスクは尤度計算で使用）
    def A(x):
        """恒等演算子（マスクは別途適用）"""
        return x

    def A_adjoint(y):
        """恒等演算子の随伴"""
        return y

    return A, A_adjoint, obs_mask


# GaussianScoreクラスにマスク対応を追加
class GaussianScoreMasked:
    """マスク対応のガウス尤度スコア

    注意:
    - score_net と cond は外部から渡してクラス内で保持
    - Tweedie推定量の計算に必要
    """

    def __init__(self, y_obs, A, A_adjoint, sigma, obs_mask, sde, score_net, cond, gamma=1.0):
        """
        Args:
            y_obs: 観測データ (T, 2, H, W)
            A: 観測演算子（恒等）
            A_adjoint: 随伴演算子（恒等）
            sigma: 観測ノイズ標準偏差
            obs_mask: 観測マスク (H, W), 1=観測点, 0=未観測
            sde: SDEインスタンス（係数計算用）
            score_net: スコアネットワーク（Tweedie推定に必要）
            cond: 条件チャネル (2, H, W)（幾何条件）
            gamma: 尤度の重み
        """
        self.y_obs = y_obs
        self.A = A
        self.A_adjoint = A_adjoint
        self.sigma = sigma
        self.obs_mask = obs_mask
        self.sde = sde
        self.score_net = score_net
        self.cond = cond
        self.gamma = gamma

    def __call__(self, x_t, t):
        """尤度スコア ∇ log p(y|x_t)（観測点のみ）

        Args:
            x_t: 拡散状態 (B, 2, H, W)
            t: 時刻 (scalar or B,)

        Returns:
            likelihood_score: (B, 2, H, W)
        """
        # Tweedie推定でx_0を近似
        score_prior = self.score_net(x_t, t, self.cond)
        x_0_hat = tweedie_estimator(x_t, t, score_prior, self.sde)

        # 観測との残差（観測点のみ）
        residual = (self.A(x_0_hat) - self.y_obs) * self.obs_mask[None, None, :, :]

        # 尤度勾配（観測点のみで作用）
        grad = -self.A_adjoint(residual) / (self.sigma ** 2)

        # 正規化（観測点数で割る）
        n_obs = self.obs_mask.sum()
        grad = grad * (self.obs_mask.numel() / (n_obs + 1e-12))  # スケール補正

        # SDEの係数で調整
        alpha_t = self.sde.get_alpha(t)
        likelihood_score = self.gamma * grad * alpha_t

        return likelihood_score
```

**重要な変更点**:
1. 観測演算子Aは恒等（マスクは尤度計算で使用）
2. 残差計算で obs_mask を掛けて観測点のみ評価
3. スケール補正（観測点数で正規化）
4. 円柱内部は常に未観測（obs_mask = 0）

### 2.2 スパース観測の生成

```python
# 観測率50%の場合
A, obs_mask = make_sparse_observation_operator(observation_rate=0.5, seed=42)

# 観測（マスク位置のみノイズ付加）
y_star = A(x_star) + torch.normal(0, sigma_obs, size=x_star.shape) * obs_mask[None, None, :, :]

# 実効的な観測点数
cylinder_mask = create_cylinder_mask(size=64)
n_fluid = cylinder_mask.sum()  # 流体領域の点数
n_observed = obs_mask.sum()    # 観測点数
effective_rate = n_observed / n_fluid
print(f"Observation rate: {effective_rate:.1%} ({n_observed}/{n_fluid} points)")
```

### 2.3 スパース観測用スコア

```python
# スパース観測では GaussianScoreMasked を使用
A, A_adjoint, obs_mask = make_sparse_observation_operator(
    observation_rate=0.5,
    cylinder_mask=cylinder_mask,
    seed=42
)

# マスク対応尤度スコア
likelihood_score = GaussianScoreMasked(
    y_obs=y_star,
    A=A,
    A_adjoint=A_adjoint,
    sigma=sigma_obs,
    obs_mask=obs_mask,
    sde=sde,
    score_net=score_net,  # Tweedie推定に必要
    cond=cond,            # 幾何条件
    gamma=2.0,            # 荒い観測より大きく
)

# サンプリング（制約付き）
x_recon = sample_with_constraints(
    sde=sde,
    score_net=score_net,
    cond=cond,
    mask=cylinder_mask,
    y_obs=y_star,
    A=A,
    sigma_obs=sigma_obs,
    likelihood_score=likelihood_score,  # マスク対応尤度を渡す
    steps=256,
    corrections=2,  # スパース観測では補正が重要
    tau=0.5,
    eta_wall=1e-4,
    eta_div=1e-4,
).cpu()

# 硬制約投影は sample_with_constraints 内で毎ステップ実行済み
```

### 2.4 実験パラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| 観測率 | [0.5, 0.25, 0.1] | 5%は除外（過度にスパース） |
| 観測ノイズ σ | 0.1 | 荒い観測と同じ |
| 時間窓 | 8時刻 | 同上 |
| 補正ステップ | 2 | スパース観測では重要 |
| gamma | 2.0 | 荒い観測の2倍 |
| ランダムシード | 42 | 再現性のため固定 |

---

## スクリプト構成

### ファイル構造

```
sda/experiments/ibpm/
├── eval_utils.py           # 共通ユーティリティ
│   ├── create_cylinder_mask()
│   ├── apply_cylinder_constraint()
│   ├── compute_vorticity()
│   ├── compute_metrics()
│   └── save_results_to_csv()
├── eval_coarse.py          # 実験1: 荒い観測
└── eval_sparse.py          # 実験2: スパース観測
```

### ジョブ配列の設計

#### eval_coarse.py
```python
# ジョブID構造: job_id = sample_idx * 12 + subsample_idx * 4 + correction_idx
# - sample_idx: 0-4 (5サンプル)
# - subsample_idx: 0-2 (3つのrate)
# - correction_idx: 0-3 (4つのcorrections)
# 合計: 5 × 3 × 4 = 60 ジョブ

@job(array=60, cpus=2, gpus=1, ram='16GB', time='02:00:00')
def evaluate_coarse_reconstruction(job_id: int):
    sample_idx = job_id // 12
    remainder = job_id % 12
    subsample_idx = remainder // 4
    correction_idx = remainder % 4
    # ...
```

#### eval_sparse.py
```python
# ジョブID構造: job_id = sample_idx * 9 + obs_rate_idx * 3 + seed_idx
# - sample_idx: 0-4 (5サンプル)
# - obs_rate_idx: 0-2 (3つの観測率)
# - seed_idx: 0-2 (3つの異なるマスクパターン)
# 合計: 5 × 3 × 3 = 45 ジョブ

@job(array=45, cpus=2, gpus=1, ram='16GB', time='02:00:00')
def evaluate_sparse_reconstruction(job_id: int):
    sample_idx = job_id // 9
    remainder = job_id % 9
    obs_rate_idx = remainder // 3
    seed_idx = remainder % 3
    # ...
```

---

## 技術的課題と対策

### 1. Tweedie推定量の実装

**課題**: SDEによってσ(t)の定義が異なる
**対策**:
- VP-SDE: σ²(t) = 1 - exp(-∫β(s)ds) = 1 - α²(t)
- 実装のSDE定義に合わせて係数を調整
- ユニットテストで x̂ → x_0 の収束を確認

### 2. 円柱制約の扱い

**課題**: 推論中に円柱内部が非ゼロになる可能性
**対策**:
- **学習時**: データ自体が物体内=0（データセット前処理で保証）
- **推論時**: 毎拡散ステップで硬制約投影 `x_t *= mask`
- **評価時**: 流体領域のみで指標計算（物体内は無視）
- 可視化時も円柱内部をグレーアウト

### 3. 境界条件とpadding

**課題**: IBPMは非周期境界（Dirichlet + Neumann混在）
**対策**:
- **短期**: reflect paddingで近似（全辺同一の妥協）
- **中期**: 手動パディング（物理境界値を埋める）→ padding='valid'
- **評価**: 境界帯（4px）を除外するオプションで2系統評価
- 境界近傍の精度低下を文書化

### 4. 損失係数のチューニング

**課題**: 物理罰則の最適な重みが未知
**対策**:
- 初期値: `{'score': 1.0, 'mask': 1.0, 'wall': 0.1, 'div': 0.05, 'out': 0.05}`
- W&Bで各損失の値を監視
- 物理罰則が支配的にならないよう調整（score_loss > 物理loss）
- 検証セットでの復元精度で最終決定

### 5. ガイダンス係数のチューニング

**課題**: 推論時の制約ガイダンスの最適な重みが未知
**対策**:
- 初期値: eta_wall=1e-4, eta_div=1e-4（小さめに）
- Gradient clippingで数値安定性確保（max_norm=10.0）
- 過度なガイダンスはartifactを生むため注意
- obs_error ≈ sigma_obsとなるようgammaも調整

### 6. 計算時間とリソース

**課題**: steps=256 + 物理罰則の微分は計算コスト大
**対策**:
- ジョブ配列で並列化（60+45=105ジョブ）
- タイムリミット2時間/ジョブ
- 必要に応じてsteps削減（128に）
- ガイダンス係数を小さくしてautograd回数削減

### 7. 数値安定性

**課題**: 境界での微分計算、マスク演算でNaNが発生しうる
**対策**:
- 全ての除算に `+ 1e-12` を追加
- Gradient clippingを推論・学習両方で適用（`max_norm=1.0`）
- 損失値をログ監視、NaN発生時にチェックポイント復帰
- W&Bで損失値の履歴を可視化

### 8. 条件チャネルの拡張（オプション）

**課題**: 境界近傍の学習が不安定になる可能性
**対策** (余裕があれば):
- **SDF（Signed Distance Field）** を追加: 境界までの符号付き距離（1ch）
- 実装例:
```python
def build_sdf(cylinder_mask, H=64, W=64):
    """符号付き距離場: 外部=正、内部=負"""
    from scipy.ndimage import distance_transform_edt

    fluid_region = cylinder_mask.cpu().numpy()
    solid_region = 1 - fluid_region

    dist_fluid = distance_transform_edt(fluid_region)   # 外部での距離
    dist_solid = distance_transform_edt(solid_region)   # 内部での距離

    sdf = dist_fluid - dist_solid  # 符号付き
    return torch.from_numpy(sdf).float()

# 条件チャネルを3chに拡張
cond = torch.stack([cylinder_mask, inflow_profile, sdf], dim=0)  # (3, H, W)
```
- `cond_channels=3` に変更
- 境界勾配を陽に与えることで壁面ペナルティの学習が安定

---

## 期待される出力

### 数値結果

#### coarse_reconstruction.csv
```csv
job_id,sample_id,subsample_rate,corrections,obs_error,recon_error,vorticity_error,relative_error,kinetic_energy_error
0,0,2,0,0.1023,0.2145,0.1876,0.1543,0.0234
1,0,2,1,0.1012,0.1876,0.1543,0.1234,0.0198
...
```

#### sparse_reconstruction.csv
```csv
job_id,sample_id,observation_rate,seed,obs_error,recon_error,vorticity_error,relative_error,kinetic_energy_error,n_observed,n_fluid
0,0,0.5,42,0.1034,0.1987,0.1654,0.1432,0.0256,2048,4064
...
```

### 可視化

- `figures/coarse_sub{rate}_sample{id}_corr{C}.png`
  - 真値、観測、復元の渦度場を並べて表示
  - 円柱領域をグレーアウト

- `figures/sparse_obs{rate}_sample{id}_seed{s}.png`
  - 同上、観測マスクも表示

---

## 成功基準（受け入れ基準）

### 必須条件（Acceptance Criteria）
1. ✅ **境界整合**: 学習・推論とも**reflect padding**で一貫（circularは登場しない）
2. ✅ **条件一貫**: cond は 幾何（mask＋inflow）のみ。外力チャネルは一切不使用
3. ✅ **物理Loss**: 学習ログに masked_mse, wall_penalty, div_penalty, out_grad_penalty が出力
4. ✅ **推論制約**: 生成中にマスク投影が毎ステップ適用されている（ユニットテストで0域確認）
5. ✅ **評価**:
   - fluid_only 指標と fluid_minus_boundary 指標が両方出力
   - coarse(s=2,4,8)・sparse(r=50,25,10%) で一通り走る
6. ✅ **データ整合性**: obs_error ≈ σ = 0.1 (±20%)
7. ✅ **物理制約**: 円柱内部で速度=0（厳密、推論時に確認）
8. ✅ **完全実行**: 全105ジョブが正常終了（NaNなし）
9. ✅ **コード健全性**: 既存Kolmogorovパイプは壊さない（設定で分岐）

### 復元精度の目標
- **coarse rate=2**: relative_error < 15%（interior）
- **coarse rate=4**: relative_error < 30%（interior）
- **sparse obs=50%**: relative_error < 20%（interior）
- **sparse obs=25%**: relative_error < 35%（interior）

### 補正効果とガイダンス
- corrections増加で誤差減少が観測されること
- corrections=1 vs 0で10%以上の改善
- ガイダンス（eta_wall, eta_div）が発散罰則を抑制すること

---

## 実装スケジュール

### フェーズ1: モデル構築（新規）
- ⬜ `sda/models/local_score_unet.py` の修正（padding_mode='reflect', cond_channels=2）
- ⬜ `sda/data/ibpm_dataset.py` の実装（幾何条件の生成）
- ⬜ `sda/losses.py` の実装（Tweedie推定量＋物理損失）
- ⬜ `configs/config_ibpm.yaml` の作成
- ⬜ Tweedie推定量のユニットテスト

### フェーズ2: 学習実行
- ⬜ `sda/train_ibpm.py` の実装
- ⬜ ローカルでの学習デバッグ（1 epoch）
- ⬜ W&Bの設定とログ確認
- ⬜ 本学習の実行（GPUクラスタ）
- ⬜ 学習済みモデルの検証（サンプリングテスト）

### フェーズ3: 評価スクリプト
- ⬜ `sda/mcs.py` の修正（制約付きサンプリング）
- ⬜ `sda/eval/metrics.py` の実装（2系統評価）
- ⬜ `eval_coarse.py` の実装（ローパスフィルタ付き）
- ⬜ `eval_sparse.py` の実装（マスク対応尤度）
- ⬜ ローカルでの動作確認（sample_id=0のみ）
- ⬜ パラメータ調整（gamma, eta_wall, eta_div）

### フェーズ4: 本実行
- ⬜ SLURMジョブ配列の投入（105ジョブ）
- ⬜ 結果の監視とデバッグ（NaN検出）
- ⬜ CSVデータの集約
- ⬜ 物理制約の検証（円柱内=0の確認）

### フェーズ5: 分析と文書化
- ⬜ 可視化スクリプトの作成
- ⬜ 統計分析とプロット（2系統比較）
- ⬜ 論文用図表の生成
- ⬜ 技術報告書の執筆（境界条件の妥協点を明記）

---

## 参考資料

### 技術参考
- **Tweedie推定量**: Song et al. "Score-Based Generative Modeling through SDEs" (2021)
- **物理インフォームドNN**: Raissi et al. "Physics-Informed Neural Networks" (2019)
- **Immersed Boundary Method**: Colonius & Taira "A fast immersed boundary method" (2008)

### 実装参考
- `sda/models/local_score_unet.py`: U-Netベースのスコアネット
- `sda/score.py`: GaussianScore クラス（尤度計算）
- `sda/mcs.py`: PC sampler（予測-補正サンプリング）
- Lorenz実験: データ同化の評価指標の参考

### 変更履歴
- **v1**: Kolmogorov学習済みモデルの転用方針（破棄）
- **v2**: IBPM専用モデルの学習方針（本文書）
  - Tweedie推定量を使った物理損失
  - マスク対応の尤度計算
  - ローパスフィルタ付きcoarse観測
  - 正しい相対誤差の定義
  - 厳密な随伴演算子（conv_transpose2d）
  - 責務分離（SDEは係数のみ、スコアネットは外部）
  - 時系列データの形状整形
  - Gradient clipping（学習・推論）

---

## 実装の要点まとめ

### コアの修正点（必須）

1. **Tweedie推定量の使用** ✓
   - 物理損失は `x̂ = x_t + σ²(t) * score` に対して適用
   - `masked_mse(x_hat, x_true, mask)` で流体領域のみ評価

2. **GaussianScoreMaskedの統一** ✓
   - sparse観測では必ず `GaussianScoreMasked` を使用
   - `score_net`, `cond` をクラス内に保持
   - 観測点のみで尤度を計算（未観測点は無視）

3. **厳密な随伴演算子** ✓
   - coarse: `F.avg_pool2d` + `F.interpolate(nearest)` + スケーリング
   - Gaussian: `F.conv2d` + `F.conv_transpose2d`（厳密な転置畳み込み）

4. **時系列の形状整形** ✓
   - `(B, T, C, H, W) -> (B*T, C, H, W)` で展開
   - `cond` と `mask` をバッチ分拡張

5. **数値安定性** ✓
   - 全除算に `+ 1e-12`
   - Gradient clipping: `max_norm=1.0`（学習）、`max_norm=10.0`（推論ガイダンス）
   - ガイダンス係数: `eta_wall=eta_div=1e-4`（小さめに）

6. **責務分離** ✓
   - SDEは係数計算のみ（`get_sigma`, `get_alpha`）
   - スコアネットは外部から呼ぶ
   - 尤度は `GaussianScore(Masked).__call__` に閉じ込める

7. **相対誤差の正しい定義** ✓
   - `||diff||_2 / ||x||_2`（次元整合）
   - メトリクスdictにコメント付きで明記

8. **発散・壁面罰則** ✓
   - 中心差分＋片側差分（境界）
   - 流体領域のみで評価（`div * fluid_mask`）
   - 距離場を使った境界近傍の重み付け（オプション）
