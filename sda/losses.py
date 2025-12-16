r"""Loss functions for physics-informed score-based models

物理インフォームド損失関数:
- Tweedie推定量を使った物理損失
- マスク付きMSE
- No-slip境界条件の罰則
- 発散（非圧縮性）罰則
- 流出境界勾配罰則
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


def tweedie_estimator(
    x_t: Tensor,
    t: Tensor,
    score: Tensor,
    sde: 'VPSDE',
) -> Tensor:
    """Tweedie推定量: x̂ = x_t + σ²(t) * ∇ log p_t

    Args:
        x_t: 拡散状態 (B, C, H, W)
        t: 時刻 (B,) or (B, 1, 1, 1)
        score: スコアネット出力 ∇ log p_t (B, C, H, W)
        sde: SDEインスタンス（σ(t)の計算用）

    Returns:
        x_hat: クリーンデータの推定 (B, C, H, W)
    """
    # VP-SDEの場合: σ²(t) = 1 - α²(t)
    # 実装のSDE定義に応じて調整が必要
    if hasattr(sde, 'get_sigma_squared'):
        sigma_sq = sde.get_sigma_squared(t)  # (B,)
    elif hasattr(sde, 'sigma'):
        sigma_sq = sde.sigma(t) ** 2
    else:
        # 簡易的な実装: β(t) から計算
        # β(t) = β_min + t * (β_max - β_min)
        beta_t = sde.beta_min + t * (sde.beta_max - sde.beta_min)
        sigma_sq = 1 - torch.exp(-beta_t)

    # 形状を合わせる
    while sigma_sq.ndim < score.ndim:
        sigma_sq = sigma_sq.unsqueeze(-1)

    x_hat = x_t + sigma_sq * score
    return x_hat


def masked_mse(
    x_hat: Tensor,
    x_true: Tensor,
    fluid_mask: Tensor
) -> Tensor:
    """流体領域のみでMSE計算

    Args:
        x_hat: Tweedie推定量 (B, C, H, W)
        x_true: 真値 (B, C, H, W)
        fluid_mask: 流体領域マスク (H, W), 1=流体, 0=物体

    Returns:
        loss: スカラー損失
    """
    # マスクを拡張
    mask_expanded = fluid_mask[None, None, :, :].expand_as(x_hat)

    # 流体領域のみで差分を計算
    diff = (x_hat - x_true) * mask_expanded
    n_fluid = mask_expanded.sum()

    return diff.pow(2).sum() / (n_fluid + 1e-12)


def compute_distance_field(
    cylinder_mask: Tensor,
    max_dist: float = 5.0
) -> Tensor:
    """円柱境界からの距離場を計算（境界付近の重み付け用）

    Args:
        cylinder_mask: 円柱マスク (H, W), 1=流体, 0=物体
        max_dist: 最大距離（正規化用）

    Returns:
        weight: (H, W) テンソル、境界に近いほど大きい
    """
    from scipy.ndimage import distance_transform_edt

    # 流体領域での距離（物体境界からの距離）
    fluid_region = cylinder_mask.cpu().numpy()
    dist = distance_transform_edt(fluid_region)
    dist = torch.from_numpy(dist).float()

    # 距離が小さいほど重みを大きく（境界近傍のペナルティ）
    weight = torch.exp(-dist / max_dist)
    return weight


def no_slip_penalty(
    x_hat: Tensor,
    fluid_mask: Tensor,
    distance_field: Optional[Tensor] = None
) -> Tensor:
    """壁・円柱境界近傍での速度罰則

    Args:
        x_hat: Tweedie推定量 (B, C, H, W)
        fluid_mask: 流体領域マスク (H, W)
        distance_field: 境界からの距離場 (H, W)（オプション）

    Returns:
        loss: スカラー損失
    """
    # 境界付近のバンド領域を抽出
    if distance_field is None:
        # 簡易版：境界から2px以内
        from scipy.ndimage import binary_erosion
        eroded = torch.from_numpy(
            binary_erosion(fluid_mask.cpu().numpy(), iterations=2)
        ).float().to(fluid_mask.device)
        boundary_band = fluid_mask - eroded
    else:
        # 距離場を使った重み付け（より精密）
        boundary_band = distance_field

    boundary_band = boundary_band.to(x_hat.device)

    # 境界近傍での速度の大きさをペナルティ
    velocity_mag = x_hat.abs().sum(dim=1)  # (B, H, W)
    penalty = (velocity_mag * boundary_band[None, :, :]).mean()

    return penalty


def divergence_penalty(
    x_hat: Tensor,
    fluid_mask: Tensor
) -> Tensor:
    """非圧縮性制約: ∇·u = 0（流体領域のみ）

    中心差分＋境界では片側差分を使用

    注意:
    - グリッド間隔 Δx = Δy = 1 を仮定
    - 実スケールの場合は du_dx / dx_physical などで正規化

    Args:
        x_hat: Tweedie推定量 (B, C, H, W)、C=2（u, v成分）
        fluid_mask: 流体領域マスク (H, W)

    Returns:
        loss: スカラー損失
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


def outflow_grad_penalty(x_hat: Tensor) -> Tensor:
    """流出境界での勾配抑制: ∂u/∂x ≈ 0, ∂v/∂x ≈ 0（右端境界）

    注意:
    - u, v 両成分に同じペナルティを適用
    - 流出境界では速度の x 方向勾配が小さいことを仮定

    Args:
        x_hat: Tweedie推定量 (B, C, H, W)、C=2（u, v成分）

    Returns:
        loss: スカラー損失
    """
    # 右端列とその前列の差分（u, v 両方）
    u_out = x_hat[:, :, :, -1]   # (B, C, H)
    u_prev = x_hat[:, :, :, -2]
    grad = u_out - u_prev         # ∂u/∂x ≈ (u[x] - u[x-1]) / Δx, Δx=1
    return grad.pow(2).mean()


def composite_loss(
    score: Tensor,
    x_t: Tensor,
    t: Tensor,
    x_true: Tensor,
    cond: Tensor,
    fluid_mask: Tensor,
    sde: 'VPSDE',
    weights: Dict[str, float],
    score_matching_fn: Optional[Callable] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """統合損失関数

    Args:
        score: スコアネット出力 (B, C, H, W)
        x_t: 拡散状態 (B, C, H, W)
        t: 時刻 (B,)
        x_true: 真値 (B, C, H, W)
        cond: 条件 (C_cond, H, W) or (B, C_cond, H, W)
        fluid_mask: 流体マスク (H, W)
        sde: SDEインスタンス
        weights: 損失の重み dict with keys ['score', 'mask', 'wall', 'div', 'out']
        score_matching_fn: スコアマッチング損失関数（オプション）

    Returns:
        total_loss: 総損失
        loss_dict: 各損失の値
    """
    # スコアマッチング損失
    if score_matching_fn is not None:
        loss_score = score_matching_fn(score, x_t, t, x_true, sde)
    else:
        # デフォルト: L2損失
        # 本来のスコアマッチング損失は ∇ log p_t と score の一致
        # 簡易的には (x_true - x_t) / σ²(t) との L2 距離
        target_score = (x_true - x_t)
        loss_score = F.mse_loss(score, target_score)

    # Tweedie推定量（クリーンデータの推定）
    x_hat = tweedie_estimator(x_t, t, score, sde)

    # 物理損失（x̂に対して適用）
    loss_mask = masked_mse(x_hat, x_true, fluid_mask)
    loss_wall = no_slip_penalty(x_hat, fluid_mask)
    loss_div = divergence_penalty(x_hat, fluid_mask)
    loss_out = outflow_grad_penalty(x_hat)

    # 総損失
    total = (
        weights.get('score', 1.0) * loss_score +
        weights.get('mask', 1.0) * loss_mask +
        weights.get('wall', 0.1) * loss_wall +
        weights.get('div', 0.05) * loss_div +
        weights.get('out', 0.05) * loss_out
    )

    # 損失の詳細
    loss_dict = {
        'score': loss_score.item(),
        'mask': loss_mask.item(),
        'wall': loss_wall.item(),
        'div': loss_div.item(),
        'out': loss_out.item(),
    }

    return total, loss_dict
