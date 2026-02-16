#!/usr/bin/env python3
"""汎化テストで精度が逆転する問題の診断スクリプト

問題:
    - Baseline (ibpm_h5_400x200) の RMSE=0.518, Energy Ratio=0.62
    - 汎化テスト (ibpm_h5_gen_v2_*) の RMSE=0.03, Energy Ratio=1.00
    - 未学習データの方が17倍高精度という矛盾

仮説:
    HDF5軸順序の不一致。BaselineはTN形式 (T, N, C, H, W)、
    汎化データはNT形式 (N, T, C, H, W) で格納されており、
    evaluate.py が data[0, :16] でアクセスする際に意味が異なる。

実験:
    1-A: 全データセットのHDF5軸順序を監査
    1-B: Baseline の軸順序を修正して再構成し、RMSE改善を確認
    1-C: 条件テンソルの影響度を検証（ablation）

Usage:
    # 診断のみ（GPU不要）
    python experiments/ibpm/diagnose_generalization.py --mode audit

    # 軸修正実験（GPU必要）
    python experiments/ibpm/diagnose_generalization.py --mode axis-fix

    # 条件ablation（GPU必要）
    python experiments/ibpm/diagnose_generalization.py --mode conditioning

    # 全実験実行
    python experiments/ibpm/diagnose_generalization.py --mode all
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sda.paths import get_data_dir


# ===========================================================================
# データパス定義（evaluate.py と同じ）
# ===========================================================================
GEOMETRY_DATA_PATHS = {
    "baseline": "ibpm_h5_400x200",
    "y_m01": "ibpm_h5_gen_v2_y_m01",
    "y_m02": "ibpm_h5_gen_v2_y_m02",
    "y_p02": "ibpm_h5_gen_v2_y_p02",
    "r_04": "ibpm_h5_gen_v2_r04",
    "r_06": "ibpm_h5_gen_v2_r06",
    "r_025": "ibpm_h5_gen_r025",
    "r_075": "ibpm_h5_gen_r075",
    "y_p10": "ibpm_h5_gen_y_p10",
    "y_m10": "ibpm_h5_gen_y_m10",
    "r_015": "ibpm_h5_gen_r015",
    "r_100": "ibpm_h5_gen_r100",
    "y_p20": "ibpm_h5_gen_y_p20",
    "y_m20": "ibpm_h5_gen_y_m20",
}

# fluid-sbi/data/ ディレクトリのルート
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


# ===========================================================================
# 実験 1-A: データ構造監査
# ===========================================================================
def audit_data_structures(output_dir: Path) -> dict:
    """全データセットのHDF5軸順序を監査

    各データセットについて:
    - shape と shape_description 属性を確認
    - フレーム間RMSE で時間的整合性を判定（TN vs NT 推定）

    Returns:
        監査結果の辞書
    """
    print("=" * 70)
    print("実験 1-A: データ構造監査")
    print("=" * 70)

    results = {}

    # ヘッダー
    print(f"\n{'Name':<12} {'Shape':<28} {'shape_description':<50} {'推定形式'}")
    print("-" * 120)

    for name, dir_name in GEOMETRY_DATA_PATHS.items():
        path = DATA_ROOT / dir_name / "test.h5"
        if not path.exists():
            print(f"{name:<12} --- データなし ({path})")
            continue

        with h5py.File(path, "r") as f:
            shape = f["x"].shape
            shape_desc = f["x"].attrs.get("shape_description", "属性なし")
            data = torch.from_numpy(f["x"][:]).float()

        # フレーム間RMSE で時間的整合性を判定
        # data[0, :16] (NTアクセス) vs data[:16, 0] (TNアクセス) の
        # 連続フレーム間差分が小さい方が「正しい時系列」
        n_frames = min(16, shape[0], shape[1])

        # パターンA: data[0, :n] → dim0=N, dim1=T のアクセス（NT形式前提）
        seq_nt = data[0, :n_frames]  # (n, C, H, W)
        diff_nt = torch.sqrt(torch.mean((seq_nt[1:] - seq_nt[:-1]) ** 2)).item()

        # パターンB: data[:n, 0] → dim0=T, dim1=N のアクセス（TN形式前提）
        seq_tn = data[:n_frames, 0]  # (n, C, H, W)
        diff_tn = torch.sqrt(torch.mean((seq_tn[1:] - seq_tn[:-1]) ** 2)).item()

        # 差分が小さい方が連続的な時系列（=正しいアクセス）
        if diff_nt < diff_tn:
            estimated = "NT (N,T,C,H,W)"
        else:
            estimated = "TN (T,N,C,H,W)"

        shape_str = str(shape)
        print(f"{name:<12} {shape_str:<28} {shape_desc:<50} {estimated} (diff_nt={diff_nt:.4f}, diff_tn={diff_tn:.4f})")

        results[name] = {
            "dir": dir_name,
            "shape": list(shape),
            "shape_description": shape_desc,
            "diff_nt": diff_nt,
            "diff_tn": diff_tn,
            "estimated_format": estimated,
        }

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー:")
    tn_datasets = [k for k, v in results.items() if "TN" in v["estimated_format"]]
    nt_datasets = [k for k, v in results.items() if "NT" in v["estimated_format"]]
    print(f"  TN形式（推定）: {tn_datasets}")
    print(f"  NT形式（推定）: {nt_datasets}")

    if tn_datasets and nt_datasets:
        print("\n  [問題検出] 混在する軸順序が発見されました!")
        print("  evaluate.py は data[0, :n_timesteps] で NT形式を前提にアクセスしています。")
        print("  TN形式のデータでは、異なるサンプルのスナップショットを")
        print("  「時系列」として扱うため、RMSEが大きくなります。")

    # JSON保存
    report_path = output_dir / "audit_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  結果を保存: {report_path}")

    return results


# ===========================================================================
# 実験 1-B: 軸修正実験
# ===========================================================================
def experiment_axis_fix(output_dir: Path, run_dir: Path = None) -> dict:
    """Baseline の軸順序を修正して再構成し、RMSE改善を確認

    - 方法A（現行）: data[0, :16] — TN形式なら非整合
    - 方法B（修正）: data[:16, 0] — TN形式なら正しい時系列

    両方で復元パイプラインを実行し、RMSE/Energy Ratio を比較。

    Returns:
        実験結果の辞書
    """
    from experiments.ibpm.utils import load_trained_model, compute_vorticity, plot_velocity_and_vorticity
    from sda.data.ibpm_dataset import IBPMNormalizer, build_cylinder_mask, build_inflow_profile
    from sda.score import VPSDE, GaussianScore
    from sda.paths import get_runs_dir

    print("\n" + "=" * 70)
    print("実験 1-B: 軸修正実験（核心）")
    print("=" * 70)

    # モデルロード
    if run_dir is None:
        runs_dir = get_runs_dir() / "ibpm"
        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = runs[0]

    print(f"  モデル: {run_dir.name}")
    score, config = load_trained_model(run_dir, device="cuda")
    window = config.get("window", 16)

    normalizer = IBPMNormalizer()

    # Baseline データをロード（生HDF5）
    baseline_path = DATA_ROOT / "ibpm_h5_400x200" / "test.h5"
    with h5py.File(baseline_path, "r") as f:
        raw = torch.from_numpy(f["x"][:]).float()
        shape_desc = f["x"].attrs.get("shape_description", "不明")

    print(f"  データshape: {raw.shape}")
    print(f"  shape_description: {shape_desc}")

    # 評価パラメータ
    n_timesteps = min(window, raw.shape[0], raw.shape[1])
    H, W = raw.shape[-2], raw.shape[-1]

    # 条件テンソル（evaluate.py と同じ設定）
    TRAIN_COND_CENTER = (63.5, 63.5)
    TRAIN_COND_RADIUS = 15.875
    cylinder_mask = build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS)
    inflow_profile = build_inflow_profile(H, W, U=1.0)
    cond = torch.stack([cylinder_mask, inflow_profile], dim=0).unsqueeze(0).cuda()

    # Clamped Linear Scaling パラメータ
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02
    sub = 4

    results = {}

    for method_name, x_star_raw in [
        ("A_current (data[0, :n])", raw[0, :n_timesteps]),   # NT前提アクセス
        ("B_fixed  (data[:n, 0])", raw[:n_timesteps, 0]),    # TN前提アクセス
    ]:
        print(f"\n  --- {method_name} ---")
        T, C = x_star_raw.shape[0], x_star_raw.shape[1]
        print(f"    x_star_raw shape: {x_star_raw.shape}")

        # フレーム間差分（時間的整合性の指標）
        frame_diff = torch.sqrt(torch.mean((x_star_raw[1:] - x_star_raw[:-1]) ** 2)).item()
        print(f"    フレーム間RMSE: {frame_diff:.6f}")

        # 正規化 & flatten
        x_star_norm = normalizer.normalize(x_star_raw)
        x_star_flat = x_star_norm.flatten(0, 1)

        # サブサンプリング & 再構成
        n_obs_ref = (H // 4) * (W // 4) * T * C
        n_obs = (H // sub) * (W // sub) * T * C
        ratio = n_obs / n_obs_ref
        std_scaled = max(BASE_STD * ratio, MIN_STD)
        gamma_scaled = max(BASE_GAMMA * (ratio ** 2), MIN_GAMMA)

        def A(x, s=sub):
            return x[..., ::s, ::s]

        y_star = torch.normal(A(x_star_flat), BASE_STD)

        sde = VPSDE(
            GaussianScore(
                y_star, A=A, std=std_scaled, gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape, eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]), c=cond, steps=256, corrections=1, tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        # メトリクス
        rmse = torch.sqrt(torch.mean((x_recon_norm - x_star_norm) ** 2)).item()
        energy_ratio = x_recon_norm.std().item() / x_star_norm.std().item()

        print(f"    RMSE: {rmse:.6f}")
        print(f"    Energy Ratio: {energy_ratio:.4f}")

        # 可視化
        x_recon_phys = normalizer.denormalize(x_recon_norm)
        x_gt_phys = normalizer.denormalize(x_star_norm)

        fig, axes = plt.subplots(2, min(8, T), figsize=(20, 6))
        if T == 1:
            axes = axes[:, None]
        for t in range(min(8, T)):
            w_gt = compute_vorticity(x_gt_phys[t:t+1])[0]
            w_recon = compute_vorticity(x_recon_phys[t:t+1])[0]
            vmax = max(abs(w_gt.min()), abs(w_gt.max()))
            axes[0, t].imshow(w_gt.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            axes[0, t].set_title(f"t={t}", fontsize=8)
            axes[0, t].axis("off")
            axes[1, t].imshow(w_recon.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            axes[1, t].axis("off")
        axes[0, 0].set_ylabel("GT")
        axes[1, 0].set_ylabel("Recon")

        safe_name = method_name.split("(")[0].strip()
        fig.suptitle(f"{method_name}\nRMSE={rmse:.4f}, Energy={energy_ratio:.3f}", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / f"axis_fix_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        results[safe_name] = {
            "rmse": rmse,
            "energy_ratio": energy_ratio,
            "frame_diff": frame_diff,
        }

    # 判定
    print("\n" + "=" * 70)
    rmse_a = results["A_current"]["rmse"]
    rmse_b = results["B_fixed"]["rmse"]
    if rmse_b < rmse_a * 0.5:
        print(f"  [確定] 軸順序修正により RMSE が大幅改善:")
        print(f"    修正前: {rmse_a:.4f} → 修正後: {rmse_b:.4f} ({rmse_b/rmse_a:.1%})")
        print(f"  → Baseline は TN形式 (T,N,C,H,W)。data[:n, 0] でアクセスすべき。")
    elif rmse_a < rmse_b * 0.5:
        print(f"  [確定] 現行アクセスが正しい（NT形式）:")
        print(f"    現行: {rmse_a:.4f}, 修正: {rmse_b:.4f}")
    else:
        print(f"  [不確定] 大きな差がない（別の原因の可能性）:")
        print(f"    現行: {rmse_a:.4f}, 修正: {rmse_b:.4f}")

    # JSON保存
    report_path = output_dir / "axis_fix_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  結果を保存: {report_path}")

    return results


# ===========================================================================
# 実験 1-C: 条件テンソル ablation
# ===========================================================================
def experiment_conditioning(output_dir: Path, run_dir: Path = None) -> dict:
    """条件テンソルの影響度を検証

    (a) 正常条件（学習時と同じ設定）
    (b) ゼロ条件（全チャネル0）
    (c) ランダム条件

    三者のRMSE差が小さければ、モデルは条件を無視している。

    Returns:
        実験結果の辞書
    """
    from experiments.ibpm.utils import load_trained_model, load_ibpm_data
    from sda.data.ibpm_dataset import IBPMNormalizer, build_cylinder_mask, build_inflow_profile
    from sda.score import VPSDE, GaussianScore
    from sda.paths import get_runs_dir

    print("\n" + "=" * 70)
    print("実験 1-C: 条件テンソル ablation")
    print("=" * 70)

    # モデルロード
    if run_dir is None:
        runs_dir = get_runs_dir() / "ibpm"
        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = runs[0]

    print(f"  モデル: {run_dir.name}")
    score, config = load_trained_model(run_dir, device="cuda")
    window = config.get("window", 16)

    normalizer = IBPMNormalizer()

    # 汎化データの1つを使用（NT形式の可能性が高く、RMSEが低い方）
    gen_data_path = DATA_ROOT / "ibpm_h5_gen_v2_y_m01"
    test_data = load_ibpm_data(gen_data_path, split="test")
    n_timesteps = min(window, test_data.shape[1])
    x_star_raw = test_data[0, :n_timesteps]
    T, C, H, W = x_star_raw.shape

    x_star_norm = normalizer.normalize(x_star_raw)
    x_star_flat = x_star_norm.flatten(0, 1)

    print(f"  テストデータ: ibpm_h5_gen_v2_y_m01, shape={x_star_raw.shape}")

    # 条件テンソルのバリエーション
    TRAIN_COND_CENTER = (63.5, 63.5)
    TRAIN_COND_RADIUS = 15.875

    conditions = {
        "normal": torch.stack([
            build_cylinder_mask(H, W, center=TRAIN_COND_CENTER, radius=TRAIN_COND_RADIUS),
            build_inflow_profile(H, W, U=1.0),
        ], dim=0).unsqueeze(0),
        "zero": torch.zeros(1, 2, H, W),
        "random": torch.randn(1, 2, H, W),
    }

    # サブサンプリングパラメータ
    sub = 4
    BASE_STD = 0.2
    BASE_GAMMA = 0.04
    MIN_STD = 0.15
    MIN_GAMMA = 0.02
    n_obs_ref = (H // 4) * (W // 4) * T * C
    n_obs = (H // sub) * (W // sub) * T * C
    ratio = n_obs / n_obs_ref
    std_scaled = max(BASE_STD * ratio, MIN_STD)
    gamma_scaled = max(BASE_GAMMA * (ratio ** 2), MIN_GAMMA)

    def A(x, s=sub):
        return x[..., ::s, ::s]

    y_star = torch.normal(A(x_star_flat), BASE_STD)

    results = {}

    for cond_name, cond_tensor in conditions.items():
        print(f"\n  --- 条件: {cond_name} ---")
        cond = cond_tensor.cuda()

        sde = VPSDE(
            GaussianScore(
                y_star, A=A, std=std_scaled, gamma=gamma_scaled,
                sde=VPSDE(score.kernel, shape=(), eta=0.01),
            ),
            shape=x_star_flat.shape, eta=0.01,
        ).cuda()

        x_recon_flat = sde.sample(
            torch.Size([1]), c=cond, steps=256, corrections=1, tau=0.5,
        ).cpu()[0]

        x_recon_norm = x_recon_flat.unflatten(0, (T, C))

        rmse = torch.sqrt(torch.mean((x_recon_norm - x_star_norm) ** 2)).item()
        energy_ratio = x_recon_norm.std().item() / x_star_norm.std().item()

        print(f"    RMSE: {rmse:.6f}")
        print(f"    Energy Ratio: {energy_ratio:.4f}")

        results[cond_name] = {
            "rmse": rmse,
            "energy_ratio": energy_ratio,
        }

    # 判定
    print("\n" + "=" * 70)
    rmse_normal = results["normal"]["rmse"]
    rmse_zero = results["zero"]["rmse"]
    rmse_random = results["random"]["rmse"]
    max_diff = max(abs(rmse_normal - rmse_zero), abs(rmse_normal - rmse_random))

    if max_diff < 0.05:
        print(f"  [発見] 条件テンソルの影響は微小（最大差={max_diff:.4f}）")
        print(f"  → モデルは条件をほぼ無視している可能性")
    else:
        print(f"  [発見] 条件テンソルは結果に影響あり（最大差={max_diff:.4f}）")
        print(f"    normal={rmse_normal:.4f}, zero={rmse_zero:.4f}, random={rmse_random:.4f}")

    # JSON保存
    report_path = output_dir / "conditioning_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  結果を保存: {report_path}")

    return results


# ===========================================================================
# メイン
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="汎化テスト精度逆転問題の診断")
    parser.add_argument(
        "--mode", type=str, default="audit",
        choices=["audit", "axis-fix", "conditioning", "all"],
        help="実行モード",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="学習済みモデルのディレクトリ")
    parser.add_argument("--output-dir", type=Path, default=None, help="出力ディレクトリ")
    args = parser.parse_args()

    # 出力ディレクトリ
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).resolve().parents[3] / "sda" / "results" / "ibpm" / f"diagnose_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {args.output_dir}")

    all_results = {}

    if args.mode in ("audit", "all"):
        all_results["audit"] = audit_data_structures(args.output_dir)

    if args.mode in ("axis-fix", "all"):
        all_results["axis_fix"] = experiment_axis_fix(args.output_dir, args.run_dir)

    if args.mode in ("conditioning", "all"):
        all_results["conditioning"] = experiment_conditioning(args.output_dir, args.run_dir)

    # 全結果を保存
    report_path = args.output_dir / "diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n全結果を保存: {report_path}")


if __name__ == "__main__":
    main()
