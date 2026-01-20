import os
import re
from typing import Sequence, Optional, Union, Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
import numpy as np
from math import sqrt

metric_of_interest = ['correct_rate', 'mean', 'score', 'fts']

def simplify_model_name(model_name):
    if 'nonlatent' in model_name:
        return 'nonlatent model'
    elif 'true' in model_name:
        return 'model with latent on'
    elif 'false' in model_name:
        return 'model with latent off'

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson 95% CI for binomial proportion.
    返回 (lower, upper) 均在 [0,1] 范围内。
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    lower, upper = max(0.0, centre - margin), min(1.0, centre + margin)
    return lower, upper

# 2) 新增 t-based CI 函数（允许结果超出 [0,1]）
def t_ci(mu: float, std: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Student-t based two-sided CI for the mean.
    返回 (lower, upper)。当 n <= 1 时返回 (mu, mu)。
    如果 scipy.stats.t 不可用则退回正态近似（z=1.96）。
    """
    if n <= 1 or np.isnan(mu):
        return mu, mu
    se = std / np.sqrt(n)
    try:
        from scipy.stats import t
        tval = float(t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        # fallback to normal approx
        tval = 1.96
    return mu - tval * se, mu + tval * se

def compute_binned_summary_and_smoothing(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int = 10,
    unique_threshold: int = 12,
    spline_min_points: int = 4,
    smoothing_points: int = 400,
    seed_for_debug: int = 42,
    mode: str = 'difference',  # 'proportion' (原行为) 或 'difference'（对比两个模型时的差值）
    alpha: float = 0.05,       # 置信度阈值（用于 t-CI，默认 95%）
) -> Dict[str, Any]:
    """
    将样本 (x, y) 分箱并计算每个箱的统计量（n, k, p, ci_low, ci_high）、Pareto 上界，并尝试平滑曲线。
    返回字典包含：
      - 'summary': pd.DataFrame with columns ['bin_center','n','k','p','ci_low','ci_high','pareto']
      - 'xx': np.ndarray (用于平滑曲线的 x)
      - 'yy': np.ndarray (用于平滑曲线的 y)
      - 'x': 原始 x (np.ndarray)
      - 'y': 原始 y (np.ndarray)
      - 'meta': dict 含一些中间参数（如 bins、centers 等）
    行为和你原代码一致（当唯一长度较少时每个长度为一箱，否则使用自适应等距 nbins）。
    """
    # 参数检查 / 复制输入为 numpy 数组
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # empty handling
    if x.size == 0 or y.size == 0:
        summary = pd.DataFrame(columns=['bin_center', 'n', 'k', 'p', 'ci_low', 'ci_high', 'pareto'])
        return {'summary': summary, 'xx': np.array([]), 'yy': np.array([]), 'x': x, 'y': y, 'meta': {}}

    unique_lengths = np.unique(x)
    meta: Dict[str, Any] = {}
    # 分箱逻辑
    if len(unique_lengths) <= unique_threshold:
        # 每个长度为一组
        bin_center = x.copy()
        plot_df = pd.DataFrame({'bin_center': bin_center, 'y': y})
        grp = plot_df.groupby('bin_center')
        meta['bins'] = None
        meta['centers'] = np.sort(np.unique(bin_center))
    else:
        # 自适应等距 nbins
        min_l, max_l = x.min(), x.max()
        if max_l == min_l:
            bins = np.array([min_l - 0.5, max_l + 0.5])
        else:
            bins = np.linspace(min_l, max_l, nbins + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0
        labels = centers
        # pd.cut -> categories 为中心点（float），include_lowest 保证边界包含
        binned = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
        plot_df = pd.DataFrame({'bin_center': binned.astype(float), 'y': y})
        grp = plot_df.groupby('bin_center')
        meta['bins'] = bins
        meta['centers'] = centers

    # 统计每箱 n,k,p,ci
    summary_rows = []
    for center, g in grp:
        # center 有可能为 NaN（如果某些 x 落在 NaN bin），跳过
        if pd.isna(center):
            continue

        vals = g['y'].values.astype(float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)

        if mode == 'proportion':
            # 原始行为（假设 y 为 0/1）
            k = int(np.nansum(vals))
            p = k / n if n > 0 else np.nan
            lo, hi = wilson_ci(k, n)
            summary_rows.append((float(center), n, k, p, lo, hi))
        elif mode == 'difference':
            # 新增行为：y 为样本级差值（可为负或 >1）
            # k 保留为 sum(vals) 以兼容下游（但语义与 proportion 不同）
            k = float(np.nansum(vals)) if n > 0 else np.nan
            mu = float(np.nanmean(vals)) if n > 0 else np.nan
            std = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
            lo, hi = t_ci(mu, std, n, alpha=alpha)
            # 为了保持与原代码列结构一致，这里仍将 mean 放到 'p' 列
            summary_rows.append((float(center), n, k, mu, lo, hi))
        else:
            raise ValueError("mode must be 'proportion' or 'difference'")
    summary = pd.DataFrame(summary_rows, columns=['bin_center', 'n', 'k', 'p', 'ci_low', 'ci_high'])
    summary = summary.sort_values('bin_center').reset_index(drop=True)

    # Pareto upper envelope（对 p 做 cummax）
    if not summary.empty:
        summary['pareto'] = summary['p'].cummax().fillna(method='ffill').fillna(0.0)
    else:
        summary['pareto'] = pd.Series(dtype=float)

    # 平滑趋势：先尝试 UnivariateSpline（带权重），失败则退回 polyfit
    xx = np.array([])
    yy = np.array([])
    try:
        from scipy.interpolate import UnivariateSpline  # optional dependency
        mask = ~np.isnan(summary['p'].values)
        x_s = summary['bin_center'].values[mask]
        y_s = summary['p'].values[mask]
        w = np.sqrt(np.maximum(summary['n'].values[mask], 1.0))
        if len(x_s) >= spline_min_points:
            # s 参数调节平滑，保守设置以避免过拟合（与原代码取法类似）
            s_val = max(1.0, len(x_s) * 0.5)
            spline = UnivariateSpline(x_s, y_s, w=w, s=s_val)
            xx = np.linspace(summary['bin_center'].min(), summary['bin_center'].max(), smoothing_points)
            yy = spline(xx)
        else:
            # too few points for spline -> 跳到 except 分支以使用 polyfit 回退
            raise RuntimeError("too few points for UnivariateSpline")
    except Exception:
        # fallback to polynomial fit degree 2/3 if点数合适
        xp = summary['bin_center'].values
        yp = summary['p'].values
        mask = (~np.isnan(xp)) & (~np.isnan(yp))
        xp, yp = xp[mask], yp[mask]
        if len(xp) >= 3:
            deg = 2 if len(xp) < 6 else 3
            coef = np.polyfit(xp, yp, deg)
            poly = np.poly1d(coef)
            xx = np.linspace(xp.min(), xp.max(), smoothing_points)
            yy = poly(xx)
        else:
            xx = np.array([])
            yy = np.array([])

    return {'summary': summary, 'xx': xx, 'yy': yy, 'x': x, 'y': y, 'meta': meta}


# =========================
# 绘图逻辑：接收 compute 的结果并绘图保存
# =========================
def plot_pareto_frontier_from_summary(
    summary: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    title: str = "",
    x_col: str = "output_length",
    y_col: str = "metric",
    out_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 200,
    jitter_scale: float = 0.4,
    rng_seed: int = 42,
) -> plt.Figure:
    """
    根据 summary(含 pareto) 和原始样本 (x,y) 绘制三段图：
      - 主图：抖动散点 + binned mean + Wilson 95% CI + 平滑趋势 + Pareto 上界
      - 次图：每箱样本数柱状图
      - rug：样本分布短竖线

    如果 out_path 提供则会保存图片（自动创建目录），并在结束时关闭 figure。
    返回 matplotlib.figure.Figure（调用者可选择保存或进一步处理）。
    """
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.4], hspace=0.05)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_count = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_rug = fig.add_subplot(gs[2, 0], sharex=ax_main)

    # 主图：原始抖动散点（0/1）
    rng = np.random.default_rng(seed=rng_seed)
    jitter_x = x + rng.normal(scale=jitter_scale, size=len(x))
    jitter_y = y
    ax_main.scatter(jitter_x, jitter_y, s=18, alpha=0.18, marker='o', label='raw samples (jittered)')

    # binned means + Wilson CI
    if summary is not None and not summary.empty:
        lower_err = (summary['p'] - summary['ci_low']).fillna(0.0).astype(float).to_numpy()
        upper_err = (summary['ci_high'] - summary['p']).fillna(0.0).astype(float).to_numpy()
        lower_err = np.maximum(lower_err, 0.0)
        upper_err = np.maximum(upper_err, 0.0)
        yerr = np.vstack([lower_err, upper_err])
        ax_main.errorbar(summary['bin_center'].to_numpy(), summary['p'].to_numpy(),
                         yerr=yerr, fmt='o', capsize=3, markersize=6, label='binned mean (Wilson 95% CI)')

    # 平滑趋势线（若存在）
    if xx is not None and len(xx) > 0:
        ax_main.plot(xx, yy, linewidth=2.2, label='smoothed trend')

    # Pareto 上界曲线
    if summary is not None and not summary.empty:
        # 使用虚线红色（与原版一致）
        ax_main.plot(summary['bin_center'], summary['pareto'], linestyle='--', linewidth=2.0, color='red', label='Pareto upper envelope')

    ax_main.set_ylabel(y_col)
    y_min = min(yy.min(), y.min()) - 0.05
    y_max = max(yy.max(), y.max()) + 0.05
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_title(title)
    ax_main.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax_main.legend(loc='lower right', fontsize='small')

    # 下方：每箱样本数柱形图
    if summary is not None and not summary.empty:
        centers = summary['bin_center'].values
        # 计算 bar width：基于相邻中心距离的中位数
        if len(centers) <= 1:
            bar_width = 1.0
        else:
            widths = np.diff(centers)
            medw = np.median(np.abs(widths))
            bar_width = medw * 0.9 if medw > 0 else 1.0
        ax_count.bar(centers, summary['n'].to_numpy(), width=bar_width, align='center')
    ax_count.set_ylabel('count')
    ax_count.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    # 最下方：rug plot（样本分布）
    ax_rug.plot(x, np.zeros_like(x), '|', markersize=8, alpha=0.6)
    ax_rug.set_yticks([])
    ax_rug.set_xlabel(x_col)

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None  # 已保存且关闭图像，返回 None 表示图已写入文件
    else:
        return fig  # 返回 figure，调用者负责显示或保存

def eval_single_pair(task_name, input_dir, output_dir, model_A, model_B):
    try:
        df_A = pd.read_csv(os.path.join(dir, 'eval_results_' + model_A + '.csv'))
        df_B = pd.read_csv(os.path.join(dir, 'eval_results_' + model_B + '.csv'))
    except:
        return
    
    x_col = 'output_length'
    y_col = next(col for col in df_A.columns if col in metric_of_interest)
    
    plot_df_A = df_A[[x_col, y_col]]
    plot_df_B = df_B[[x_col, y_col]]

    plot_df_A, plot_df_B = plot_df_A.align(plot_df_B, join='inner', axis=0)

    invalid_mask = (
        plot_df_A.isna().any(axis=1) |
        plot_df_B.isna().any(axis=1)
    )

    plot_df_A = plot_df_A[~invalid_mask]
    plot_df_B = plot_df_B[~invalid_mask]

    diff_df = plot_df_A.copy()
    diff_df[x_col] = plot_df_A[x_col] - plot_df_B[x_col]
    diff_df[y_col] = plot_df_A[y_col] - plot_df_B[y_col]
    
    # clamp delta x to [-1000, 1000]
    diff_df[x_col] = np.clip(diff_df[x_col], -1000, 1000)
    
    # clip rows that delta x out of [-1000, 1000]
    # diff_df = diff_df[(diff_df[x_col] >= -1000) & (diff_df[x_col] <= 1000)]
    
    x = diff_df[x_col].astype(float).to_numpy()
    y = diff_df[y_col].astype(float).to_numpy()
    
    compute_res = compute_binned_summary_and_smoothing(x, y, nbins=20)
    
    title = f"{simplify_model_name(model_A)} vs \n{simplify_model_name(model_B)} on \n{task_name}"
    # file name too long lol
    out_path = os.path.join(output_dir, task_name, f"{simplify_model_name(model_A)}__{simplify_model_name(model_B)}.png")
    # 注意：plot_pareto_frontier_from_summary 在提供 out_path 时会保存并关闭 figure（返回 None）
    plot_pareto_frontier_from_summary(
        summary=compute_res['summary'],
        x=compute_res['x'],
        y=compute_res['y'],
        xx=compute_res['xx'],
        yy=compute_res['yy'],
        title=title,
        x_col=f"delta_{x_col}",
        y_col=f"delta_{y_col}",
        out_path=out_path,
    )
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_A', type=str, required=True)
    parser.add_argument('--model_B', type=str, required=True)
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    
    banned_tasks = [
        'drd',
        'gsk',
        'jnk',
        'nepp'
    ]
    
    all_task_names = []
    all_task_dirs = []
    for csv_file in root_dir.rglob('*.csv'):
        subtask_path = csv_file.parent
        subtask_name = subtask_path.name
        
        if not subtask_name in all_task_names:
            all_task_names.append(subtask_name)
            all_task_dirs.append(subtask_path)
    
    for task, dir in zip(all_task_names, all_task_dirs):
        if task in banned_tasks:
            continue
        eval_single_pair(task, dir, args.output_dir, args.model_A, args.model_B)
