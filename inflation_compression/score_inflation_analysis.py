#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分膨胀 (Score Inflation) 与方差压缩 (Variance Compression) 实证分析
- 按周汇总评委单分（1–10 分），仅保留 score > 0 且非 N/A
- 量化：周均分、标准差、方差、变异系数 (CV)
- 显著性：周序与均分/方差的趋势检验
- 输出：汇总表、图表、统计检验结果（供 draft 引用）
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats

# 路径
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "2026_MCM_Problem_C_Data.csv"
FIG_DIR = BASE / "figures"
OUT_TABLE = BASE / "score_by_week_summary.csv"
OUT_STATS = BASE / "inflation_compression_stats.txt"

FIG_DIR.mkdir(exist_ok=True)

# 绘图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_judge_scores_long(path: Path) -> pd.DataFrame:
    """读取原始数据，将每一条评委单分展开为一行（仅 score>0 且非 NaN）。"""
    df = pd.read_csv(path, na_values=["N/A", "NA", ""], keep_default_na=True)
    pattern = re.compile(r"^week(\d+)_judge(\d+)_score$")
    rows = []
    for _, r in df.iterrows():
        season = r["season"]
        for c in df.columns:
            m = pattern.match(c)
            if not m:
                continue
            week, judge = int(m.group(1)), int(m.group(2))
            val = r[c]
            if pd.isna(val) or float(val) <= 0:
                continue
            rows.append({"season": season, "week": week, "judge": judge, "score": float(val)})
    return pd.DataFrame(rows)


def summarize_by_week(long_df: pd.DataFrame) -> pd.DataFrame:
    """按周汇总：n, mean, std, var, CV, sem。"""
    g = long_df.groupby("week")["score"]
    n = g.count()
    mean = g.mean()
    std = g.std()
    var = g.var()
    sem = g.sem()
    cv = (std / mean * 100).reindex(mean.index)  # CV %
    summary = pd.DataFrame({
        "week": n.index,
        "n_scores": n.values,
        "mean": mean.values,
        "std": std.values,
        "variance": var.values,
        "CV_pct": cv.values,
        "sem": sem.values,
    })
    return summary


def run_trend_tests(summary: pd.DataFrame) -> dict:
    """周序与均分、标准差的趋势检验（Spearman + 简单线性回归）。"""
    week = summary["week"].values
    mean = summary["mean"].values
    std = summary["std"].values
    var = summary["variance"].values

    r_mean, p_mean = stats.spearmanr(week, mean)
    r_std, p_std = stats.spearmanr(week, std)
    r_var, p_var = stats.spearmanr(week, var)

    # 简单 OLS：mean ~ week, std ~ week
    slope_mean, intercept_mean, r2_mean, p_slope_mean, se_slope = stats.linregress(week, mean)
    slope_std, intercept_std, r2_std, p_slope_std, _ = stats.linregress(week, std)

    return {
        "spearman_mean_r": r_mean,
        "spearman_mean_p": p_mean,
        "spearman_std_r": r_std,
        "spearman_std_p": p_std,
        "spearman_var_r": r_var,
        "spearman_var_p": p_var,
        "ols_mean_slope": slope_mean,
        "ols_mean_p": p_slope_mean,
        "ols_std_slope": slope_std,
        "ols_std_p": p_slope_std,
        "n_weeks": len(week),
    }


def plot_mean_by_week(summary: pd.DataFrame, out_path: Path) -> None:
    """图1：周均分随周序变化（得分膨胀）。"""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    weeks = summary["week"]
    means = summary["mean"]
    sems = summary["sem"]
    ax.errorbar(weeks, means, yerr=sems, fmt="o-", capsize=4, capthick=1.5, color="C0", label="Mean ± SEM")
    ax.set_xlabel("Week of season")
    ax.set_ylabel("Mean judge score (1–10)")
    ax.set_title("Score inflation: mean judge score by week")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_std_by_week(summary: pd.DataFrame, out_path: Path) -> None:
    """图2：周内标准差随周序变化（方差压缩）。"""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    weeks = summary["week"]
    stds = summary["std"]
    ax.plot(weeks, stds, "s-", color="C1", label="Std dev")
    ax.set_xlabel("Week of season")
    ax.set_ylabel("Std dev of judge scores")
    ax.set_title("Variance compression: standard deviation by week")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_by_week(long_df: pd.DataFrame, out_path: Path) -> None:
    """图3：每周得分分布（箱线图），展示膨胀+压缩。"""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    weeks_sorted = sorted(long_df["week"].unique())
    data = [long_df.loc[long_df["week"] == w, "score"].values for w in weeks_sorted]
    bp = ax.boxplot(data, positions=weeks_sorted, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.set_xlabel("Week of season")
    ax.set_ylabel("Judge score (1–10)")
    ax.set_title("Score distribution by week (ceiling effect)")
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_early_vs_late_hist(long_df: pd.DataFrame, out_path: Path) -> None:
    """图4：早期周 vs 后期周得分直方图对比。"""
    if not HAS_MPL:
        return
    early = long_df[long_df["week"] <= 3]["score"]
    late = long_df[long_df["week"] >= 7]["score"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].hist(early, bins=np.arange(0.5, 11.5, 1), edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Judge score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Early weeks (1–3)")
    axes[0].set_xlim(0, 11)
    axes[1].hist(late, bins=np.arange(0.5, 11.5, 1), edgecolor="black", alpha=0.7, color="C1")
    axes[1].set_xlabel("Judge score")
    axes[1].set_title("Late weeks (7–11)")
    axes[1].set_xlim(0, 11)
    fig.suptitle("Score distribution: early vs late season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stats_report(stats_dict: dict, out_path: Path) -> None:
    """把趋势检验结果写入文本，供 report 引用。"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Trend tests (weeks 1–11)\n")
        f.write("Score inflation (mean vs week): Spearman r = {:.4f}, p = {:.4f}\n".format(
            stats_dict["spearman_mean_r"], stats_dict["spearman_mean_p"]))
        f.write("Variance compression (std vs week): Spearman r = {:.4f}, p = {:.4f}\n".format(
            stats_dict["spearman_std_r"], stats_dict["spearman_std_p"]))
        f.write("Variance compression (var vs week): Spearman r = {:.4f}, p = {:.4f}\n".format(
            stats_dict["spearman_var_r"], stats_dict["spearman_var_p"]))
        f.write("OLS mean ~ week: slope = {:.4f}, p = {:.4f}\n".format(
            stats_dict["ols_mean_slope"], stats_dict["ols_mean_p"]))
        f.write("OLS std ~ week: slope = {:.4f}, p = {:.4f}\n".format(
            stats_dict["ols_std_slope"], stats_dict["ols_std_p"]))


def main():
    print("Loading judge-level scores (score > 0, non-NA)...")
    long_df = load_judge_scores_long(DATA_PATH)
    print(f"  Total valid scores: {len(long_df)}")

    summary = summarize_by_week(long_df)
    summary.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    print(f"  Summary table saved: {OUT_TABLE}")

    stats_dict = run_trend_tests(summary)
    save_stats_report(stats_dict, OUT_STATS)
    print(f"  Stats saved: {OUT_STATS}")

    if HAS_MPL:
        plot_mean_by_week(summary, FIG_DIR / "fig1_mean_by_week.png")
        plot_std_by_week(summary, FIG_DIR / "fig2_std_by_week.png")
        plot_distribution_by_week(long_df, FIG_DIR / "fig3_boxplot_by_week.png")
        plot_early_vs_late_hist(long_df, FIG_DIR / "fig4_early_vs_late_hist.png")
        print(f"  Figures saved in: {FIG_DIR}")
    else:
        print("  matplotlib not available, skipping figures.")

    return summary, stats_dict, long_df


if __name__ == "__main__":
    main()
