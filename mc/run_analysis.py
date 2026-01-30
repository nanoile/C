#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主流程：加载数据 → 按 (season, week) 采样潜在观众投票 → 计算一致性 C 与确定性 → 保存表格与图表。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from data_prep import build_all_seasons
from model_inference import (
    sample_v_rejection,
    consistency_per_week,
    certainty_per_contestant,
)

# 输出目录
OUT_DIR = BASE / "output"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_SAMPLES_PER_WEEK = 1000
RNG = np.random.default_rng(42)


def run_week(season_data: dict, week: int) -> dict | None:
    """对单季单周做采样与指标。"""
    wd = season_data["week_data"].get(week)
    if not wd:
        return None
    idx = wd["idx"]
    j_full = wd["j"]
    eliminated_global = wd["eliminated"]

    j_active = j_full[idx]
    n_active = len(idx)
    elim_pos = [idx.index(i) for i in eliminated_global if i in idx]
    if n_active <= 1:
        return None

    rule = season_data["rule"]
    samples, attempts = sample_v_rejection(
        j_active, elim_pos, rule, N_SAMPLES_PER_WEEK, rng=RNG
    )
    if len(samples) == 0:
        return None

    cons = consistency_per_week(samples, j_active, elim_pos, rule) if elim_pos else np.nan
    cert = certainty_per_contestant(samples)
    mean_v = samples.mean(axis=0)
    std_v = samples.std(axis=0)

    contestants = season_data["contestants"]
    rows = []
    for p, global_i in enumerate(idx):
        name = contestants[global_i]["name"]
        rows.append({
            "season": season_data["season"],
            "week": week,
            "celebrity_name": name,
            "vote_share_mean": mean_v[p],
            "vote_share_std": std_v[p],
            "certainty": cert[p],
            "eliminated_this_week": global_i in eliminated_global,
        })
    return {
        "consistency": cons,
        "n_samples": len(samples),
        "n_attempts": attempts,
        "rows": rows,
        "samples": samples,
        "j_active": j_active,
        "idx": idx,
        "elim_pos": elim_pos,
    }


def main():
    print("Loading seasons...")
    all_seasons = build_all_seasons()
    if not all_seasons:
        # 若 clean 不在 mc 上级，尝试直接用父目录的 cleaned
        from data_prep import CLEAN_PATH, BASE
        alt = BASE / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
        if not alt.exists():
            alt = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
        if alt.exists():
            all_seasons = build_all_seasons(cleaned_path=alt)
        if not all_seasons:
            raise FileNotFoundError("No cleaned CSV found. Run data_cleaning first.")
    print(f"  Loaded {len(all_seasons)} seasons.")

    all_rows = []
    week_results = []
    for sd in all_seasons:
        for week in sd["weeks"]:
            res = run_week(sd, week)
            if res is None:
                continue
            week_results.append({
                "season": sd["season"],
                "week": week,
                "rule": sd["rule"],
                "consistency": res["consistency"],
                "n_samples": res["n_samples"],
                "n_attempts": res["n_attempts"],
            })
            all_rows.extend(res["rows"])

    df_weeks = pd.DataFrame(week_results)
    df_votes = pd.DataFrame(all_rows)

    # 整体一致性（仅统计有淘汰的周）
    df_with_elim = df_weeks[df_weeks["consistency"].notna()]
    C_overall = df_with_elim["consistency"].mean() if len(df_with_elim) > 0 else 0.0
    C_by_season = df_weeks.groupby("season")["consistency"].agg(lambda x: x.dropna().mean() if x.notna().any() else np.nan)

    df_weeks.to_csv(OUT_DIR / "consistency_by_week.csv", index=False, encoding="utf-8-sig")
    df_votes.to_csv(OUT_DIR / "estimated_fan_votes.csv", index=False, encoding="utf-8-sig")
    C_by_season.to_csv(OUT_DIR / "consistency_by_season.csv", encoding="utf-8-sig")

    print(f"Overall consistency C = {C_overall:.4f}")
    print(f"  By-week stats saved: {OUT_DIR / 'consistency_by_week.csv'}")
    print(f"  Vote estimates saved: {OUT_DIR / 'estimated_fan_votes.csv'}")

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False

        # 图1: 每周一致性
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(df_weeks))
        ax.bar(x, df_weeks["consistency"], color="steelblue", alpha=0.8)
        ax.axhline(C_overall, color="red", linestyle="--", label=f"Mean C = {C_overall:.3f}")
        ax.set_xlabel("(Season, Week) index")
        ax.set_ylabel("Consistency (match rate)")
        ax.set_title("Reconstruction consistency per week")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig1_consistency_by_week.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图2: 按季平均一致性
        fig, ax = plt.subplots(figsize=(8, 4))
        seasons = C_by_season.index
        ax.bar(seasons, C_by_season.values, color="teal", alpha=0.8)
        ax.axhline(C_overall, color="red", linestyle="--", label=f"Overall = {C_overall:.3f}")
        ax.set_xlabel("Season")
        ax.set_ylabel("Mean consistency")
        ax.set_title("Consistency by season")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig2_consistency_by_season.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图3: 确定性分布
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_votes["certainty"], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Certainty (1 - range of vote share)")
        ax.set_ylabel("Count (contestant-weeks)")
        ax.set_title("Distribution of certainty in vote estimates")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_certainty_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图4: 确定性 vs 是否被淘汰
        fig, ax = plt.subplots(figsize=(5, 4))
        elim = df_votes[df_votes["eliminated_this_week"]]["certainty"]
        surv = df_votes[~df_votes["eliminated_this_week"]]["certainty"]
        ax.hist([surv, elim], bins=25, label=["Survived", "Eliminated"], alpha=0.7)
        ax.set_xlabel("Certainty")
        ax.set_ylabel("Count")
        ax.set_title("Certainty: eliminated vs survived")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_certainty_eliminated_vs_survived.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图5: 示例—某季某周投票估计（后验均值 ± 1 std）
        example = None
        for sd in all_seasons:
            if sd["season"] != 3 or 5 not in sd["weeks"]:
                continue
            res = run_week(sd, 5)
            if res is None:
                continue
            example = (sd, 5, res)
            break
        if example is not None:
            sd, week, res = example
            names = [sd["contestants"][i]["name"] for i in res["idx"]]
            mean_v = res["samples"].mean(axis=0)
            std_v = res["samples"].std(axis=0)
            y = range(len(names))
            fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.35)))
            ax.barh(y, mean_v, xerr=std_v, capsize=2, color="coral", alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Estimated vote share (posterior mean ± 1 std)")
            ax.set_title(f"Example: Season 3, Week 5 — estimated fan vote shares")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig5_example_vote_estimates.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Figures saved in: {FIG_DIR}")
    except Exception as e:
        print(f"  Figures skipped: {e}")

    # 汇总统计写入
    with open(OUT_DIR / "summary_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall consistency C = {C_overall:.4f}\n")
        f.write(f"Certainty mean = {df_votes['certainty'].mean():.4f}\n")
        f.write(f"Certainty std = {df_votes['certainty'].std():.4f}\n")
        f.write(f"Total contestant-weeks = {len(df_votes)}\n")
        f.write(f"Total (season, week) cells = {len(df_weeks)}\n")

    return df_weeks, df_votes, C_overall


if __name__ == "__main__":
    main()
