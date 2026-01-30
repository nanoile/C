#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM+EM 主流程：用 GMM 建模人气 α，EM 从接受的 α 样本更新先验；
最终采样得到投票后验，计算一致性 C 与确定性，输出到 gaussian/output 与 figures。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
# 使用 mc 的数据准备
sys.path.insert(0, str(BASE.parent))
from mc.data_prep import build_all_seasons

from gmm_em_inference import (
    sample_v_via_gmm,
    fit_gmm_em,
    consistency_per_week,
    certainty_per_contestant,
)

OUT_DIR = BASE / "output"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_SAMPLES_PER_WEEK = 800
N_SAMPLES_EM_ROUND = 400  # 每周采样数（EM 更新用，少一点以加速）
N_EM_ITERATIONS = 2
N_GMM_COMPONENTS = 3
RNG = np.random.default_rng(42)


def get_cleaned_path():
    p = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
    if not p.exists():
        p = BASE.parent / "2026_MCM_Problem_C_Data_cleaned.csv"
    return p


def run_week(
    season_data: dict,
    week: int,
    gmm_weights: np.ndarray,
    gmm_means: np.ndarray,
    gmm_stds: np.ndarray,
    n_samples: int = N_SAMPLES_PER_WEEK,
) -> dict | None:
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
    v_samples, alpha_samples, attempts = sample_v_via_gmm(
        j_active, elim_pos, rule,
        gmm_weights, gmm_means, gmm_stds,
        n_samples, rng=RNG,
    )
    if len(v_samples) == 0:
        return None

    cons = consistency_per_week(v_samples, j_active, elim_pos, rule) if elim_pos else np.nan
    cert = certainty_per_contestant(v_samples)
    mean_v = v_samples.mean(axis=0)
    std_v = v_samples.std(axis=0)

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
        "n_samples": len(v_samples),
        "n_attempts": attempts,
        "rows": rows,
        "samples": v_samples,
        "alpha_samples": alpha_samples,
        "j_active": j_active,
        "idx": idx,
        "elim_pos": elim_pos,
    }


def main():
    print("Loading seasons...")
    clean_path = get_cleaned_path()
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {clean_path}")
    all_seasons = build_all_seasons(cleaned_path=clean_path)
    print(f"  Loaded {len(all_seasons)} seasons.")

    # 初始化 GMM：用一批“平坦”α 得到初始 α 池（从 N(0,2) 采样，经约束接受）
    print("Warm-up: collecting initial alpha pool for GMM init...")
    alpha_pool = []
    warm_weights = np.ones(N_GMM_COMPONENTS) / N_GMM_COMPONENTS
    warm_means = np.array([-1.0, 0.0, 1.0])
    warm_stds = np.array([1.0, 1.0, 1.0])
    for sd in all_seasons[:3]:  # 前几季做 warm-up
        for week in sd["weeks"]:
            res = run_week(sd, week, warm_weights, warm_means, warm_stds)
            if res and res.get("alpha_samples") is not None and len(res["alpha_samples"]) > 0:
                alpha_pool.append(res["alpha_samples"].ravel())
    if alpha_pool:
        alpha_flat = np.concatenate(alpha_pool)
        if len(alpha_flat) >= N_GMM_COMPONENTS * 10:
            gmm_weights, gmm_means, gmm_stds = fit_gmm_em(
                alpha_flat, N_GMM_COMPONENTS, max_iter=50, rng=RNG
            )
            print(f"  Initial GMM: weights={gmm_weights}, means={gmm_means}, stds={gmm_stds}")
        else:
            gmm_weights, gmm_means, gmm_stds = warm_weights, warm_means, warm_stds
    else:
        gmm_weights, gmm_means, gmm_stds = warm_weights, warm_means, warm_stds

    # EM 迭代：每轮采样（用较少样本加速），用接受的 α 更新 GMM
    for em_iter in range(N_EM_ITERATIONS):
        alpha_pool = []
        for sd in all_seasons:
            for week in sd["weeks"]:
                res = run_week(sd, week, gmm_weights, gmm_means, gmm_stds, n_samples=N_SAMPLES_EM_ROUND)
                if res and res.get("alpha_samples") is not None and len(res["alpha_samples"]) > 0:
                    alpha_pool.append(res["alpha_samples"].ravel())
        if alpha_pool:
            alpha_flat = np.concatenate(alpha_pool)
            gmm_weights, gmm_means, gmm_stds = fit_gmm_em(
                alpha_flat, N_GMM_COMPONENTS, max_iter=50, rng=RNG
            )
            print(f"  EM iter {em_iter + 1}/{N_EM_ITERATIONS}: means={gmm_means}, stds={gmm_stds}")

    # 最终一轮：用收敛后的 GMM 采样，汇总一致性与确定性
    all_rows = []
    week_results = []
    for sd in all_seasons:
        for week in sd["weeks"]:
            res = run_week(sd, week, gmm_weights, gmm_means, gmm_stds)
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

    df_with_elim = df_weeks[df_weeks["consistency"].notna()]
    C_overall = df_with_elim["consistency"].mean() if len(df_with_elim) > 0 else 0.0
    C_by_season = df_weeks.groupby("season")["consistency"].agg(
        lambda x: x.dropna().mean() if x.notna().any() else np.nan
    )

    df_weeks.to_csv(OUT_DIR / "consistency_by_week.csv", index=False, encoding="utf-8-sig")
    df_votes.to_csv(OUT_DIR / "estimated_fan_votes.csv", index=False, encoding="utf-8-sig")
    C_by_season.to_csv(OUT_DIR / "consistency_by_season.csv", encoding="utf-8-sig")

    cert_mean = float(df_votes["certainty"].mean())
    cert_std = float(df_votes["certainty"].std())
    with open(OUT_DIR / "summary_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"GMM+EM (K={N_GMM_COMPONENTS}, EM_iters={N_EM_ITERATIONS})\n")
        f.write(f"Overall consistency C = {C_overall:.4f}\n")
        f.write(f"Certainty mean = {cert_mean:.4f}\n")
        f.write(f"Certainty std = {cert_std:.4f}\n")
        f.write(f"Total contestant-weeks = {len(df_votes)}\n")
        f.write(f"Total (season, week) cells = {len(df_weeks)}\n")
        f.write(f"Final GMM: weights={gmm_weights}, means={gmm_means}, stds={gmm_stds}\n")

    print(f"Overall consistency C = {C_overall:.4f}")
    print(f"Certainty mean = {cert_mean:.4f} (std = {cert_std:.4f})")
    print(f"  Output: {OUT_DIR}")

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(df_weeks))
        ax.bar(x, df_weeks["consistency"], color="steelblue", alpha=0.8)
        ax.axhline(C_overall, color="red", linestyle="--", label=f"Mean C = {C_overall:.3f}")
        ax.set_xlabel("(Season, Week) index")
        ax.set_ylabel("Consistency")
        ax.set_title("GMM+EM: Reconstruction consistency per week")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig1_consistency_by_week.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        seasons = C_by_season.index
        ax.bar(seasons, C_by_season.values, color="teal", alpha=0.8)
        ax.axhline(C_overall, color="red", linestyle="--", label=f"Overall = {C_overall:.3f}")
        ax.set_xlabel("Season")
        ax.set_ylabel("Mean consistency")
        ax.set_title("GMM+EM: Consistency by season")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig2_consistency_by_season.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_votes["certainty"], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Certainty (1 - range of vote share)")
        ax.set_ylabel("Count (contestant-weeks)")
        ax.set_title("GMM+EM: Distribution of certainty")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_certainty_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        elim = df_votes[df_votes["eliminated_this_week"]]["certainty"]
        surv = df_votes[~df_votes["eliminated_this_week"]]["certainty"]
        ax.hist([surv, elim], bins=25, label=["Survived", "Eliminated"], alpha=0.7)
        ax.set_xlabel("Certainty")
        ax.set_ylabel("Count")
        ax.set_title("GMM+EM: Certainty — eliminated vs survived")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_certainty_eliminated_vs_survived.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 示例：某季某周投票估计
        example = None
        for sd in all_seasons:
            if sd["season"] != 3 or 5 not in sd["weeks"]:
                continue
            res = run_week(sd, 5, gmm_weights, gmm_means, gmm_stds)
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
            ax.set_title("GMM+EM: Example vote estimates (Season 3, Week 5)")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig5_example_vote_estimates.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Figures: {FIG_DIR}")
    except Exception as e:
        print(f"  Figures skipped: {e}")

    return df_weeks, df_votes, C_overall


if __name__ == "__main__":
    main()
