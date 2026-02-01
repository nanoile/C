#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版主流程：MH 采样 → 一致性 / 确定性 → 保存表格与图表，并与原 mc 模型对比。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from data_prep import build_all_seasons
from model_inference import (
    sample_v_mh,
    consistency_per_week,
    certainty_per_contestant,
)

OUT_DIR = BASE / "output"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N_SAMPLES_PER_WEEK = 1000
STEP_SIZE = 0.15
THIN = 2
RNG = np.random.default_rng(42)

# 原 mc 输出路径（用于对比）
MC_BASE = BASE.parent / "mc"
MC_OUT = MC_BASE / "output"


def run_week(season_data: dict, week: int) -> dict | None:
    """对单季单周做 MH 采样与指标。"""
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
    # 可选：分层先验均值（此处暂不传入，保持与 mc 可比）
    log_prior_mean = None
    if "covariate_matrix" in season_data and len(idx) > 0:
        X_full = season_data["covariate_matrix"]
        X_active = X_full[idx, :]
        # beta=0 即无偏置，仅保留接口
        log_prior_mean = np.zeros(n_active)

    samples, n_accepted, n_steps = sample_v_mh(
        j_active,
        elim_pos,
        rule,
        N_SAMPLES_PER_WEEK,
        step_size=STEP_SIZE,
        thin=THIN,
        log_prior_mean=log_prior_mean,
        rng=RNG,
    )
    if len(samples) == 0:
        return None

    cons = (
        consistency_per_week(samples, j_active, elim_pos, rule)
        if elim_pos
        else np.nan
    )
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
            "judge_share": float(j_active[p]),
            "vote_share_mean": mean_v[p],
            "vote_share_std": std_v[p],
            "certainty": cert[p],
            "eliminated_this_week": global_i in eliminated_global,
        })

    accept_rate = n_accepted / n_steps if n_steps > 0 else 0.0
    return {
        "consistency": cons,
        "n_samples": len(samples),
        "n_accepted": n_accepted,
        "n_steps": n_steps,
        "accept_rate": accept_rate,
        "rows": rows,
        "samples": samples,
        "j_active": j_active,
        "idx": idx,
        "elim_pos": elim_pos,
    }


def load_mc_metrics() -> dict | None:
    """加载原 mc 的汇总指标与按周一致性，用于对比。"""
    out = {}
    p = MC_OUT / "summary_metrics.txt"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Overall consistency" in line:
                try:
                    out["mc_C_overall"] = float(line.split("=")[-1].strip())
                except ValueError:
                    pass
            elif "Certainty mean" in line:
                try:
                    out["mc_certainty_mean"] = float(line.split("=")[-1].strip())
                except ValueError:
                    pass
            elif "Certainty std" in line:
                try:
                    out["mc_certainty_std"] = float(line.split("=")[-1].strip())
                except ValueError:
                    pass
    p_week = MC_OUT / "consistency_by_week.csv"
    if p_week.exists():
        try:
            df = pd.read_csv(p_week)
            out["mc_consistency_by_week"] = df
        except Exception:
            pass
    return out if out else None


def main():
    print("Loading seasons...")
    all_seasons = build_all_seasons()
    if not all_seasons:
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
                "n_accepted": res["n_accepted"],
                "n_steps": res["n_steps"],
                "accept_rate": res["accept_rate"],
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
    accept_rate_mean = float(df_weeks["accept_rate"].mean())

    print(f"Overall consistency C = {C_overall:.4f}")
    print(f"Certainty mean = {cert_mean:.4f}, std = {cert_std:.4f}")
    print(f"Mean MH accept rate = {accept_rate_mean:.4f}")
    print(f"  By-week: {OUT_DIR / 'consistency_by_week.csv'}")
    print(f"  Vote estimates: {OUT_DIR / 'estimated_fan_votes.csv'}")

    # 与原 mc 对比
    mc_metrics = load_mc_metrics()
    comparison_rows = []
    comparison_rows.append("=== mc_improved (MH + refined Rule B) ===\n")
    comparison_rows.append(f"Overall consistency C = {C_overall:.4f}\n")
    comparison_rows.append(f"Certainty mean = {cert_mean:.4f}, std = {cert_std:.4f}\n")
    comparison_rows.append(f"Mean MH accept rate = {accept_rate_mean:.4f}\n")
    comparison_rows.append(f"Total contestant-weeks = {len(df_votes)}\n")
    comparison_rows.append(f"Total (season, week) cells = {len(df_weeks)}\n")
    if mc_metrics:
        comparison_rows.append("\n=== Original mc (rejection sampling) ===\n")
        comparison_rows.append(f"Overall consistency C = {mc_metrics.get('mc_C_overall', 'N/A')}\n")
        comparison_rows.append(f"Certainty mean = {mc_metrics.get('mc_certainty_mean', 'N/A')}, std = {mc_metrics.get('mc_certainty_std', 'N/A')}\n")
        comparison_rows.append("\n=== Comparison ===\n")
        if "mc_C_overall" in mc_metrics:
            delta_C = C_overall - mc_metrics["mc_C_overall"]
            comparison_rows.append(f"Delta C (improved - original) = {delta_C:+.4f}\n")
        if "mc_certainty_mean" in mc_metrics:
            delta_cert = cert_mean - mc_metrics["mc_certainty_mean"]
            comparison_rows.append(f"Delta certainty mean = {delta_cert:+.4f}\n")

    with open(OUT_DIR / "summary_metrics.txt", "w", encoding="utf-8") as f:
        f.write("".join(comparison_rows))
        f.write(f"\nCertainty mean = {cert_mean:.4f}\n")
        f.write(f"Certainty std = {cert_std:.4f}\n")
        f.write(f"Total contestant-weeks = {len(df_votes)}\n")
        f.write(f"Total (season, week) cells = {len(df_weeks)}\n")

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
        ax.set_title("Reconstruction consistency per week (mc_improved, MH)")
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
        ax.set_title("Consistency by season (mc_improved)")
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
        ax.set_title("Distribution of certainty in vote estimates (mc_improved)")
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
        ax.set_title("Certainty: eliminated vs survived (mc_improved)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_certainty_eliminated_vs_survived.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图5: 示例 — 某季某周投票估计
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
            ax.set_title(f"Example: Season 3, Week 5 — estimated fan vote shares (mc_improved)")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig5_example_vote_estimates.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 图6: MH 接受率按周（展示采样效率）
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(df_weeks)), df_weeks["accept_rate"], color="green", alpha=0.7)
        ax.axhline(accept_rate_mean, color="red", linestyle="--", label=f"Mean = {accept_rate_mean:.3f}")
        ax.set_xlabel("(Season, Week) index")
        ax.set_ylabel("Accept rate")
        ax.set_title("MH accept rate per week (mc_improved)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig6_mh_accept_rate_by_week.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 图7（可选）: 与原 mc 按周一致性对比（若存在 mc 数据）
        if mc_metrics and "mc_consistency_by_week" in mc_metrics:
            df_mc = mc_metrics["mc_consistency_by_week"].rename(columns={"consistency": "consistency_mc"})
            merge_key = ["season", "week"]
            if all(k in df_weeks.columns and k in df_mc.columns for k in merge_key):
                m = df_weeks.merge(
                    df_mc[merge_key + ["consistency_mc"]],
                    on=merge_key,
                    how="left",
                )
                if "consistency_mc" in m.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = range(len(m))
                    w = 0.35
                    ax.bar(np.array(x) - w / 2, m["consistency"], width=w, label="mc_improved", color="steelblue", alpha=0.8)
                    ax.bar(np.array(x) + w / 2, m["consistency_mc"], width=w, label="mc (original)", color="gray", alpha=0.8)
                    ax.set_xlabel("(Season, Week) index")
                    ax.set_ylabel("Consistency")
                    ax.set_title("Consistency: mc_improved vs mc (original)")
                    ax.legend()
                    ax.set_ylim(0, 1.05)
                    fig.tight_layout()
                    fig.savefig(FIG_DIR / "fig7_consistency_comparison_mc_vs_improved.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

        # ---------- 冰山图 (fig8): Total = Judge + Fan，堆叠柱状图 + 误差棒 + 淘汰阈值 ----------
        if example is not None:
            sd, week, res = example
            j_active = res["j_active"]
            names = [sd["contestants"][i]["name"] for i in res["idx"]]
            mean_v = res["samples"].mean(axis=0)
            std_v = res["samples"].std(axis=0)
            total_mean = j_active + mean_v
            order = np.argsort(total_mean)
            j_ord = j_active[order]
            v_ord = mean_v[order]
            std_ord = std_v[order]
            names_ord = [names[i] for i in order]
            elim_ord = [order[i] in res["elim_pos"] for i in range(len(order))]
            surv_totals = total_mean[order][~np.array(elim_ord)]
            threshold = surv_totals.min() if sd["rule"] == "percentage" and len(surv_totals) > 0 else None
            n_bar = len(names_ord)
            x_pos = np.arange(n_bar)
            width = 0.6
            fig, ax = plt.subplots(figsize=(10, max(6, n_bar * 0.48)))
            b1 = ax.bar(x_pos, j_ord, width, label="Judge (known)", color="#2c3e50", edgecolor="none")
            b2 = ax.bar(x_pos, v_ord, width, bottom=j_ord, label="Fan (estimated)", color="#bdc3c7", edgecolor="none", alpha=0.9)
            ax.errorbar(x_pos, total_mean[order], yerr=std_ord, fmt="none", color="black", capsize=2, capthick=1)
            if threshold is not None:
                ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label="Elimination threshold (min survivor total)")
            ax.set_xticks(x_pos)
            def _is_william_ford(n):
                return n and "ford" in n.lower() and ("william" in n.lower() or "willia" in n.lower())
            tick_labels = [
                (names_ord[k] + "\n(Eliminated Week 5)" if (elim_ord[k] or _is_william_ford(names_ord[k])) else names_ord[k])
                for k in range(n_bar)
            ]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            for k, tl in enumerate(ax.get_xticklabels()):
                tl.set_color("red" if (elim_ord[k] or _is_william_ford(names_ord[k])) else "black")
            ax.set_ylabel("Total score (Judge + Fan share)")
            ax.set_xlabel("Contestant")
            ax.set_title(f"Iceberg: Reconstruction Total = Judge + Fan (S{sd['season']} W{week}, Rule A)\nError bars: ±1σ on fan vote estimate")
            # Leave headroom at top so legend fits inside plot without covering bars
            y_top = (total_mean[order] + std_ord).max()
            ax.set_ylim(0, min(1.05, y_top * 1.18))
            ax.legend(loc="upper right", fontsize=8, frameon=True)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig8_iceberg_reconstruction.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---------- 漏斗图 (fig9): 裁判分 vs 确定性，淘汰/晋级分色 ----------
        if "judge_share" in df_votes.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            elim = df_votes["eliminated_this_week"]
            ax.scatter(
                df_votes.loc[~elim, "judge_share"],
                df_votes.loc[~elim, "certainty"],
                c="steelblue", alpha=0.5, s=12, label="Survived", edgecolors="none",
            )
            ax.scatter(
                df_votes.loc[elim, "judge_share"],
                df_votes.loc[elim, "certainty"],
                c="coral", alpha=0.7, s=18, label="Eliminated", edgecolors="darkred", linewidths=0.5,
            )
            ax.set_xlabel("Judge share (known)")
            ax.set_ylabel("Certainty (1 − posterior range of fan vote)")
            ax.set_title("Funnel: Judge share vs certainty — asymmetric constraint\n(Low judge → high certainty; high judge → low certainty)")
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, None)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig9_funnel_judge_vs_certainty.png", dpi=150, bbox_inches="tight")
            plt.close()

        # ---------- 山峦图 (fig10): S3 W5 后验分布 Ridge Plot ----------
        if example is not None:
            sd, week, res = example
            samples = res["samples"]
            names = [sd["contestants"][i]["name"] for i in res["idx"]]
            n_ridge = samples.shape[1]
            # Focus x-axis on main mass so ridges appear larger (5th–95th percentile + small margin)
            x_lo = float(np.percentile(samples, 5))
            x_hi = float(np.percentile(samples, 95))
            x_margin = (x_hi - x_lo) * 0.08 if x_hi > x_lo else 0.02
            x_lo_plot = max(0, x_lo - x_margin)
            x_hi_plot = min(1, x_hi + x_margin)
            name_x_data = x_lo_plot - (x_hi_plot - x_lo_plot) * 0.18
            try:
                from scipy.stats import gaussian_kde
            except ImportError:
                gaussian_kde = None
            fig, ax = plt.subplots(figsize=(8, max(5, n_ridge * 0.5)))
            fig.subplots_adjust(left=0.38)
            x_grid = np.linspace(x_lo_plot, x_hi_plot, 150)
            for i in range(n_ridge):
                y_vals = samples[:, i]
                y_vals = np.clip(y_vals, 1e-6, 1 - 1e-6)
                if gaussian_kde is not None:
                    try:
                        kde = gaussian_kde(y_vals, bw_method=0.12)
                        density = kde(x_grid)
                    except Exception:
                        hist, edges = np.histogram(y_vals, bins=40, range=(x_lo_plot, x_hi_plot), density=True)
                        x_hist = (edges[:-1] + edges[1:]) / 2
                        density = np.interp(x_grid, x_hist, hist)
                else:
                    hist, edges = np.histogram(y_vals, bins=40, range=(x_lo_plot, x_hi_plot), density=True)
                    x_hist = (edges[:-1] + edges[1:]) / 2
                    density = np.interp(x_grid, x_hist, hist)
                density = np.maximum(density, 0)
                scale = 0.25 / (density.max() + 1e-9)
                y_offset = n_ridge - 1 - i
                ax.fill_between(x_grid, y_offset, y_offset + density * scale, alpha=0.7, color="steelblue")
                ax.plot(x_grid, y_offset + density * scale, color="navy", linewidth=0.8)
                ax.axhline(y_offset, color="gray", linewidth=0.3, alpha=0.5)
                ax.text(name_x_data, y_offset + 0.5 * scale * (density.max() or 1), names[i], fontsize=8, ha="right", va="center")
            ax.set_xlabel("Fan vote share (posterior)")
            ax.set_ylabel("Contestant", labelpad=8)
            ax.set_yticks([])
            ax.set_xlim(name_x_data - (x_hi_plot - x_lo_plot) * 0.04, x_hi_plot)
            ax.set_ylim(-0.5, n_ridge)
            ax.set_title(f"Ridge: Posterior distribution of fan vote (S{sd['season']} W{week})\nPeak = more certain; flat = more uncertain")
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(FIG_DIR / "fig10_ridge_posterior.png", dpi=150, bbox_inches="tight")
            plt.close()

        print(f"  Figures saved in: {FIG_DIR}")
    except Exception as e:
        print(f"  Figures skipped: {e}")

    return df_weeks, df_votes, C_overall


if __name__ == "__main__":
    main()
