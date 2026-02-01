#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3 Improved V4: Compare Strategy A (Lagged Judge), Strategy B (Binary Survival),
and A+B together. Four variants: Baseline, A-only, B-only, A+B.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
CLEAN_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
LONG_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_long.csv"
OUT_DIR = BASE / "output"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def group_industry(ind):
    if pd.isna(ind):
        return "Other"
    s = str(ind).lower()
    if any(x in s for x in ["actor", "actress", "film", "movie"]):
        return "Actor"
    if any(x in s for x in ["singer", "rapper", "musician", "pop", "country"]):
        return "Musician"
    if any(x in s for x in ["athlete", "nfl", "nba", "olympic", "football", "basketball", "baseball", "skater", "gymnast"]):
        return "Athlete"
    if any(x in s for x in ["reality", "bachelor", "survivor", "housewife", "tv personality"]):
        return "Reality TV"
    if any(x in s for x in ["host", "anchor", "presenter"]):
        return "Host"
    if any(x in s for x in ["comedian", "comic"]):
        return "Comedian"
    if "model" in s and "social" not in s:
        return "Model"
    if any(x in s for x in ["youtube", "social media", "influencer", "tiktok"]):
        return "Social Media"
    return "Other"


def load_weekly_all():
    """Week-level: judge_z, lag_judge_z, fan_proxy_z, survived_next, Age, Industry_Group, ballroom_partner."""
    clean = pd.read_csv(CLEAN_PATH)
    long = pd.read_csv(LONG_PATH)
    meta = clean[["celebrity_name", "season", "ballroom_partner", "celebrity_industry", "celebrity_age_during_season"]].copy()
    meta = meta.rename(columns={"celebrity_age_during_season": "Age", "celebrity_industry": "celebrity_industry"})
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Industry_Group"] = meta["celebrity_industry"].apply(group_industry)
    long["normalized"] = pd.to_numeric(long["normalized"], errors="coerce")
    long = long.dropna(subset=["normalized"])
    df = long[["celebrity_name", "season", "week", "normalized"]].merge(meta, on=["celebrity_name", "season"], how="left")
    df = df.dropna(subset=["Age", "Industry_Group", "ballroom_partner"])
    df["ballroom_partner"] = df["ballroom_partner"].astype(str).str.strip()
    df["Industry_Group"] = df["Industry_Group"].astype(str)
    df["judge_z"] = df.groupby("season")["normalized"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df = df.sort_values(["celebrity_name", "season", "week"])
    df["lag_judge_z"] = df.groupby(["celebrity_name", "season"])["judge_z"].shift(1)
    max_weeks = df.groupby("season")["week"].max().reset_index().rename(columns={"week": "max_weeks"})
    df = df.merge(max_weeks, on="season")
    df["survival_rate"] = df["week"] / df["max_weeks"]
    ols_proxy = smf.ols("survival_rate ~ judge_z", data=df).fit()
    df["fan_proxy"] = ols_proxy.resid
    df["fan_proxy_z"] = (df["fan_proxy"] - df["fan_proxy"].mean()) / (df["fan_proxy"].std() + 1e-9)
    max_week_per_cs = df.groupby(["celebrity_name", "season"])["week"].max().reset_index().rename(columns={"week": "last_week"})
    df = df.merge(max_week_per_cs, on=["celebrity_name", "season"])
    df["survived_next"] = (df["week"] < df["last_week"]).astype(int)
    return df


def fit_ols_partner_lsdv(formula, data):
    model = smf.ols(formula, data=data).fit()
    params = model.params
    partner_coefs = {}
    for k, v in params.items():
        if "ballroom_partner" in str(k):
            try:
                name = str(k).split("[T.")[1].split("]")[0]
                partner_coefs[name] = float(v)
            except Exception:
                continue
    vals = np.array(list(partner_coefs.values())) if partner_coefs else np.array([0.0])
    centered = {k: v - float(np.mean(vals)) for k, v in partner_coefs.items()}
    return model, centered


def extract_industry_effects(model):
    out = {}
    for k, v in model.params.items():
        if "Industry_Group" in str(k):
            try:
                name = str(k).split("[T.")[1].split("]")[0]
                out[name] = float(v)
            except Exception:
                continue
    return out


def pseudo_r2_logit(m):
    """McFadden pseudo R² = 1 - (llf / llf_null)."""
    llf = m.llf
    # null model: intercept only
    try:
        y = m.model.endog
        n1 = float(np.sum(y))
        n0 = float(len(y) - n1)
        llf_null = n1 * np.log(n1 / len(y)) + n0 * np.log(n0 / len(y))
    except Exception:
        llf_null = 0.0
    if llf_null >= 0 or llf >= 0:
        return np.nan
    return 1.0 - (llf / llf_null)


def main():
    print("Problem 3 Improved V4: Strategy A (Lagged) vs B (Binary Survival) vs A+B")
    df = load_weekly_all()
    df_with_lag = df.dropna(subset=["lag_judge_z"]).copy()
    n_full = len(df)
    n_lag = len(df_with_lag)
    print(f"  Week-level full: N = {n_full}, with lag (week>=2): N = {n_lag}")

    base_form_judge = "judge_z ~ Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    aug_form_judge = "judge_z ~ lag_judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    base_form_fan = "fan_proxy_z ~ Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    surv_form = "survived_next ~ judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"

    results = {}

    # ---------- Baseline (no A, no B) ----------
    m_judge_base, _ = fit_ols_partner_lsdv(base_form_judge, df)
    m_fan_base, _ = fit_ols_partner_lsdv(base_form_fan, df)
    results["Baseline"] = {
        "judge": m_judge_base,
        "fan": m_fan_base,
        "fan_type": "proxy_z",
        "r2_judge": m_judge_base.rsquared,
        "r2_fan": m_fan_base.rsquared,
        "age_judge": m_judge_base.params.get("Age", np.nan),
        "age_fan": m_fan_base.params.get("Age", np.nan),
        "n_judge": len(df),
        "n_fan": len(df),
    }
    print(f"  Baseline: R² Judge = {results['Baseline']['r2_judge']:.4f}, R² Fan = {results['Baseline']['r2_fan']:.4f}")

    # ---------- Strategy A only (Lagged Judge; Fan = proxy) ----------
    m_judge_a, _ = fit_ols_partner_lsdv(aug_form_judge, df_with_lag)
    m_fan_a, _ = fit_ols_partner_lsdv(base_form_fan, df)
    results["Strategy A only"] = {
        "judge": m_judge_a,
        "fan": m_fan_a,
        "fan_type": "proxy_z",
        "r2_judge": m_judge_a.rsquared,
        "r2_fan": m_fan_a.rsquared,
        "age_judge": m_judge_a.params.get("Age", np.nan),
        "age_fan": m_fan_a.params.get("Age", np.nan),
        "lag_coef": m_judge_a.params.get("lag_judge_z", np.nan),
        "n_judge": len(df_with_lag),
        "n_fan": len(df),
    }
    print(f"  Strategy A only: R² Judge = {results['Strategy A only']['r2_judge']:.4f}, R² Fan = {results['Strategy A only']['r2_fan']:.4f}")

    # ---------- Strategy B only (Judge = base; Fan = Logit(survived_next)) ----------
    m_judge_b, _ = fit_ols_partner_lsdv(base_form_judge, df)
    m_fan_b = Logit.from_formula(surv_form, data=df).fit(disp=0)
    pseudo_r2_b = pseudo_r2_logit(m_fan_b)
    results["Strategy B only"] = {
        "judge": m_judge_b,
        "fan": m_fan_b,
        "fan_type": "logit",
        "r2_judge": m_judge_b.rsquared,
        "r2_fan": pseudo_r2_b,
        "age_judge": m_judge_b.params.get("Age", np.nan),
        "age_fan": m_fan_b.params.get("Age", np.nan),
        "n_judge": len(df),
        "n_fan": len(df),
    }
    print(f"  Strategy B only: R² Judge = {results['Strategy B only']['r2_judge']:.4f}, Pseudo-R² Fan = {pseudo_r2_b:.4f}")

    # ---------- Strategy A+B ----------
    m_judge_ab, _ = fit_ols_partner_lsdv(aug_form_judge, df_with_lag)
    m_fan_ab = Logit.from_formula(surv_form, data=df).fit(disp=0)
    pseudo_r2_ab = pseudo_r2_logit(m_fan_ab)
    results["Strategy A+B"] = {
        "judge": m_judge_ab,
        "fan": m_fan_ab,
        "fan_type": "logit",
        "r2_judge": m_judge_ab.rsquared,
        "r2_fan": pseudo_r2_ab,
        "age_judge": m_judge_ab.params.get("Age", np.nan),
        "age_fan": m_fan_ab.params.get("Age", np.nan),
        "lag_coef": m_judge_ab.params.get("lag_judge_z", np.nan),
        "n_judge": len(df_with_lag),
        "n_fan": len(df),
    }
    print(f"  Strategy A+B: R² Judge = {results['Strategy A+B']['r2_judge']:.4f}, Pseudo-R² Fan = {pseudo_r2_ab:.4f}")

    # ----- Save comparison table -----
    comp_rows = []
    for name, r in results.items():
        comp_rows.append({
            "variant": name,
            "r2_judge": r["r2_judge"],
            "r2_fan_or_pseudo": r["r2_fan"],
            "age_coef_judge": r["age_judge"],
            "age_coef_fan": r["age_fan"],
            "fan_type": r["fan_type"],
            "n_judge": r["n_judge"],
            "n_fan": r["n_fan"],
        })
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "comparison_all_variants.csv", index=False)
    print("  comparison_all_variants.csv")

    # ----- Industry effects from A+B (judge + fan/survival) for high-impact figure -----
    ie_judge = extract_industry_effects(results["Strategy A+B"]["judge"])
    m_fan_ab = results["Strategy A+B"]["fan"]
    ie_fan = {}
    for k, v in m_fan_ab.params.items():
        if "Industry_Group" in str(k):
            try:
                name = str(k).split("[T.")[1].split("]")[0]
                ie_fan[name] = float(v)
            except Exception:
                continue
    all_ind = sorted(set(ie_judge.keys()) | set(ie_fan.keys()))
    pd.DataFrame([{"Industry": k, "Judge_Effect": ie_judge.get(k, np.nan), "Survival_Effect": ie_fan.get(k, np.nan)} for k in all_ind]).to_csv(OUT_DIR / "industry_effects_ab.csv", index=False)

    # ----- Figures -----
    # 1. R² comparison (Judge and Fan) across four variants
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    variants = list(results.keys())
    r2j = [results[v]["r2_judge"] for v in variants]
    r2f = [results[v]["r2_fan"] for v in variants]
    colors = ["#7f7f7f", "#4c72b0", "#c44e52", "#55a868"]
    axes[0].bar(variants, r2j, color=colors[:len(variants)], alpha=0.85)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Judge Model: Baseline vs A vs B vs A+B")
    axes[0].set_ylim(0, max(r2j) * 1.15 if r2j else 1)
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(variants, r2f, color=colors[:len(variants)], alpha=0.85)
    axes[1].set_ylabel("R² or Pseudo-R²")
    axes[1].set_title("Fan/Survival Model: Baseline vs A vs B vs A+B")
    axes[1].set_ylim(0, max(r2f) * 1.15 if r2f else 1)
    axes[1].tick_params(axis="x", rotation=18)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_r2_comparison_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_r2_comparison_variants.png")

    # 2. Age coefficient: Judge vs Fan across variants
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(variants))
    w = 0.35
    age_j = [results[v]["age_judge"] for v in variants]
    age_f = [results[v]["age_fan"] for v in variants]
    ax.bar(x - w/2, age_j, w, label="Judge", color="#4c72b0", alpha=0.85)
    ax.bar(x + w/2, age_f, w, label="Fan/Survival", color="#c44e52", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18)
    ax.set_ylabel("Age Coefficient")
    ax.set_title("Age Effect: Judge vs Fan/Survival (All Variants)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_age_coef_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_age_coef_variants.png")

    # 3. Industry bias scatter: Judge Effect vs Survival Effect (A+B) — "Reality TV = low score, high survival"
    if all_ind:
        j_vals = [ie_judge.get(i, np.nan) for i in all_ind]
        f_vals = [ie_fan.get(i, np.nan) for i in all_ind]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(j_vals, f_vals, s=120, alpha=0.85)
        for i, ind in enumerate(all_ind):
            ax.annotate(ind, (j_vals[i], f_vals[i]), textcoords="offset points", xytext=(6, 4), fontsize=10)
        ax.axhline(0, color="grey", linestyle="--")
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_xlabel("Effect on Judge Score (vs Actor)")
        ax.set_ylabel("Effect on Survival Probability (vs Actor)")
        ax.set_title("Industry Bias: Reality TV = Low Score, High Survival (Strategy A+B)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig3_industry_scatter_judge_vs_survival.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig3_industry_scatter_judge_vs_survival.png")

    # 4. Bar chart industry effects (A+B): Judge vs Survival
    if all_ind:
        df_ind = pd.DataFrame({"Industry": all_ind, "Judge": j_vals, "Survival": f_vals})
        df_ind = df_ind.melt(id_vars="Industry", var_name="Model", value_name="Effect")
        fig, ax = plt.subplots(figsize=(10, max(4, len(all_ind) * 0.35)))
        sns.barplot(data=df_ind, x="Industry", y="Effect", hue="Model", ax=ax, palette=["#4c72b0", "#c44e52"])
        ax.set_title("Industry Effects (Strategy A+B): Judge vs Survival")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.axhline(0, color="black")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig4_industry_bars_ab.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig4_industry_bars_ab.png")

    # 5. Composite for paper
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(variants, r2j, color=colors[:len(variants)], alpha=0.85)
    ax1.set_ylabel("R²")
    ax1.set_title("Judge Model Fit")
    ax1.tick_params(axis="x", rotation=18)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(variants, r2f, color=colors[:len(variants)], alpha=0.85)
    ax2.set_ylabel("R² or Pseudo-R²")
    ax2.set_title("Fan/Survival Model Fit")
    ax2.tick_params(axis="x", rotation=18)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w/2, age_j, w, label="Judge", color="#4c72b0", alpha=0.85)
    ax3.bar(x + w/2, age_f, w, label="Fan/Survival", color="#c44e52", alpha=0.85)
    ax3.axhline(0, color="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(variants, rotation=18)
    ax3.set_ylabel("Age Coefficient")
    ax3.set_title("Age Effect")
    ax3.legend()
    ax4 = fig.add_subplot(gs[1, 1])
    if all_ind:
        ax4.scatter(j_vals, f_vals, s=100, alpha=0.85)
        for i, ind in enumerate(all_ind):
            ax4.annotate(ind, (j_vals[i], f_vals[i]), textcoords="offset points", xytext=(4, 4), fontsize=9)
        ax4.axhline(0, color="grey", linestyle="--")
        ax4.axvline(0, color="grey", linestyle="--")
        ax4.set_xlabel("Judge Effect")
        ax4.set_ylabel("Survival Effect")
        ax4.set_title("Industry: Judge vs Survival (A+B)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prob3_v4_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  prob3_v4_analysis_plot.png")

    # ----- Summary text -----
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Problem 3 Improved V4: Strategy Comparison\n")
        f.write("Baseline: Judge ~ Age+Industry+Partner, Fan ~ fan_proxy_z (same)\n")
        f.write("Strategy A only: Judge ~ lag_judge_z + ..., Fan = proxy (same)\n")
        f.write("Strategy B only: Judge = base, Fan = Logit(survived_next ~ judge_z + ...)\n")
        f.write("Strategy A+B: Judge with lag, Fan = Logit(survived_next ~ ...)\n")
        for name, r in results.items():
            f.write(f"\n{name}: R² Judge = {r['r2_judge']:.4f}, R²/Pseudo-R² Fan = {r['r2_fan']:.4f}\n")
    print("  summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
