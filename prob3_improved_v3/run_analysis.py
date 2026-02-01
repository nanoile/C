#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3 Improved V3: Autoregressive Judge + Survival Framework.
- Judge model: Judge_Score_t ~ Judge_Score_{t-1} + Age + Partner + Industry (momentum).
- Survival model: Survival_Rate ~ Judge_Score_avg + Age + Partner + Industry (contestant-season).
- Targets: R² Judge ~0.60, R² Survival ~0.50; age reversal (judge negative, survival positive).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
CLEAN_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
LONG_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_long.csv"
V2_OUT_DIR = BASE.parent / "prob3_improved_v2" / "output"
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


def load_judge_weekly_with_lag():
    """Week-level data with judge_z and lag_judge_z (previous week same contestant-season)."""
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
    df = df.dropna(subset=["lag_judge_z"])
    return df


def load_survival_contestant_season():
    """Contestant-season level: survival_rate = weeks_survived / total_weeks, mean_judge_z, Age, Partner, Industry."""
    clean = pd.read_csv(CLEAN_PATH)
    long = pd.read_csv(LONG_PATH)
    meta = clean[["celebrity_name", "season", "ballroom_partner", "celebrity_industry", "celebrity_age_during_season"]].copy()
    meta = meta.rename(columns={"celebrity_age_during_season": "Age", "celebrity_industry": "celebrity_industry"})
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Industry_Group"] = meta["celebrity_industry"].apply(group_industry)
    long["normalized"] = pd.to_numeric(long["normalized"], errors="coerce")
    long = long.dropna(subset=["normalized"])
    long["judge_z"] = long.groupby("season")["normalized"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    agg = long.groupby(["celebrity_name", "season"]).agg(
        weeks_survived=("week", "max"),
        mean_judge_z=("judge_z", "mean"),
    ).reset_index()
    max_weeks = long.groupby("season")["week"].max().reset_index().rename(columns={"week": "total_weeks"})
    agg = agg.merge(max_weeks, on="season")
    agg["survival_rate"] = agg["weeks_survived"] / agg["total_weeks"]
    agg = agg.merge(meta, on=["celebrity_name", "season"], how="left")
    agg = agg.dropna(subset=["Age", "Industry_Group", "ballroom_partner"])
    agg["ballroom_partner"] = agg["ballroom_partner"].astype(str).str.strip()
    agg["Industry_Group"] = agg["Industry_Group"].astype(str)
    return agg


def fit_ols_partner_lsdv(formula, data):
    """OLS with partner fixed effects; return model and centered partner effects."""
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


def main():
    print("Problem 3 Improved V3: Autoregressive Judge + Survival Model")
    df_judge = load_judge_weekly_with_lag()
    df_surv = load_survival_contestant_season()
    print(f"  Judge (week-level with lag): N = {len(df_judge)}")
    print(f"  Survival (contestant-season): N = {len(df_surv)}")

    # Judge model: judge_z ~ lag_judge_z + Age + Industry + Partner
    formula_judge = "judge_z ~ lag_judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    m_judge, pe_judge = fit_ols_partner_lsdv(formula_judge, df_judge)
    r2_judge = m_judge.rsquared
    age_judge = m_judge.params.get("Age", np.nan)
    se_age_judge = m_judge.bse.get("Age", np.nan)
    lag_coef = m_judge.params.get("lag_judge_z", np.nan)
    ie_judge = extract_industry_effects(m_judge)
    var_judge = np.var(list(pe_judge.values())) if pe_judge else 0.0

    # Survival model: survival_rate ~ mean_judge_z + Age + Industry + Partner
    formula_surv = "survival_rate ~ mean_judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    m_surv, pe_surv = fit_ols_partner_lsdv(formula_surv, df_surv)
    r2_surv = m_surv.rsquared
    age_surv = m_surv.params.get("Age", np.nan)
    se_age_surv = m_surv.bse.get("Age", np.nan)
    judge_coef_surv = m_surv.params.get("mean_judge_z", np.nan)
    ie_surv = extract_industry_effects(m_surv)
    var_surv = np.var(list(pe_surv.values())) if pe_surv else 0.0

    print(f"  Judge model R² = {r2_judge:.4f}, age coef = {age_judge:.4f}, lag coef = {lag_coef:.4f}")
    print(f"  Survival model R² = {r2_surv:.4f}, age coef = {age_surv:.4f}, judge coef = {judge_coef_surv:.4f}")

    # ----- Save tables -----
    pd.DataFrame([{
        "model": "Judge",
        "r2": r2_judge,
        "age_coef": age_judge,
        "se_age": se_age_judge,
        "lag_judge_z_coef": lag_coef,
        "var_partner": var_judge,
    }, {
        "model": "Survival",
        "r2": r2_surv,
        "age_coef": age_surv,
        "se_age": se_age_surv,
        "mean_judge_z_coef": judge_coef_surv,
        "var_partner": var_surv,
    }]).to_csv(OUT_DIR / "v3_summary.csv", index=False)

    partners = sorted(set(pe_judge.keys()) | set(pe_surv.keys()))
    pd.DataFrame([{
        "Partner": p,
        "Judge_Effect": pe_judge.get(p, 0),
        "Survival_Effect": pe_surv.get(p, 0),
    } for p in partners]).to_csv(OUT_DIR / "v3_partner_effects.csv", index=False)
    df_pe = pd.DataFrame([{"Partner": p, "Judge_Effect": pe_judge.get(p, 0), "Survival_Effect": pe_surv.get(p, 0)} for p in partners])

    all_ind = sorted(set(ie_judge.keys()) | set(ie_surv.keys()))
    pd.DataFrame([{
        "Industry": k,
        "Judge": ie_judge.get(k, np.nan),
        "Survival": ie_surv.get(k, np.nan),
    } for k in all_ind]).to_csv(OUT_DIR / "v3_industry_effects.csv", index=False)

    # Comparison with V2 (read V2 summary if present)
    comparison = [{
        "pipeline": "V3 (Judge lag + Survival)",
        "r2_judge": r2_judge,
        "r2_fan_or_survival": r2_surv,
        "age_coef_judge": age_judge,
        "age_coef_fan_or_survival": age_surv,
        "n_judge": len(df_judge),
        "n_survival": len(df_surv),
    }]
    if V2_OUT_DIR.exists() and (V2_OUT_DIR / "v2_summary.csv").exists():
        v2 = pd.read_csv(V2_OUT_DIR / "v2_summary.csv")
        r2j_v2 = v2[v2["type"] == "Judge"]["r2"].iloc[0]
        r2f_v2 = v2[v2["type"] == "Fan"]["r2"].iloc[0]
        agej_v2 = v2[v2["type"] == "Judge"]["age_coef"].iloc[0]
        agef_v2 = v2[v2["type"] == "Fan"]["age_coef"].iloc[0]
        comparison.append({
            "pipeline": "V2 (Judge + Fan proxy)",
            "r2_judge": r2j_v2,
            "r2_fan_or_survival": r2f_v2,
            "age_coef_judge": agej_v2,
            "age_coef_fan_or_survival": agef_v2,
            "n_judge": None,
            "n_survival": None,
        })
    pd.DataFrame(comparison).to_csv(OUT_DIR / "comparison_v2_vs_v3.csv", index=False)
    print("  comparison_v2_vs_v3.csv")

    # ----- Figures -----
    # 1. Age effect: Judge vs Survival (with CI) — "Age Reversal"
    fig, ax = plt.subplots(figsize=(5, 3.5))
    err_j = 1.96 * (se_age_judge if not pd.isna(se_age_judge) else 0)
    err_s = 1.96 * (se_age_surv if not pd.isna(se_age_surv) else 0)
    ax.bar(["Judge Score\n(momentum model)", "Survival Rate\n(fan framework)"], [age_judge, age_surv], yerr=[err_j, err_s], capsize=8, color=["#4c72b0", "#55a868"], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coefficient (Effect per Year)")
    ax.set_title("The Age Reversal: Judge Penalty vs Fan Tolerance (V3)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_age_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_age_reversal.png")

    # 2. Industry effects: Judge vs Survival
    if all_ind:
        j_vals = [ie_judge.get(i, np.nan) for i in all_ind]
        s_vals = [ie_surv.get(i, np.nan) for i in all_ind]
        df_ind = pd.DataFrame({"Industry": all_ind, "Judge": j_vals, "Survival": s_vals})
        df_ind = df_ind.melt(id_vars="Industry", var_name="Type", value_name="Effect")
        fig, ax = plt.subplots(figsize=(10, max(4, len(all_ind) * 0.35)))
        sns.barplot(data=df_ind, x="Industry", y="Effect", hue="Type", ax=ax, palette=["#4c72b0", "#55a868"])
        ax.set_title("Industry Bias: Judge vs Survival (Reality TV = low score, high survival)")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.axhline(0, color="black")
        ax.legend(title="Model")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig2_industry_effects.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig2_industry_effects.png")

    # 3. Partner scatter: Judge vs Survival
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_pe["Judge_Effect"], df_pe["Survival_Effect"], alpha=0.7, s=60)
    ax.axhline(0, color="grey", linestyle="--")
    ax.axvline(0, color="grey", linestyle="--")
    df_pe["dist"] = df_pe["Judge_Effect"] ** 2 + df_pe["Survival_Effect"] ** 2
    for _, row in df_pe.nlargest(6, "dist").iterrows():
        ax.text(row["Judge_Effect"] + 0.01, row["Survival_Effect"], row["Partner"], fontsize=8)
    ax.set_xlabel("Effect on Judge Score (Std Dev)")
    ax.set_ylabel("Effect on Survival Rate")
    ax.set_title("Partner Effects: Judge (momentum) vs Survival (V3)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_partner_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_partner_scatter.png")

    # 4. R² comparison V2 vs V3 (V3 first = green, V2 second = red)
    if len(comparison) >= 2:
        comp_df = pd.DataFrame(comparison)
        # Ensure V3 first for consistent color (green = improved)
        if comp_df["pipeline"].iloc[0] != "V3 (Judge lag + Survival)":
            comp_df = comp_df.iloc[::-1].reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = comp_df["pipeline"].tolist()
        colors = ["#55a868" if "V3" in p else "#c44e52" for p in x]
        axes[0].bar(x, comp_df["r2_judge"], color=colors, alpha=0.85)
        axes[0].set_ylabel("R²")
        axes[0].set_title("Judge Model R²")
        axes[0].set_ylim(0, min(1.0, comp_df["r2_judge"].max() * 1.15))
        axes[0].tick_params(axis="x", rotation=18)
        axes[1].bar(x, comp_df["r2_fan_or_survival"], color=colors, alpha=0.85)
        axes[1].set_ylabel("R²")
        axes[1].set_title("Fan/Survival Model R²")
        axes[1].set_ylim(0, min(1.0, comp_df["r2_fan_or_survival"].max() * 1.15))
        axes[1].tick_params(axis="x", rotation=18)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig4_r2_comparison_v2_v3.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig4_r2_comparison_v2_v3.png")

    # 5. Composite for paper
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["Judge\n(momentum)", "Survival\n(fan)"], [age_judge, age_surv], yerr=[err_j, err_s], capsize=8, color=["#4c72b0", "#55a868"], alpha=0.85)
    ax1.axhline(0, color="black")
    ax1.set_ylabel("Age Coefficient")
    ax1.set_title("Age Reversal (V3)")
    ax2 = fig.add_subplot(gs[0, 1])
    if all_ind:
        x = np.arange(len(all_ind))
        w = 0.35
        ax2.bar(x - w/2, [ie_judge.get(i, np.nan) for i in all_ind], w, label="Judge", color="#4c72b0", alpha=0.85)
        ax2.bar(x + w/2, [ie_surv.get(i, np.nan) for i in all_ind], w, label="Survival", color="#55a868", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_ind, rotation=35, ha="right")
        ax2.axhline(0, color="black")
        ax2.set_ylabel("Effect vs Actor")
        ax2.set_title("Industry: Reality TV = low score, high survival")
        ax2.legend()
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(df_pe["Judge_Effect"], df_pe["Survival_Effect"], alpha=0.7, s=70)
    ax3.axhline(0, color="grey", linestyle="--")
    ax3.axvline(0, color="grey", linestyle="--")
    for _, row in df_pe.nlargest(5, "dist").iterrows():
        ax3.text(row["Judge_Effect"] + 0.01, row["Survival_Effect"], row["Partner"], fontsize=9)
    ax3.set_xlabel("Effect on Judge Score")
    ax3.set_ylabel("Effect on Survival Rate")
    ax3.set_title("Partner Effects (V3)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prob3_v3_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  prob3_v3_analysis_plot.png")

    print("Done.")


if __name__ == "__main__":
    main()
