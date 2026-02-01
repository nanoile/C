#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3 Improved V2: Multilevel analysis of pro dancers and celebrity characteristics.
- Uses industry grouping (Actor, Athlete, Reality TV, etc.) and OLS with partner fixed effects
  (LSDV) when MixedLM fails (singular covariance / convergence).
- Two pipelines: (A) Original-style with mc_improved fan estimates + MixedLM or OLS fallback;
  (B) Improved v2 with fan proxy (survival residual) and OLS only.
- Outputs: tables, figures, comparison vs original prob3.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
CLEAN_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
LONG_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_long.csv"
MC_VOTES_PATH = BASE.parent / "mc_improved" / "output" / "estimated_fan_votes.csv"
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
    if any(x in s for x in ["reality", "bachelor", "survivor", "housewife", "tv personality", "tv personality"]):
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


def load_weekly_with_proxy():
    """Build contestant-week data from long + clean; add judge_z and fan_proxy_z (no fan estimates)."""
    clean = pd.read_csv(CLEAN_PATH)
    long = pd.read_csv(LONG_PATH)
    clean = clean.rename(columns={
        "ballroom_partner": "ballroom_partner",
        "celebrity_industry": "celebrity_industry",
        "celebrity_age_during_season": "Age",
    })
    meta = clean[["celebrity_name", "season", "ballroom_partner", "celebrity_industry", "Age"]].copy()
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Industry_Group"] = meta["celebrity_industry"].apply(group_industry)
    long["normalized"] = pd.to_numeric(long["normalized"], errors="coerce")
    long = long.dropna(subset=["normalized"])
    df = long[["celebrity_name", "season", "week", "normalized"]].merge(
        meta, on=["celebrity_name", "season"], how="left"
    )
    df = df.dropna(subset=["Age", "Industry_Group", "ballroom_partner"])
    df["ballroom_partner"] = df["ballroom_partner"].astype(str).str.strip()
    df["Industry_Group"] = df["Industry_Group"].astype(str)

    # Judge Z per season
    df["judge_z"] = df.groupby("season")["normalized"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    # Survival proxy
    max_weeks = df.groupby("season")["week"].max().reset_index().rename(columns={"week": "max_weeks"})
    df = df.merge(max_weeks, on="season")
    df["survival_rate"] = df["week"] / df["max_weeks"]
    # Fan proxy = residual of survival ~ judge_z, then z-score
    import statsmodels.formula.api as smf
    ols_proxy = smf.ols("survival_rate ~ judge_z", data=df).fit()
    df["fan_proxy"] = ols_proxy.resid
    df["fan_proxy_z"] = (df["fan_proxy"] - df["fan_proxy"].mean()) / (df["fan_proxy"].std() + 1e-9)
    return df


def load_weekly_with_fan_estimates():
    """Merge long + clean + mc_improved fan votes; judge_z and fan_log_odds."""
    clean = pd.read_csv(CLEAN_PATH)
    long = pd.read_csv(LONG_PATH)
    fan = pd.read_csv(MC_VOTES_PATH)
    meta = clean[["celebrity_name", "season", "ballroom_partner", "celebrity_industry", "celebrity_age_during_season"]].copy()
    meta = meta.rename(columns={"celebrity_age_during_season": "Age", "celebrity_industry": "celebrity_industry"})
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Industry_Group"] = meta["celebrity_industry"].apply(group_industry)
    long = long[["celebrity_name", "season", "week", "normalized"]].copy()
    long["normalized"] = pd.to_numeric(long["normalized"], errors="coerce")
    long = long.dropna(subset=["normalized"])
    fan = fan[["season", "week", "celebrity_name", "vote_share_mean"]].copy()
    df = long.merge(meta, on=["celebrity_name", "season"], how="left")
    df = df.merge(fan, on=["celebrity_name", "season", "week"], how="inner")
    df = df.dropna(subset=["Age", "Industry_Group", "ballroom_partner"])
    df["ballroom_partner"] = df["ballroom_partner"].astype(str).str.strip()
    df["Industry_Group"] = df["Industry_Group"].astype(str)

    df["judge_z"] = df.groupby("season")["normalized"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    v = np.clip(df["vote_share_mean"].values, 1e-4, 1 - 1e-4)
    df["fan_log_odds"] = np.log(v / (1 - v))
    return df


def fit_ols_partner_lsdv(formula, data):
    """OLS with Age + Industry + C(ballroom_partner). Returns model and centered partner effects."""
    import statsmodels.formula.api as smf
    model = smf.ols(formula, data=data).fit()
    params = model.params
    partner_coefs = {}
    for k, v in params.items():
        if "ballroom_partner" in k:
            try:
                name = k.split("[T.")[1].split("]")[0]
                partner_coefs[name] = v
            except Exception:
                continue
    vals = np.array(list(partner_coefs.values())) if partner_coefs else np.array([0.0])
    mean_val = np.mean(vals)
    centered = {k: v - mean_val for k, v in partner_coefs.items()}
    return model, centered


def extract_industry_effects(model):
    params = model.params
    out = {}
    for k, v in params.items():
        if "Industry_Group" in str(k):
            try:
                name = str(k).split("[T.")[1].split("]")[0]
                out[name] = float(v)
            except Exception:
                continue
    return out


def run_pipeline_v2(df, name_judge_y="judge_z", name_fan_y="fan_proxy_z"):
    """Improved v2: OLS only (Judge and Fan) with Age + Industry_Group + C(ballroom_partner)."""
    import statsmodels.formula.api as smf
    formula_judge = f"{name_judge_y} ~ Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    formula_fan = f"{name_fan_y} ~ Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"

    ols_judge, pe_judge = fit_ols_partner_lsdv(formula_judge, df)
    ols_fan, pe_fan = fit_ols_partner_lsdv(formula_fan, df)

    ie_judge = extract_industry_effects(ols_judge)
    ie_fan = extract_industry_effects(ols_fan)
    age_judge = ols_judge.params.get("Age", np.nan)
    age_fan = ols_fan.params.get("Age", np.nan)
    se_age_judge = ols_judge.bse.get("Age", np.nan)
    se_age_fan = ols_fan.bse.get("Age", np.nan)

    var_judge = np.var(list(pe_judge.values())) if pe_judge else 0.0
    var_fan = np.var(list(pe_fan.values())) if pe_fan else 0.0

    partners = list(set(pe_judge.keys()) | set(pe_fan.keys()))
    df_pe = pd.DataFrame([
        {"Partner": p, "Judge_Effect": pe_judge.get(p, 0), "Fan_Effect": pe_fan.get(p, 0)}
        for p in partners
    ])
    return {
        "ols_judge": ols_judge,
        "ols_fan": ols_fan,
        "pe_judge": pe_judge,
        "pe_fan": pe_fan,
        "df_pe": df_pe,
        "ie_judge": ie_judge,
        "ie_fan": ie_fan,
        "age_judge": age_judge,
        "age_fan": age_fan,
        "se_age_judge": se_age_judge,
        "se_age_fan": se_age_fan,
        "var_partner_judge": var_judge,
        "var_partner_fan": var_fan,
        "r2_judge": ols_judge.rsquared,
        "r2_fan": ols_fan.rsquared,
        "method": "OLS_LSDV",
        "n": len(df),
    }


def run_pipeline_original(df, name_judge_y="judge_z", name_fan_y="fan_log_odds"):
    """Original-style: OLS LSDV on data with estimated fan votes (same form as V2 for fair comparison)."""
    return run_pipeline_v2(df, name_judge_y=name_judge_y, name_fan_y=name_fan_y)


def save_results(r, prefix, df_pe):
    r["df_pe"].to_csv(OUT_DIR / f"{prefix}_partner_effects.csv", index=False)
    pd.DataFrame([
        {"type": "Judge", "age_coef": r["age_judge"], "se_age": r["se_age_judge"], "var_partner": r["var_partner_judge"], "r2": r.get("r2_judge")},
        {"type": "Fan", "age_coef": r["age_fan"], "se_age": r["se_age_fan"], "var_partner": r["var_partner_fan"], "r2": r.get("r2_fan")},
    ]).to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
    all_ind = sorted(set(r["ie_judge"].keys()) | set(r["ie_fan"].keys()))
    ind_rows = [{"Industry": k, "Judge": r["ie_judge"].get(k, np.nan), "Fan": r["ie_fan"].get(k, np.nan)} for k in all_ind]
    pd.DataFrame(ind_rows).to_csv(OUT_DIR / f"{prefix}_industry_effects.csv", index=False)


def plot_all(r, prefix, title_suffix=""):
    df_pe = r["df_pe"]
    # 1. Partner scatter: Judge vs Fan
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_pe["Judge_Effect"], df_pe["Fan_Effect"], alpha=0.7, s=60)
    ax.axhline(0, color="grey", linestyle="--")
    ax.axvline(0, color="grey", linestyle="--")
    df_pe["dist"] = df_pe["Judge_Effect"] ** 2 + df_pe["Fan_Effect"] ** 2
    top = df_pe.nlargest(6, "dist")
    for _, row in top.iterrows():
        ax.text(row["Judge_Effect"] + 0.01, row["Fan_Effect"], row["Partner"], fontsize=8)
    ax.set_xlabel("Effect on Judge Score (Std Dev)")
    ax.set_ylabel("Effect on Fan Support (Std Dev)")
    ax.set_title(f"Partner 'Halo' Effect: Judge vs Fan {title_suffix}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_partner_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Age effect with CI
    fig, ax = plt.subplots(figsize=(5, 3))
    err_j = 1.96 * r["se_age_judge"] if not np.isnan(r["se_age_judge"]) else 0
    err_f = 1.96 * r["se_age_fan"] if not np.isnan(r["se_age_fan"]) else 0
    ax.bar(["Judge Score", "Fan Support"], [r["age_judge"], r["age_fan"]], yerr=[err_j, err_f], capsize=8, color=["#4c72b0", "#dd8452"], alpha=0.8)
    ax.axhline(0, color="black")
    ax.set_ylabel("Coefficient (Effect per Year)")
    ax.set_title(f"The 'Age Penalty' {title_suffix}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_age_effect.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Industry effects
    ie_judge = r["ie_judge"]
    ie_fan = r["ie_fan"]
    ind_set = sorted(set(ie_judge.keys()) | set(ie_fan.keys()))
    if ind_set:
        j_vals = [ie_judge.get(i, np.nan) for i in ind_set]
        f_vals = [ie_fan.get(i, np.nan) for i in ind_set]
        df_ind = pd.DataFrame({"Industry": ind_set, "Judge": j_vals, "Fan": f_vals})
        df_ind = df_ind.melt(id_vars="Industry", var_name="Type", value_name="Effect")
        fig, ax = plt.subplots(figsize=(10, max(4, len(ind_set) * 0.35)))
        sns.barplot(data=df_ind, x="Industry", y="Effect", hue="Type", ax=ax, palette="muted")
        ax.set_title(f"Industry Bias: Judge vs Fan {title_suffix}")
        ax.tick_params(axis="x", rotation=35)
        plt.setp(ax.get_xticklabels(), ha="right", rotation=35)
        ax.axhline(0, color="black")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{prefix}_industry_effects.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Partner variance
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Judge Model", "Fan Model"], [r["var_partner_judge"], r["var_partner_fan"]], color=["#4c72b0", "#dd8452"], alpha=0.8)
    ax.set_ylabel("Variance of Partner Effects")
    ax.set_title(f"Variance Explained by Partners {title_suffix}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_variance_partner.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Problem 3 Improved V2: Multilevel analysis (Industry grouping + OLS fallback)")
    # Pipeline B: Improved v2 (fan proxy only, no mc_improved)
    df_proxy = load_weekly_with_proxy()
    print(f"  Proxy data: N = {len(df_proxy)}, partners = {df_proxy['ballroom_partner'].nunique()}, industries = {df_proxy['Industry_Group'].nunique()}")

    r_v2 = run_pipeline_v2(df_proxy, name_judge_y="judge_z", name_fan_y="fan_proxy_z")
    print(f"  V2 pipeline: method = {r_v2['method']}, R² Judge = {r_v2['r2_judge']:.4f}, R² Fan = {r_v2['r2_fan']:.4f}")
    save_results(r_v2, "v2", r_v2["df_pe"])
    plot_all(r_v2, "v2", title_suffix="(Improved V2 — Fan Proxy)")

    # Pipeline A: Original-style with fan estimates (if available)
    comparison_rows = []
    if MC_VOTES_PATH.exists():
        df_fan = load_weekly_with_fan_estimates()
        print(f"  Fan-estimate data: N = {len(df_fan)}")
        r_orig = run_pipeline_original(df_fan, name_judge_y="judge_z", name_fan_y="fan_log_odds")
        print(f"  Original-style pipeline: method = {r_orig['method']}")
        save_results(r_orig, "original", r_orig["df_pe"])
        plot_all(r_orig, "original", title_suffix="(Original — Estimated Fan Votes)")

        comparison_rows = [
            {
                "pipeline": "Original (fan estimates)",
                "method": r_orig["method"],
                "n": r_orig["n"],
                "age_coef_judge": r_orig["age_judge"],
                "age_coef_fan": r_orig["age_fan"],
                "var_partner_judge": r_orig["var_partner_judge"],
                "var_partner_fan": r_orig["var_partner_fan"],
                "r2_judge": r_orig.get("r2_judge"),
                "r2_fan": r_orig.get("r2_fan"),
            },
            {
                "pipeline": "Improved V2 (fan proxy)",
                "method": r_v2["method"],
                "n": r_v2["n"],
                "age_coef_judge": r_v2["age_judge"],
                "age_coef_fan": r_v2["age_fan"],
                "var_partner_judge": r_v2["var_partner_judge"],
                "var_partner_fan": r_v2["var_partner_fan"],
                "r2_judge": r_v2["r2_judge"],
                "r2_fan": r_v2["r2_fan"],
            },
        ]
    else:
        comparison_rows = [
            {
                "pipeline": "Improved V2 (fan proxy)",
                "method": r_v2["method"],
                "n": r_v2["n"],
                "age_coef_judge": r_v2["age_judge"],
                "age_coef_fan": r_v2["age_fan"],
                "var_partner_judge": r_v2["var_partner_judge"],
                "var_partner_fan": r_v2["var_partner_fan"],
                "r2_judge": r_v2["r2_judge"],
                "r2_fan": r_v2["r2_fan"],
            },
        ]

    pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "comparison_original_vs_v2.csv", index=False)
    print("  comparison_original_vs_v2.csv")

    # Single combined figure for paper (V2 as main)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.scatter(r_v2["df_pe"]["Judge_Effect"], r_v2["df_pe"]["Fan_Effect"], alpha=0.7, s=80)
    ax1.axhline(0, color="grey", linestyle="--")
    ax1.axvline(0, color="grey", linestyle="--")
    r_v2["df_pe"]["dist"] = r_v2["df_pe"]["Judge_Effect"] ** 2 + r_v2["df_pe"]["Fan_Effect"] ** 2
    for _, row in r_v2["df_pe"].nlargest(6, "dist").iterrows():
        ax1.text(row["Judge_Effect"] + 0.01, row["Fan_Effect"], row["Partner"], fontsize=9)
    ax1.set_xlabel("Impact on Judge Score (Std Dev)")
    ax1.set_ylabel("Impact on Fan Support (Std Dev)")
    ax1.set_title("Partner 'Halo' Effect: Judge vs Fan (Improved V2)")

    ax2 = fig.add_subplot(gs[0, 1])
    err_j = 1.96 * r_v2["se_age_judge"]
    err_f = 1.96 * r_v2["se_age_fan"]
    ax2.bar(["Judge Score", "Fan Support"], [r_v2["age_judge"], r_v2["age_fan"]], yerr=[err_j, err_f], capsize=10, color=["#4c72b0", "#dd8452"], alpha=0.8)
    ax2.axhline(0, color="black")
    ax2.set_ylabel("Coefficient (Effect per Year)")
    ax2.set_title("The 'Age Penalty'")

    ax3 = fig.add_subplot(gs[1, 1])
    ind_set = sorted(set(r_v2["ie_judge"].keys()) | set(r_v2["ie_fan"].keys()))
    if ind_set:
        j_vals = [r_v2["ie_judge"].get(i, np.nan) for i in ind_set]
        f_vals = [r_v2["ie_fan"].get(i, np.nan) for i in ind_set]
        x = np.arange(len(ind_set))
        w = 0.35
        ax3.bar(x - w / 2, j_vals, w, label="Judge", color="#4c72b0", alpha=0.8)
        ax3.bar(x + w / 2, f_vals, w, label="Fan", color="#dd8452", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(ind_set, rotation=35, ha="right")
        ax3.axhline(0, color="black")
        ax3.set_ylabel("Effect vs Actor")
        ax3.set_title("Industry Bias: Reality Stars vs Athletes")
        ax3.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prob3_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  prob3_analysis_plot.png")

    print("Done.")


if __name__ == "__main__":
    main()
