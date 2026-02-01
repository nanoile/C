#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3: Bayesian multilevel model for DWTS.
y_ij ~ N(β_0 + β_age * Age_i + α_ind[i] + γ_partner[i], σ_y^2)
Applied to: (1) Z-score of judge score, (2) Log-odds of fan vote share.
Outputs: tables, figures, model draft.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

BASE = Path(__file__).resolve().parent
CLEAN_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
LONG_PATH = BASE.parent / "clean" / "2026_MCM_Problem_C_Data_long.csv"
MC_VOTES_PATH = BASE.parent / "mc_improved" / "output" / "estimated_fan_votes.csv"
OUT_DIR = BASE / "output"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_merged_data():
    """Merge clean (age, industry), long (judge normalized, partner), fan votes (vote_share_mean)."""
    clean = pd.read_csv(CLEAN_PATH)
    long = pd.read_csv(LONG_PATH)
    fan = pd.read_csv(MC_VOTES_PATH)
    # Clean: one row per (celebrity_name, season); need age, industry, partner
    meta = clean[["celebrity_name", "season", "ballroom_partner", "celebrity_industry", "celebrity_age_during_season"]].copy()
    meta = meta.rename(columns={"ballroom_partner": "partner", "celebrity_industry": "industry", "celebrity_age_during_season": "age"})
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    # Long: (celebrity_name, season, week, normalized)
    long = long[["celebrity_name", "season", "week", "normalized"]].copy()
    long["normalized"] = pd.to_numeric(long["normalized"], errors="coerce")
    long = long.dropna(subset=["normalized"])
    # Fan: (season, week, celebrity_name, vote_share_mean, judge_share)
    fan = fan[["season", "week", "celebrity_name", "vote_share_mean"]].copy()
    # Merge long with meta on (celebrity_name, season)
    df = long.merge(meta, on=["celebrity_name", "season"], how="left")
    df = df.merge(fan, on=["celebrity_name", "season", "week"], how="inner")
    df = df.dropna(subset=["age", "industry", "partner"])
    df["industry"] = df["industry"].astype(str).str.strip()
    df["partner"] = df["partner"].astype(str).str.strip()
    return df


def build_model_matrices(df):
    """Create design matrix: intercept, age, industry dummies. Groups = partner."""
    df = df.copy()
    df["const"] = 1.0
    age = df["age"].values
    industries = df["industry"].unique().tolist()
    ind_dummies = np.zeros((len(df), len(industries)))
    for i, ind in enumerate(industries):
        ind_dummies[:, i] = (df["industry"].values == ind).astype(float)
    # Drop one industry for reference (avoid collinearity)
    exog = np.column_stack([df["const"].values, age, ind_dummies[:, 1:]])
    exog_names = ["intercept", "age"] + [f"industry_{industries[j]}" for j in range(1, len(industries))]
    groups = df["partner"].values
    return exog, exog_names, groups, df, industries


def fit_mixed_lm(y, exog, groups, exog_names):
    """Fit mixed linear model: y ~ exog + (1|groups). Returns fit and random effects."""
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
        model = MixedLM(y, exog, groups)
        result = model.fit(method="lbfgs", maxiter=200)
        re = result.random_effects
        return result, re, None
    except Exception as e:
        return None, None, str(e)


def main():
    print("Problem 3: Multilevel model (age, industry, partner)")
    df = load_merged_data()
    print(f"  Merged rows: {len(df)}, celebrities: {df['celebrity_name'].nunique()}, partners: {df['partner'].nunique()}, industries: {df['industry'].nunique()}")

    # Outcomes
    y_judge_raw = df["normalized"].values
    y_judge = (y_judge_raw - y_judge_raw.mean()) / (y_judge_raw.std() + 1e-9)
    v = np.clip(df["vote_share_mean"].values, 1e-4, 1 - 1e-4)
    y_fan = np.log(v / (1 - v))

    exog, exog_names, groups, df, industries = build_model_matrices(df)

    # Fit judge model
    res_judge, re_judge, err_judge = fit_mixed_lm(y_judge, exog, groups, exog_names)
    if err_judge:
        print("  Judge model failed:", err_judge)
    else:
        print("  Judge model: OK")

    # Fit fan model
    res_fan, re_fan, err_fan = fit_mixed_lm(y_fan, exog, groups, exog_names)
    if err_fan:
        print("  Fan model failed:", err_fan)
    else:
        print("  Fan model: OK")

    # ----- Tables -----
    tables = []
    if res_judge is not None:
        coef_judge = pd.DataFrame({
            "effect": exog_names,
            "coef_judge": res_judge.fe_params,
            "se_judge": res_judge.bse_fe,
        })
        coef_judge.to_csv(OUT_DIR / "coef_judge.csv", index=False)
        tables.append("coef_judge.csv")
        try:
            var_partner_judge = float(res_judge.cov_re.iloc[0, 0])
            var_resid_judge = float(res_judge.scale)
            pd.DataFrame([{"var_partner_judge": var_partner_judge, "var_resid_judge": var_resid_judge}]).to_csv(OUT_DIR / "variance_components_judge.csv", index=False)
        except Exception:
            if re_judge is not None:
                vpj = np.var(list(v.iloc[0] for v in re_judge.values()))
                pd.DataFrame([{"var_partner_judge": vpj, "var_resid_judge": float(res_judge.scale)}]).to_csv(OUT_DIR / "variance_components_judge.csv", index=False)

    if res_fan is not None:
        coef_fan = pd.DataFrame({
            "effect": exog_names,
            "coef_fan": res_fan.fe_params,
            "se_fan": res_fan.bse_fe,
        })
        coef_fan.to_csv(OUT_DIR / "coef_fan.csv", index=False)
        tables.append("coef_fan.csv")
        try:
            var_partner_fan = float(res_fan.cov_re.iloc[0, 0])
            var_resid_fan = float(res_fan.scale)
            pd.DataFrame([{"var_partner_fan": var_partner_fan, "var_resid_fan": var_resid_fan}]).to_csv(OUT_DIR / "variance_components_fan.csv", index=False)
        except Exception:
            if re_fan is not None:
                vpf = np.var(list(v.iloc[0] for v in re_fan.values()))
                pd.DataFrame([{"var_partner_fan": vpf, "var_resid_fan": float(res_fan.scale)}]).to_csv(OUT_DIR / "variance_components_fan.csv", index=False)

    # Partner random effects (BLUPs)
    if re_judge is not None:
        partner_effects_judge = pd.DataFrame([
            {"partner": k, "effect_judge": float(v.iloc[0])} for k, v in re_judge.items()
        ]).sort_values("effect_judge", ascending=False)
        partner_effects_judge.to_csv(OUT_DIR / "partner_effects_judge.csv", index=False)
    if re_fan is not None:
        partner_effects_fan = pd.DataFrame([
            {"partner": k, "effect_fan": float(v.iloc[0])} for k, v in re_fan.items()
        ]).sort_values("effect_fan", ascending=False)
        partner_effects_fan.to_csv(OUT_DIR / "partner_effects_fan.csv", index=False)

    # ----- Figures -----
    # 1. Age effect: coefficient with CI
    if res_judge is not None and res_fan is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        age_idx = exog_names.index("age") if "age" in exog_names else 1
        b_judge = res_judge.fe_params[age_idx]
        b_fan = res_fan.fe_params[age_idx]
        se_j = res_judge.bse_fe[age_idx]
        se_f = res_fan.bse_fe[age_idx]
        ax.barh([0], [b_judge], xerr=1.96 * se_j, color="steelblue", alpha=0.8, label="Judge (z-score)")
        ax.barh([1], [b_fan], xerr=1.96 * se_f, color="coral", alpha=0.8, label="Fan (log-odds)")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Judge score (z)", "Fan vote (log-odds)"])
        ax.set_xlabel("Coefficient (age effect)")
        ax.set_title("Age effect: negative = penalty for older celebrities")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig1_age_effect.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig1_age_effect.png")

    # 2. Industry effects (fixed): judge vs fan
    if res_judge is not None and res_fan is not None and len(exog_names) > 2:
        idx_judge = [i for i, n in enumerate(exog_names) if n.startswith("industry_")]
        idx_fan = idx_judge
        b_j = res_judge.fe_params[idx_judge].values if hasattr(res_judge.fe_params[idx_judge], "values") else res_judge.fe_params[idx_judge]
        b_f = res_fan.fe_params[idx_fan].values if hasattr(res_fan.fe_params[idx_fan], "values") else res_fan.fe_params[idx_fan]
        if isinstance(b_j, pd.Series):
            b_j = b_j.values
        if isinstance(b_f, pd.Series):
            b_f = b_f.values
        labels = [exog_names[i].replace("industry_", "") for i in idx_judge]
        n_ind = min(len(labels), len(b_j), len(b_f))
        if n_ind >= 1:
            fig, ax = plt.subplots(figsize=(8, max(4, n_ind * 0.35)))
            y_pos = np.arange(n_ind)
            ax.barh(y_pos - 0.2, b_j[:n_ind], height=0.35, color="steelblue", alpha=0.8, label="Judge")
            ax.barh(y_pos + 0.2, b_f[:n_ind], height=0.35, color="coral", alpha=0.8, label="Fan")
            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels[:n_ind], fontsize=8)
            ax.set_xlabel("Coefficient vs reference industry")
            ax.set_title("Industry effects (Judge vs Fan)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig2_industry_effects.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig2_industry_effects.png")

    # 3. Partner random effects: top and bottom (fan model - "super-weighted")
    if re_fan is not None:
        pf = partner_effects_fan
        n_show = min(20, len(pf))
        fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.3)))
        top = pf.head(n_show)
        y_pos = np.arange(len(top))
        colors = ["green" if x > 0 else "red" for x in top["effect_fan"]]
        ax.barh(y_pos, top["effect_fan"], color=colors, alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top["partner"], fontsize=8)
        ax.set_xlabel("Partner random effect (fan vote log-odds)")
        ax.set_title("Partner 'halo' effect on fan vote (top partners)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig3_partner_effects_fan.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig3_partner_effects_fan.png")

    if re_judge is not None:
        pj = partner_effects_judge
        n_show = min(20, len(pj))
        fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.3)))
        top = pj.head(n_show)
        y_pos_j = np.arange(len(top))
        colors = ["green" if x > 0 else "red" for x in top["effect_judge"]]
        ax.barh(y_pos_j, top["effect_judge"], color=colors, alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos_j)
        ax.set_yticklabels(top["partner"], fontsize=8)
        ax.set_xlabel("Partner random effect (judge z-score)")
        ax.set_title("Partner effect on judge score (top partners)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig4_partner_effects_judge.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig4_partner_effects_judge.png")

    # 4. Judge vs Fan: same industry/age impact? (scatter of coefficients)
    if res_judge is not None and res_fan is not None and len(exog_names) > 2:
        idx_ind = [i for i, n in enumerate(exog_names) if n.startswith("industry_")]
        if len(idx_ind) >= 1:
            b_j = res_judge.fe_params[idx_ind]
            b_f = res_fan.fe_params[idx_ind]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(b_j, b_f, alpha=0.7, s=50)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax.plot([min(b_j.min(), b_f.min()), max(b_j.max(), b_f.max())], [min(b_j.min(), b_f.min()), max(b_j.max(), b_f.max())], "k--", alpha=0.5, label="y=x")
            ax.set_xlabel("Coefficient (Judge model)")
            ax.set_ylabel("Coefficient (Fan model)")
            ax.set_title("Industry effects: Judge vs Fan (same direction?)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig5_judge_vs_fan_coef.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig5_judge_vs_fan_coef.png")

    # 5. Variance components (partner random effect variance)
    var_j = var_f = None
    if (OUT_DIR / "variance_components_judge.csv").exists():
        vj = pd.read_csv(OUT_DIR / "variance_components_judge.csv")
        var_j = float(vj["var_partner_judge"].iloc[0])
    if (OUT_DIR / "variance_components_fan.csv").exists():
        vf = pd.read_csv(OUT_DIR / "variance_components_fan.csv")
        var_f = float(vf["var_partner_fan"].iloc[0])
    if var_j is None and re_judge is not None:
        var_j = np.var([float(v.iloc[0]) for v in re_judge.values()])
    if var_f is None and re_fan is not None:
        var_f = np.var([float(v.iloc[0]) for v in re_fan.values()])
    if var_j is not None and var_f is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar([0], [var_j], width=0.35, label="Partner (Judge)", color="steelblue")
        ax.bar([0.4], [var_f], width=0.35, label="Partner (Fan)", color="coral")
        ax.set_xticks([0.2, 0.6])
        ax.set_xticklabels(["Judge model", "Fan model"])
        ax.set_ylabel("Variance (partner random effect)")
        ax.set_title("Partner variance: larger in fan model = 'super-weighted' pro dancers")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig6_variance_partner.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig6_variance_partner.png")

    # ----- Sensitivity: refit without age -----
    exog_no_age = np.column_stack([exog[:, 0], exog[:, 2:]])  # drop age column
    res_judge_na, _, _ = fit_mixed_lm(y_judge, exog_no_age, groups, None)
    res_fan_na, _, _ = fit_mixed_lm(y_fan, exog_no_age, groups, None)
    if res_judge_na is not None and res_fan_na is not None:
        pd.DataFrame([{
            "model": "with_age",
            "llf_judge": res_judge.llf if res_judge else np.nan,
            "llf_fan": res_fan.llf if res_fan else np.nan,
        }, {
            "model": "no_age",
            "llf_judge": res_judge_na.llf,
            "llf_fan": res_fan_na.llf,
        }]).to_csv(OUT_DIR / "sensitivity_no_age.csv", index=False)
        print("  sensitivity_no_age.csv")

    # ----- Summary -----
    with open(OUT_DIR / "model_summary.txt", "w") as f:
        f.write("Problem 3 Multilevel Model Summary\n")
        f.write(f"N observations: {len(df)}\n")
        if res_judge is not None:
            f.write(f"Judge model: age coef = {res_judge.fe_params[1]:.4f}, se = {res_judge.bse_fe[1]:.4f}\n")
        if res_fan is not None:
            f.write(f"Fan model: age coef = {res_fan.fe_params[1]:.4f}, se = {res_fan.bse_fe[1]:.4f}\n")
        if re_fan is not None and len(partner_effects_fan) > 0:
            f.write(f"Top partner (fan): {partner_effects_fan.iloc[0]['partner']} (effect = {partner_effects_fan.iloc[0]['effect_fan']:.4f})\n")
    print("  model_summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
