#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3 Improved V5: Strategy A+B with Survival model refinements.
- Judge: same A+B (lagged judge).
- Survival Logit: baseline + nonlinear (judge_z²) + Age×Week + full. Report AUC, Pseudo-R², AIC/BIC.
- Industry scatter with polished axis labels.
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
    """Week-level: judge_z, lag_judge_z, judge_z_sq, week, survived_next, Age, Industry_Group, ballroom_partner."""
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
    df["judge_z_sq"] = df["judge_z"] ** 2
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])
    max_week_per_cs = df.groupby(["celebrity_name", "season"])["week"].max().reset_index().rename(columns={"week": "last_week"})
    df = df.merge(max_week_per_cs, on=["celebrity_name", "season"])
    df["survived_next"] = (df["week"] < df["last_week"]).astype(int)
    df = df.sort_values(["celebrity_name", "season", "week"])
    df["lag_judge_z"] = df.groupby(["celebrity_name", "season"])["judge_z"].shift(1)
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


def pseudo_r2_mcfadden(m):
    llf = m.llf
    try:
        y = m.model.endog
        n1 = float(np.sum(y))
        n0 = float(len(y) - n1)
        p = n1 / len(y)
        llf_null = n1 * np.log(p) + n0 * np.log(1 - p)
    except Exception:
        llf_null = 0.0
    if llf_null >= 0 or llf >= 0:
        return np.nan
    return 1.0 - (llf / llf_null)


def auc_logit(m, data, endog_name="survived_next"):
    try:
        from sklearn.metrics import roc_auc_score
        y = data[endog_name].values
        pred = m.predict(data)
        if np.all(np.isfinite(pred)) and len(np.unique(y)) >= 2:
            return float(roc_auc_score(y, pred))
    except Exception:
        pass
    return np.nan


def main():
    print("Problem 3 Improved V5: Strategy A+B + Survival refinements (AUC, nonlinear, Age×Week)")
    df = load_weekly_all()
    df_with_lag = df.dropna(subset=["lag_judge_z"]).copy()
    n_full = len(df)
    n_lag = len(df_with_lag)
    print(f"  Week-level: N = {n_full}, with lag: N = {n_lag}")

    aug_form_judge = "judge_z ~ lag_judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    m_judge, _ = fit_ols_partner_lsdv(aug_form_judge, df_with_lag)
    r2_judge = m_judge.rsquared
    print(f"  Judge (A+B): R² = {r2_judge:.4f}")

    # ----- Survival Logit variants -----
    surv_base = "survived_next ~ judge_z + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    surv_nonlinear = "survived_next ~ judge_z + judge_z_sq + Age + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    surv_age_week = "survived_next ~ judge_z + Age + week + Age:week + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"
    surv_full = "survived_next ~ judge_z + judge_z_sq + Age + week + Age:week + C(Industry_Group, Treatment(reference='Actor')) + C(ballroom_partner)"

    logit_results = {}
    for name, formula in [
        ("Logit (baseline)", surv_base),
        ("Logit + judge_z² (safe zone)", surv_nonlinear),
        ("Logit + Age×Week (veteran)", surv_age_week),
        ("Logit full (nonlinear + Age×Week)", surv_full),
    ]:
        try:
            m = Logit.from_formula(formula, data=df).fit(disp=0)
            pseudo = pseudo_r2_mcfadden(m)
            auc = auc_logit(m, df)
            logit_results[name] = {
                "model": m,
                "pseudo_r2": pseudo,
                "auc": auc,
                "aic": m.aic,
                "bic": m.bic,
            }
            print(f"  {name}: Pseudo-R² = {pseudo:.4f}, AUC = {auc:.4f}, AIC = {m.aic:.0f}, BIC = {m.bic:.0f}")
        except Exception as e:
            print(f"  {name}: fit failed ({e})")
            logit_results[name] = {"model": None, "pseudo_r2": np.nan, "auc": np.nan, "aic": np.nan, "bic": np.nan}

    # Best by AUC (for industry effects and scatter)
    best_name = max(
        (k for k, v in logit_results.items() if v["model"] is not None and not np.isnan(v.get("auc", np.nan))),
        key=lambda k: logit_results[k]["auc"],
        default="Logit (baseline)",
    )
    if logit_results[best_name]["model"] is None and "Logit (baseline)" in logit_results and logit_results["Logit (baseline)"]["model"] is not None:
        best_name = "Logit (baseline)"
    m_surv_best = logit_results[best_name]["model"] if logit_results[best_name]["model"] is not None else logit_results.get("Logit (baseline)", {}).get("model")
    if m_surv_best is None:
        m_surv_best = next(v["model"] for v in logit_results.values() if v["model"] is not None)

    # ----- Save comparison -----
    comp_rows = []
    for name, r in logit_results.items():
        comp_rows.append({
            "survival_model": name,
            "pseudo_r2": r.get("pseudo_r2", np.nan),
            "auc": r.get("auc", np.nan),
            "aic": r.get("aic", np.nan),
            "bic": r.get("bic", np.nan),
        })
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "survival_logit_comparison.csv", index=False)
    with open(OUT_DIR / "best_model.txt", "w") as f:
        f.write(f"judge_r2,{r2_judge}\nbest_survival_model,{best_name}\n")
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Problem 3 Improved V5\n")
        f.write(f"Judge (A+B) R² = {r2_judge:.4f}\n")
        f.write(f"Best survival model (by AUC): {best_name}\n")
        for name, r in logit_results.items():
            f.write(f"  {name}: Pseudo-R² = {r.get('pseudo_r2', np.nan):.4f}, AUC = {r.get('auc', np.nan):.4f}, AIC = {r.get('aic', np.nan):.0f}, BIC = {r.get('bic', np.nan):.0f}\n")
    print("  survival_logit_comparison.csv, summary.txt")

    # ----- Industry effects (Judge + best Survival) for scatter -----
    ie_judge = extract_industry_effects(m_judge)
    ie_surv = extract_industry_effects(m_surv_best)
    all_ind = sorted(set(ie_judge.keys()) | set(ie_surv.keys()))
    pd.DataFrame([{
        "Industry": k,
        "Judge_Effect": ie_judge.get(k, np.nan),
        "Survival_Effect": ie_surv.get(k, np.nan),
    } for k in all_ind]).to_csv(OUT_DIR / "industry_effects_v5.csv", index=False)
    j_vals = [ie_judge.get(i, np.nan) for i in all_ind]
    s_vals = [ie_surv.get(i, np.nan) for i in all_ind]

    # ----- Figures -----
    colors_auc = ["#4c72b0", "#55a868", "#c44e52", "#817ab8"]
    # 1. Industry scatter — polished axis labels (killer chart)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(j_vals, s_vals, s=140, alpha=0.9, edgecolors="k", linewidths=0.8)
    for i, ind in enumerate(all_ind):
        ax.annotate(ind, (j_vals[i], s_vals[i]), textcoords="offset points", xytext=(8, 6), fontsize=11, fontweight="medium")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Impact on Judge Score (Technical Ability)", fontsize=12)
    ax.set_ylabel("Impact on Survival Odds (Fan Base Strength)", fontsize=12)
    ax.set_title("Industry Bias: Judge vs Survival (V5 Strategy A+B)", fontsize=13)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_industry_scatter_polished.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_industry_scatter_polished.png")

    # 2. AUC comparison across Logit variants
    names = [k for k in logit_results.keys()]
    aucs = [logit_results[k].get("auc", np.nan) for k in names]
    valid_auc = [a for a in aucs if not np.isnan(a)]
    if valid_auc:
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(names, aucs, color=colors_auc[:len(names)], alpha=0.85)
        ax.axhline(0.7, color="green", linestyle="--", linewidth=1, label="AUC = 0.7 (strong)")
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_title("Survival Logit: AUC by Specification (V5)")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=22)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig2_auc_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig2_auc_comparison.png")

    # 3. AIC/BIC comparison
    aics = [logit_results[k].get("aic", np.nan) for k in names]
    bics = [logit_results[k].get("bic", np.nan) for k in names]
    if any(np.isfinite(aics)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(names, aics, color=colors_auc[:len(names)] if valid_auc else "steelblue", alpha=0.85)
        axes[0].set_ylabel("AIC")
        axes[0].set_title("Survival Logit: AIC (lower = better)")
        axes[0].tick_params(axis="x", rotation=22)
        axes[1].bar(names, bics, color=colors_auc[:len(names)] if valid_auc else "coral", alpha=0.85)
        axes[1].set_ylabel("BIC")
        axes[1].set_title("Survival Logit: BIC (lower = better)")
        axes[1].tick_params(axis="x", rotation=22)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig3_aic_bic_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig3_aic_bic_comparison.png")

    # 4. ROC curve for best model + AUC 95% CI
    auc_val = np.nan
    auc_ci_low = np.nan
    auc_ci_high = np.nan
    if m_surv_best is not None and not np.isnan(logit_results.get(best_name, {}).get("auc", np.nan)):
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
            y = df["survived_next"].values
            pred = m_surv_best.predict(df)
            fpr, tpr, _ = roc_curve(y, pred)
            auc_val = roc_auc_score(y, pred)
            # Bootstrap 95% CI for AUC (resample (y, pred) pairs)
            n_boot = 1000
            rng = np.random.default_rng(42)
            auc_boot = []
            n = len(y)
            pred_arr = np.asarray(pred)
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                ya, pa = y[idx], pred_arr[idx]
                if len(np.unique(ya)) >= 2 and np.all(np.isfinite(pa)):
                    try:
                        auc_boot.append(roc_auc_score(ya, pa))
                    except Exception:
                        pass
            if len(auc_boot) >= 100:
                auc_ci_low = float(np.percentile(auc_boot, 2.5))
                auc_ci_high = float(np.percentile(auc_boot, 97.5))
                pd.DataFrame([{
                    "auc": auc_val,
                    "auc_ci_95_low": auc_ci_low,
                    "auc_ci_95_high": auc_ci_high,
                }]).to_csv(OUT_DIR / "auc_ci_95.csv", index=False)
            else:
                auc_ci_low = auc_ci_high = np.nan
            fig, ax = plt.subplots(figsize=(5, 5))
            if np.isfinite(auc_ci_low) and np.isfinite(auc_ci_high) and 0 <= auc_ci_low <= 1 and 0 <= auc_ci_high <= 1:
                ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f}, 95% CI: [{auc_ci_low:.3f}, {auc_ci_high:.3f}])")
            else:
                ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve: Survival Logit ({best_name})")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig4_roc_curve.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig4_roc_curve.png")
        except Exception:
            pass

    # 5. Marginal effect of Age by Week ("The Diminishing Age Penalty") — visual rhetoric: tight Y-axis, 95% CI, grid
    if m_surv_best is not None:
        params = m_surv_best.params
        b_age = float(params.get("Age", 0.0))
        age_week_key = "Age:week"
        if age_week_key not in params.index:
            for k in params.index:
                if "Age" in str(k) and "week" in str(k):
                    age_week_key = k
                    break
        b_age_week = float(params.get(age_week_key, 0.0))
        weeks_plot = np.arange(1, 11, dtype=float)
        marginal_effect = b_age + b_age_week * weeks_plot
        # SE of marginal effect: Var(b_Age + week*b_Age:week) = Var(b_Age) + week^2*Var(b_Age:week) + 2*week*Cov(b_Age, b_Age:week)
        try:
            vc = m_surv_best.cov_params()
            var_age = vc.loc["Age", "Age"] if "Age" in vc.index else 0.0
            var_age_week = vc.loc[age_week_key, age_week_key] if age_week_key in vc.index else 0.0
            cov_aw = vc.loc["Age", age_week_key] if "Age" in vc.index and age_week_key in vc.index else 0.0
            se_me = np.sqrt(np.maximum(0, var_age + weeks_plot**2 * var_age_week + 2 * weeks_plot * cov_aw))
            ci_low = marginal_effect - 1.96 * se_me
            ci_high = marginal_effect + 1.96 * se_me
        except Exception:
            se_me = np.full_like(weeks_plot, np.nan)
            ci_low = ci_high = np.full_like(weeks_plot, np.nan)
        # Tight Y-axis to highlight change (Strategy 3: Zoom)
        y_min = min(marginal_effect.min(), np.nanmin(ci_low) if np.any(np.isfinite(ci_low)) else marginal_effect.min())
        y_max = max(marginal_effect.max(), np.nanmax(ci_high) if np.any(np.isfinite(ci_high)) else marginal_effect.max())
        padding = (y_max - y_min) * 0.25 if (y_max - y_min) > 1e-9 else 0.01
        y_lo = y_min - padding
        y_hi = y_max + padding
        fig, ax = plt.subplots(figsize=(7, 4))
        if np.any(np.isfinite(ci_low)) and np.any(np.isfinite(ci_high)):
            ax.fill_between(weeks_plot, ci_low, ci_high, color="steelblue", alpha=0.25)
        ax.plot(weeks_plot, marginal_effect, color="steelblue", lw=2.5, marker="o", markersize=8, label="Marginal effect")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Marginal Effect of Age on Log-Odds(Survival)", fontsize=11)
        ax.set_title("The Diminishing Age Penalty (V5 Full Survival Model)", fontsize=12)
        ax.set_xticks(weeks_plot)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=10)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig5_marginal_effect_age_by_week.png", dpi=150, bbox_inches="tight")
        plt.close()
        out_df = pd.DataFrame({"week": weeks_plot, "marginal_effect_age": marginal_effect})
        if np.any(np.isfinite(se_me)):
            out_df["se"] = se_me
            out_df["ci_95_low"] = ci_low
            out_df["ci_95_high"] = ci_high
        out_df.to_csv(OUT_DIR / "marginal_effect_age_by_week.csv", index=False)
        print("  fig5_marginal_effect_age_by_week.png")

        # 5b. "The Comeback of the Veteran" — two bars: marginal effect of Age at Week 1 vs Week 8 (penalty diminishes)
        try:
            me_w1 = float(marginal_effect[0])
            me_w8 = float(marginal_effect[7])
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["Week 1\n(Age penalty strong)", "Week 8\n(Age penalty weaker)"], [me_w1, me_w8], color=["#c44e52", "#55a868"], alpha=0.85, edgecolor="black", linewidth=0.8)
            ax.set_ylabel("Marginal Effect of Age on Log-Odds(Survival)", fontsize=11)
            ax.set_title("The Comeback of the Veteran: Diminishing Age Penalty", fontsize=12)
            ax.axhline(0, color="gray", linewidth=0.8)
            y_min = min(me_w1, me_w8)
            y_max = max(me_w1, me_w8)
            padding = (y_max - y_min) * 0.3 if (y_max - y_min) > 1e-9 else 0.005
            ax.set_ylim(y_min - padding, 0 + padding)
            for bar, val in zip(bars, [me_w1, me_w8]):
                ax.text(bar.get_x() + bar.get_width() / 2, val - 0.0015 if val < 0 else val + 0.001, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig5b_comeback_of_the_veteran.png", dpi=150, bbox_inches="tight")
            plt.close()
            pd.DataFrame([{"scenario": "Week 1", "marginal_effect_age": me_w1}, {"scenario": "Week 8", "marginal_effect_age": me_w8}]).to_csv(OUT_DIR / "comeback_veteran_scenario.csv", index=False)
            print("  fig5b_comeback_of_the_veteran.png")
        except Exception as e:
            print("  fig5b skipped:", e)

        # 5c. "The Veteran's Struggle" — actual vs counterfactual (no age penalty) survival probability by week
        try:
            weeks_plot = np.arange(1, 11, dtype=float)
            p_actual = []
            p_counterfactual = []
            b_age = float(m_surv_best.params["Age"])
            b_aw = float(m_surv_best.params["Age:week"])
            base = df.iloc[[0]].copy()
            base["Age"] = 60
            for w in weeks_plot:
                r = base.copy()
                r["week"] = w
                r["judge_z_sq"] = r["judge_z"].iloc[0] ** 2
                p = float(m_surv_best.predict(r).iloc[0])
                p = np.clip(p, 1e-6, 1 - 1e-6)
                eta = np.log(p / (1 - p))
                age_contrib = 60 * b_age + 60 * w * b_aw
                eta_no_age = eta - age_contrib
                p_no = 1 / (1 + np.exp(-eta_no_age))
                p_actual.append(p)
                p_counterfactual.append(p_no)
            p_actual = np.array(p_actual)
            p_counterfactual = np.array(p_counterfactual)
            gap_w1 = p_counterfactual[0] - p_actual[0]
            gap_w8 = p_counterfactual[7] - p_actual[7]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.fill_between(weeks_plot, p_actual, p_counterfactual, color="gray", alpha=0.35, label="Lost Probability (Age Penalty)")
            ax.plot(weeks_plot, p_counterfactual, color="green", linestyle="--", lw=2.5, label="Counterfactual (No Age Penalty)")
            ax.plot(weeks_plot, p_actual, color="red", linestyle="-", lw=2.5, label="Actual (Age 60)")
            ax.set_xlabel("Week of Competition", fontsize=12)
            ax.set_ylabel("Probability of Surviving to Next Week", fontsize=11)
            ax.set_title("The Veteran's Struggle: Impact of Age on Survival Probability", fontsize=12)
            ax.set_xlim(1, 10)
            ax.set_ylim(0, 1)
            ax.set_xticks(weeks_plot)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc="upper right", fontsize=9)
            ax.annotate(f"Gap: {gap_w1:.1f}%", xy=(1.2, (p_actual[0] + p_counterfactual[0]) / 2), fontsize=10, fontweight="bold")
            ax.annotate(f"Massive Gap: {gap_w8:.1f}%\n(The 'Invisible Wall')", xy=(7.5, (p_actual[7] + p_counterfactual[7]) / 2), fontsize=10, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"), arrowprops=dict(arrowstyle="->", color="gray"))
            ax.text(8.5, 0.92, "Danger Zone\n(Late Weeks)", fontsize=9, color="gray", style="italic")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig5c_comeback_of_the_veteran.png", dpi=150, bbox_inches="tight")
            plt.close()
            pd.DataFrame({"week": weeks_plot, "p_actual_60": p_actual, "p_counterfactual_no_age": p_counterfactual}).to_csv(OUT_DIR / "veteran_struggle_by_week.csv", index=False)
            print("  fig5c_comeback_of_the_veteran.png")

            # 5d. Same as 5c but: fill below red (Actual) with low-saturation red, above green (Counterfactual) with low-saturation green
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.fill_between(weeks_plot, 0, p_actual, color="#e8b4b8", alpha=0.5, label="Actual zone (Age 60)")
            ax.fill_between(weeks_plot, p_counterfactual, 1, color="#b8d4b8", alpha=0.5, label="Counterfactual zone (No penalty)")
            ax.fill_between(weeks_plot, p_actual, p_counterfactual, color="gray", alpha=0.25, label="Lost Probability (Age Penalty)")
            ax.plot(weeks_plot, p_counterfactual, color="#2d6a2d", linestyle="--", lw=2, label="Counterfactual (No Age Penalty)")
            ax.plot(weeks_plot, p_actual, color="#8b3a3a", linestyle="-", lw=2, label="Actual (Age 60)")
            ax.set_xlabel("Week of Competition", fontsize=12)
            ax.set_ylabel("Probability of Surviving to Next Week", fontsize=11)
            ax.set_title("The Veteran's Struggle: Impact of Age on Survival Probability", fontsize=12)
            ax.set_xlim(1, 10)
            ax.set_ylim(0, 1)
            ax.set_xticks(weeks_plot)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc="upper right", fontsize=9)
            ax.annotate(f"Gap: {gap_w1:.1f}%", xy=(1.2, (p_actual[0] + p_counterfactual[0]) / 2), fontsize=10, fontweight="bold")
            ax.annotate(f"Massive Gap: {gap_w8:.1f}%\n(The 'Invisible Wall')", xy=(7.5, (p_actual[7] + p_counterfactual[7]) / 2), fontsize=10, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"), arrowprops=dict(arrowstyle="->", color="gray"))
            ax.text(8.5, 0.92, "Danger Zone\n(Late Weeks)", fontsize=9, color="gray", style="italic")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig5d_comeback_of_the_veteran.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  fig5d_comeback_of_the_veteran.png")
        except Exception as e:
            print("  fig5c skipped:", e)

    # 6. Predicted survival probability vs Judge Z-Score (safe zone / nonlinearity)
    if m_surv_best is not None:
        try:
            grid_judge_z = np.linspace(-2.5, 2.5, 150)
            pred_df = df.iloc[[0]].copy()
            pred_df = pd.concat([pred_df] * len(grid_judge_z), ignore_index=True)
            pred_df["judge_z"] = grid_judge_z
            pred_df["judge_z_sq"] = grid_judge_z ** 2
            probs = m_surv_best.predict(pred_df)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grid_judge_z, probs, color="coral", lw=2.5)
            ax.set_xlabel("Judge Z-Score (Technical Ability)", fontsize=12)
            ax.set_ylabel("Predicted P(Survive to Next Week)", fontsize=11)
            ax.set_title("Predicted Survival Probability vs Judge Score (Safe Zone at High Score)", fontsize=12)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=10)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig6_predicted_prob_vs_judge_z.png", dpi=150, bbox_inches="tight")
            plt.close()
            pd.DataFrame({"judge_z": grid_judge_z, "predicted_prob": probs}).to_csv(OUT_DIR / "predicted_prob_vs_judge_z.csv", index=False)
            print("  fig6_predicted_prob_vs_judge_z.png")
        except Exception as e:
            print("  fig6 skipped:", e)

    # 7. Composite for paper
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(j_vals, s_vals, s=120, alpha=0.9, edgecolors="k", linewidths=0.6)
    for i, ind in enumerate(all_ind):
        ax1.annotate(ind, (j_vals[i], s_vals[i]), textcoords="offset points", xytext=(5, 5), fontsize=10)
    ax1.axhline(0, color="gray", linestyle="--")
    ax1.axvline(0, color="gray", linestyle="--")
    ax1.set_xlabel("Impact on Judge Score (Technical Ability)")
    ax1.set_ylabel("Impact on Survival Odds (Fan Base Strength)")
    ax1.set_title("Industry: Judge vs Survival (V5)")
    if valid_auc:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(names, aucs, color=colors_auc[:len(names)], alpha=0.85)
        ax2.axhline(0.7, color="green", linestyle="--", label="AUC=0.7")
        ax2.set_ylabel("AUC")
        ax2.set_title("Survival Logit AUC by Spec")
        ax2.tick_params(axis="x", rotation=22)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prob3_v5_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  prob3_v5_analysis_plot.png")

    print("Done.")


if __name__ == "__main__":
    main()
