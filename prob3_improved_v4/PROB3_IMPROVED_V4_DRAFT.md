# Problem 3 Improved V4: Strategy A (Lagged Judge) vs B (Binary Survival) — Model Draft

## 1. Objective

We evaluate the **current model (V2)** on three dimensions — **explanation (R²)**, **statistical significance**, and **reliability of conclusions** — and implement two improvement strategies:

- **Strategy A**: Add **lagged judge score** (momentum) to the Judge model.
- **Strategy B**: Replace the Fan “proxy residual” with a **binary survival model** (Logit: survived to next week vs eliminated).

We run **four variants** and compare: **Baseline** (no A, no B), **Strategy A only**, **Strategy B only**, and **Strategy A+B**.

---

## 2. Model Specification

### 2.1 Data

- **Unit**: Week-level — one row per (contestant, season, week).
- **Judge**: Z-score of normalized judge score by season; **lag_judge_z** = previous week’s judge Z (same contestant-season). Rows with lag available: week ≥ 2.
- **Fan proxy (Baseline / A-only)**: `fan_proxy_z` = z-scored residual of `survival_rate ~ judge_z` (same as V2).
- **Binary survival (B-only / A+B)**: **survived_next** = 1 if contestant appears in week+1, 0 if eliminated after this week. Interpretation: *“At each week, given judge score, who survives to the next week?”*

### 2.2 Four Variants

| Variant | Judge model | Fan/Survival model |
|--------|-------------|--------------------|
| **Baseline** | judge_z ~ Age + Industry + Partner | fan_proxy_z ~ Age + Industry + Partner |
| **Strategy A only** | judge_z ~ **lag_judge_z** + Age + Industry + Partner | fan_proxy_z ~ Age + Industry + Partner (unchanged) |
| **Strategy B only** | judge_z ~ Age + Industry + Partner (unchanged) | **Logit(survived_next ~ judge_z + Age + Industry + Partner)** |
| **Strategy A+B** | judge_z ~ **lag_judge_z** + Age + Industry + Partner | **Logit(survived_next ~ judge_z + Age + Industry + Partner)** |

---

## 3. Results: R² and Age Coefficients

From `output/comparison_all_variants.csv`:

| Variant | R² Judge | R² or Pseudo-R² Fan | Age coef (Judge) | Age coef (Fan/Survival) |
|---------|----------|---------------------|------------------|-------------------------|
| **Baseline** | 0.27 | 0.057 (OLS) | −0.031 | +0.008 |
| **Strategy A only** | **0.55** | 0.057 (OLS) | −0.014 | +0.008 |
| **Strategy B only** | 0.27 | 0.034 (Logit) | −0.031 | −0.028 |
| **Strategy A+B** | **0.55** | 0.034 (Logit) | −0.014 | −0.028 |

### 3.1 Interpretation

**Strategy A (Lagged Judge)**  
- **R² Judge** rises from **0.27 to 0.55** when adding `lag_judge_z`. This shows strong **path dependence** (stickiness) in judge scores: last week’s score is a very strong predictor of this week’s score.  
- Age coefficient in the Judge model **remains negative** but becomes smaller in magnitude (−0.031 → −0.014) because part of the “age penalty” is absorbed by the lag (older contestants tend to have had lower scores the week before).

**Strategy B (Binary Survival)**  
- Fan model is now **Logit(survived_next ~ judge_z + Age + Industry + Partner)**. We are **not** predicting unobserved vote share; we are quantifying *“who survives to the next week given current judge score?”* — i.e. **“逆天改命” (overcoming the odds)**.  
- **Pseudo-R²** (McFadden) for the logit is **0.034** — lower than the OLS fan proxy R² (0.057). This is expected: (1) pseudo-R² is not directly comparable to OLS R²; (2) binary outcomes are harder to fit; (3) many partner dummies make the model conservative. The **substantive** use of Strategy B is the **sign and significance** of Age and Industry **given judge score**.  
- In our run, **Age in the logit is negative** (−0.028): at each week, holding judge score constant, older contestants are slightly **more likely to be eliminated** in that week. This is the **week-by-week** hazard. It can differ from a **season-level** survival model (e.g. V3: survival_rate ~ mean_judge_z + Age + …), where age was positive (fans keep older contestants in the season longer on average). So: **week-level “survived next week”** vs **season-level “total weeks survived”** answer different questions; both are valid.

**Strategy A+B**  
- Combines the best Judge fit (R² ≈ 0.55) with the binary survival interpretation. Industry effects from A+B are used for the **industry bias scatter** (Judge effect vs Survival effect).

---

## 4. Industry Bias: Judge vs Survival (Strategy A+B)

From `output/industry_effects_ab.csv` and **fig3_industry_scatter_judge_vs_survival.png**:

- **Reality TV**: Judge effect ≈ −0.15 (below Actor), Survival effect ≈ −0.16. Both negative, but **Survival effect is less negative** than for Model — so Reality TV is “low score, **relatively** high survival” (fans cushion them).
- **Model**: Judge ≈ −0.24, Survival ≈ −0.90. **Strongly negative on both** — “双输” (lose on both dimensions): neither judges nor fans favor them on average.
- **Comedian, Athlete**: Survival effect less negative than Model — again “low score, relatively high survival” pattern.

The **scatter plot** (X = Judge effect, Y = Survival effect) is the high-impact figure: Reality TV and Comedian sit in the “low judge, higher survival” region; Model sits in the “low judge, low survival” corner. This visualizes **divergence of evaluation criteria** — who survives despite low scores (fan-driven) vs who is eliminated on both dimensions.

---

## 5. Figures and Tables

### 5.1 Tables (`output/`)

| File | Description |
|------|-------------|
| `comparison_all_variants.csv` | R² Judge, R²/Pseudo-R² Fan, age coef (Judge/Fan), N, for all four variants. |
| `industry_effects_ab.csv` | Industry fixed effects (Judge, Survival) from Strategy A+B. |
| `summary.txt` | Short text summary of variants and R². |

### 5.2 Figures (`figures/`)

| File | Description |
|------|-------------|
| `fig1_r2_comparison_variants.png` | R² Judge and R²/Pseudo-R² Fan for Baseline, A only, B only, A+B. |
| `fig2_age_coef_variants.png` | Age coefficient (Judge vs Fan/Survival) for all four variants. |
| `fig3_industry_scatter_judge_vs_survival.png` | **Industry bias scatter**: Judge effect (x) vs Survival effect (y). Reality TV = low score, high survival; Model = low score, low survival. |
| `fig4_industry_bars_ab.png` | Bar chart: Industry effects (Judge vs Survival) from A+B. |
| `prob3_v4_analysis_plot.png` | Composite: R² comparison + age coef + industry scatter. |

---

## 6. Conclusion and Recommendation

- **Strategy A (Lagged Judge)** is **highly effective**: R² Judge roughly **doubles** (0.27 → 0.55), and the lag coefficient is large (~0.57). The paper should state: *“Introducing inertia, R² more than doubles (0.27 → 0.55), indicating strong stickiness in judge scoring.”*
- **Strategy B (Binary Survival)** changes the **question** from “predict fan proxy” to “predict survival to next week given judge score.” Pseudo-R² is modest (0.034), but the model is **interpretable** and the industry scatter (Judge vs Survival effects) is a strong visual. Recommendation: **report both** the OLS fan proxy (Baseline/A) and the Logit survival (B / A+B); use the logit for the narrative *“who survives when the judge wants you out?”* and use the industry scatter from A+B as the high-impact figure.
- **Strategy A+B** is the **preferred main specification**: best Judge fit (R² ≈ 0.55) + clear survival interpretation + industry bias scatter. Use it for the “Industry: Reality TV = low score, high survival; Model = double disadvantage” story.

All code, outputs, and figures are in **`/Users/linnoah/Documents/C/prob3_improved_v4`**. Regenerate with: `python run_analysis.py`.
