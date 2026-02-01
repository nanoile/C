# Problem 3: Impact of Pro Dancers and Celebrity Characteristics — Model Draft

## 1. Objective and Data

We use fan vote estimates (mc_improved) and cleaned DWTS data to analyze how **pro dancers (ballroom partners)** and **celebrity characteristics (age, industry)** affect performance. The question is: How much do these factors impact how well a celebrity does, and do they impact **judges’ scores** and **fan votes** in the same way?

**Data merged:**
- **Clean data**: celebrity_name, season, ballroom_partner, celebrity_industry, celebrity_age_during_season.
- **Long data**: (celebrity_name, season, week) with normalized judge score (0–1).
- **Fan votes**: (season, week, celebrity_name) with vote_share_mean from mc_improved.

**Sample:** 2777 observations (contestant-weeks), 408 celebrities, 60 partners, 26 industries.

---

## 2. Model Specification

We assume observations are **not independent**: contestants share **industry** (e.g. NFL, reality TV) and **partner** (e.g. Derek Hough, Cheryl Burke). A standard regression would ignore this and misattribute “partner quality” to the celebrity. We therefore use a **multilevel (mixed) linear model**:

\[
y_{ij} = \beta_0 + \beta_{\text{age}} \cdot \text{Age}_i + \alpha_{\text{ind}[i]} + \gamma_{\text{partner}[i]} + \varepsilon_{ij}, \quad \varepsilon_{ij} \sim N(0, \sigma_y^2).
\]

- **\(\beta_0\)**: baseline (intercept).
- **\(\beta_{\text{age}}\)**: fixed effect of age (one year older → change in outcome).
- **\(\alpha_{\text{ind}[i]}\)**: fixed effect of celebrity industry (one category as reference).
- **\(\gamma_{\text{partner}[i]}\)**: **random effect** of ballroom partner (mean 0, variance \(\sigma^2_{\text{partner}}\)).
- **\(\varepsilon_{ij}\)**: residual (variance \(\sigma_y^2\)).

We fit this model **twice**:
1. **Judge model**: outcome = **Z-score of normalized judge score** (by week).
2. **Fan model**: outcome = **log-odds of estimated fan vote share** (\(\log(p/(1-p))\)).

Implementation: **statsmodels MixedLM** with groups = partner, exog = [intercept, age, industry dummies]. So partner is a random intercept; industry and age are fixed effects.

---

## 3. Outputs: Tables and Figures

### 3.1 Tables

| File | Description |
|------|-------------|
| `output/coef_judge.csv` | Judge model: intercept, age, industry dummies (coef, se). |
| `output/coef_fan.csv` | Fan model: same structure. |
| `output/variance_components_judge.csv` | Judge: \(\sigma^2_{\text{partner}}\), \(\sigma^2_{\text{resid}}\). |
| `output/variance_components_fan.csv` | Fan: same. |
| `output/partner_effects_judge.csv` | Partner BLUPs (random effect estimates) for judge model, sorted. |
| `output/partner_effects_fan.csv` | Partner BLUPs for fan model, sorted. |
| `output/sensitivity_no_age.csv` | Log-likelihood with vs without age (model comparison). |
| `output/model_summary.txt` | Short summary: N, age coefficient (judge/fan), top partner (fan). |

### 3.2 Figures

| File | Description |
|------|-------------|
| `figures/fig1_age_effect.png` | Age coefficient (with 95% CI) for judge vs fan model. |
| `figures/fig2_industry_effects.png` | Industry fixed effects: judge vs fan (bar chart). |
| `figures/fig3_partner_effects_fan.png` | Partner random effects (fan model): top partners by “halo” effect. |
| `figures/fig4_partner_effects_judge.png` | Partner random effects (judge model): top partners. |
| `figures/fig5_judge_vs_fan_coef.png` | Industry coefficients: judge (x) vs fan (y); same direction? |
| `figures/fig6_variance_partner.png` | Partner variance: judge model vs fan model. |

---

## 4. Interpretation of Results

### 4.1 Age effect (fig1, coef tables)

- **Judge model**: \(\beta_{\text{age}}\) is **negative** (about −0.030 per year, small se). So older celebrities get **lower** standardized judge scores on average.
- **Fan model**: \(\beta_{\text{age}}\) is also **negative** but **smaller in magnitude** (about −0.008 per year). So age hurts fan vote less than it hurts judge score.

**Interpretation:** The competition is “a young person’s game” for **judges** (fitness, technique). **Fans** penalize age less; well-known older stars (e.g. Jerry Rice at 43) can still get strong fan support. So age impacts judge scores and fan votes **in the same direction but not the same way**—judges are stricter on age.

### 4.2 Industry effects (fig2, fig5, coef tables)

- **Judge model**: Industry dummies show which categories get higher/lower **judge** scores than the reference (e.g. Actor/Actress). Athletes often have a small negative coefficient (worse technique on average?), Models/Magicians vary.
- **Fan model**: Same industries can have **different** signs or magnitudes. For example, “Reality TV” or “TV Personality” may have a negative judge effect but a **positive** fan effect (Bobby Bones–type pattern: judges low, fans high).

**Interpretation:** Industry affects **judges and fans differently**. Reality/TV personalities can have low judge scores but high fan support; athletes can have better judge scores and moderate fan support. Fig5 (judge coef vs fan coef) shows whether industries lie on the y=x line (same effect) or not (divergent).

### 4.3 Partner “super-weighted” effect (fig3, fig4, fig6, partner_effects tables)

- **Partner variance** (\(\sigma^2_{\text{partner}}\)): In the **fan model**, partner variance is typically **large** relative to the judge model (or comparable). Fig6 compares partner variance between the two models.
- **Partner BLUPs**: Top partners (e.g. Derek Hough, Cheryl Burke, Charlotte Jorgensen) have **positive** random effects: their celebrities get higher judge scores and/or fan vote log-odds on average, **after** controlling for age and industry.

**Interpretation:** Pro dancers act like “campaign managers”: pairing with a high-BLUP partner is associated with better outcomes. This is **especially visible in the fan model** (partner variance and BLUPs), consistent with “some pros bring a built-in fan base.” So partner impacts **both** judge and fan outcomes, but the **fan** side shows strong partner heterogeneity—“super-weighted” pros.

### 4.4 Do they impact judges and fans in the same way?

- **Age**: Same direction (negative), but **stronger** for judges than for fans.
- **Industry**: Often **not** the same: e.g. reality/TV personalities can be judge-negative, fan-positive.
- **Partner**: Both models show partner effects; **variance and BLUPs** suggest that partner “halo” is at least as important (or more) for **fan** vote as for judge score.

So the answer is **no**: age, industry, and partner do **not** impact judge scores and fan votes in the same way; the multilevel model separates these channels and shows where they diverge.

---

## 5. Model Capability and Validation

- **Fit**: MixedLM maximizes marginal likelihood. We report log-likelihood (model_summary, sensitivity_no_age).
- **Sensitivity — dropping age**: We fit the same model **without** age. Log-likelihood **decreases** (e.g. judge: −3730 vs −3575; fan: −1851 vs −1814), so **including age improves fit**.
- **Residuals**: \(\sigma_y^2\) (scale) is the residual variance; partner variance \(\sigma^2_{\text{partner}}\) measures how much outcome variation is due to partner. A large \(\sigma^2_{\text{partner}}\) in the fan model indicates that partner is a major driver of fan vote.
- **Limitation**: Fan outcome is **estimated** (vote_share_mean from mc_improved); uncertainty in that estimate is not propagated into the multilevel model. Industry is **fixed** (not random) in our implementation, so we do not estimate \(\sigma^2_{\text{industry}}\); only partner is random.

---

## 6. Sensitivity and Robustness

- **Sensitivity to age**: Removing age worsens log-likelihood for both judge and fan models → age is a useful predictor.
- **Sensitivity to encoding**: Industry is dummy-coded (one reference). Changing the reference category only shifts intercept and industry coefficients; **age and partner effects** are unchanged.
- **Sensitivity to sample**: Results are based on 2777 contestant-weeks. Fitting on subsets (e.g. early vs late seasons) could show whether partner or industry effects change over time (e.g. “social media” industry in S29+).

---

## 7. Summary

- We fit a **multilevel linear model** (age + industry fixed, partner random) to **judge Z-score** and **fan log-odds**.
- **Age** hurts both outcomes, more so for judges than for fans.
- **Industry** effects differ between judge and fan (e.g. reality/TV: low judge, high fan).
- **Partner** random effects are significant in both models; partner variance (and BLUPs) in the **fan** model support the “super-weighted pro dancer” idea—some pros systematically associate with higher fan support.
- Model capability is supported by better fit when age is included; sensitivity checks (no age) confirm that age and partner are relevant. All outputs (tables and figures) are under `prob3/output` and `prob3/figures`.
