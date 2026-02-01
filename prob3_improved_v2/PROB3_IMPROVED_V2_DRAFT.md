# Problem 3 Improved V2: Impact of Pro Dancers and Celebrity Characteristics — Model Draft

## 1. Objective and Research Question

We use DWTS data to analyze how **pro dancers (ballroom partners)** and **celebrity characteristics (age, industry)** affect competition outcomes. The problem asks: *How much do such things impact how well a celebrity will do, and do they impact judges’ scores and fan votes in the same way?*

Because observations are **not independent** (contestants share industry and partner), we use a **multilevel (hierarchical) linear model**:

\[
y_{ij} = \beta_0 + \beta_{\text{age}} \cdot \text{Age}_i + \alpha_{\text{ind}[i]} + \gamma_{\text{partner}[i]} + \varepsilon_{ij}, \quad \varepsilon_{ij} \sim N(0, \sigma_y^2).
\]

- **\(\beta_0\)**: baseline; **\(\beta_{\text{age}}\)**: fixed effect of age.
- **\(\alpha_{\text{ind}[i]}\)**: fixed effect of celebrity **industry** (grouped).
- **\(\gamma_{\text{partner}[i]}\)**: effect of **ballroom partner** (fixed effects in our implementation for robustness).
- We fit this **twice**: once for **standardized judge score** (\(y = \text{judge\_z}\)) and once for **fan support** (\(y = \text{fan\_log\_odds}\) or **fan proxy**).

---

## 2. Data and Two Pipelines

### 2.1 Data Sources

- **Clean data**: `celebrity_name`, `season`, `ballroom_partner`, `celebrity_industry`, `celebrity_age_during_season`.
- **Long data**: contestant–week level with **normalized** judge score (0–1).
- **Fan vote estimates** (optional): from mc_improved, `vote_share_mean` per (season, week, celebrity).

### 2.2 Industry Grouping

Raw industry strings are mapped into **nine groups** (Actor as reference): Actor, Musician, Athlete, Reality TV, Host, Comedian, Model, Social Media, Other. This reduces noise and allows interpretable “industry bias” (e.g., Reality TV vs Athlete).

### 2.3 Two Pipelines for Comparison

| Pipeline | Fan outcome | Data | Purpose |
|----------|-------------|------|---------|
| **Original** | Log-odds of estimated fan vote share | Long + Clean + mc_improved | Uses **modeled fan votes**; direct answer to “impact on fan votes.” |
| **Improved V2** | Z-score of **fan proxy** (survival residual) | Long + Clean only | **No fan estimates**; proxy = residual of \(\text{survival\_rate} \sim \text{judge\_z}\). Robust, always converges. |

**Fan proxy logic**: Surviving longer than judge score predicts is interpreted as “fan favorite.” So \(\text{fan\_proxy} = \text{resid}(\text{survival\_rate} \sim \text{judge\_z})\), then z-scored.

### 2.4 Why OLS with Partner Fixed Effects (LSDV)?

The **original prob3** used **MixedLM** (random intercept by partner). In practice, that often hits:

- **Singular random-effects covariance** (e.g., many partners with few contestants).
- **Convergence failures** (“MLE on boundary,” “Hessian not positive definite”).

So we use **OLS with partner as fixed effects** (LSDV): same equation, but \(\gamma_{\text{partner}}\) are estimated as coefficients and **centered** so they behave like “deviation from average partner.” This:

- **Always converges.**
- Gives **comparable** age and industry coefficients and **partner effect rankings** to a mixed model when the latter fits.
- Is standard in applied work when the number of groups (partners) is moderate.

---

## 3. Outputs: Tables and Figures

### 3.1 Tables (in `output/`)

| File | Description |
|------|-------------|
| `comparison_original_vs_v2.csv` | Side-by-side: pipeline, method, N, age coef (judge/fan), partner variance (judge/fan), R² (judge/fan). |
| `v2_summary.csv` | V2 pipeline: age coefficient and SE, partner variance, R² for Judge and Fan models. |
| `original_summary.csv` | Same for Original pipeline (fan estimates). |
| `v2_partner_effects.csv` | Partner fixed effects (centered): Judge_Effect, Fan_Effect. |
| `v2_industry_effects.csv` | Industry fixed effects vs Actor: Judge, Fan. |
| `original_*` | Analogous files for Original pipeline. |

### 3.2 Figures (in `figures/`)

| File | Description |
|------|-------------|
| `prob3_analysis_plot.png` | **Main composite**: (1) Partner Judge vs Fan effect scatter; (2) Age penalty bar (Judge vs Fan); (3) Industry effects (Judge vs Fan). |
| `v2_partner_scatter.png` | Partner “halo”: Judge effect (x) vs Fan effect (y). Points far from origin = strong partner impact. |
| `v2_age_effect.png` | Age coefficient with 95% CI for Judge and Fan models. |
| `v2_industry_effects.png` | Bar chart: industry effects (vs Actor) for Judge and Fan. |
| `v2_variance_partner.png` | Variance of partner effects: Judge model vs Fan model. |
| `original_*` | Same set for Original pipeline (fan estimates). |

---

## 4. Interpretation of Results

### 4.1 Age Effect (“The Age Penalty”)

- **Judge model**: \(\beta_{\text{age}} \approx -0.031\) (SE ≈ 0.0017). **Negative**: older celebrities get lower standardized judge scores on average.
- **Fan model (Original, fan estimates)**: \(\beta_{\text{age}} \approx -0.009\) (smaller in magnitude). Age hurts fan vote **less** than judge score.
- **Fan model (V2, fan proxy)**: \(\beta_{\text{age}} \approx +0.008\). With the proxy, the sign can flip: “surviving longer than judge predicts” is weakly associated with **older** contestants in some seasons (e.g., loyal fan bases for veteran stars).

**Interpretation**: The competition is a “young person’s game” for **judges** (technique, fitness). **Fans** penalize age less; well-known older stars (e.g., Jerry Rice) can still get strong fan support. So age impacts judge scores and fan votes **in the same direction for true fan vote** (both negative), but **judges are stricter**. The proxy’s positive coefficient is a reminder that the proxy is not the true vote.

### 4.2 Industry Effects (“The Divergence of Evaluation Criteria”)

From **V2 industry effects** (vs Actor):

- **Reality TV**: Judge ≈ −0.33, Fan ≈ +0.18. **Strong divergence**: judges low, fans high (e.g., Bobby Bones–type pattern).
- **Comedian**: Judge ≈ −0.21, Fan ≈ +0.33. Same pattern.
- **Athlete**: Judge ≈ −0.16, Fan ≈ +0.08. Moderate; athletes get some fan boost.
- **Model**: Judge ≈ −0.56, Fan ≈ −0.15. Both negative; models underperform on average.
- **Social Media**: Judge ≈ −0.18, Fan ≈ −0.03. Near zero on fan in this sample.

**Interpretation**: Industries where **Judge and Fan effects have opposite signs** (e.g., Reality TV, Comedian) are “controversy makers”: survival is driven by fan support despite lower judge scores. This is the **divergence of evaluation criteria** — judges reward technique, fans reward entertainment and name recognition.

### 4.3 Partner Effects (“Partner Halo”)

- **Partner variance** is **larger in the Judge model** (≈ 0.30) than in the Fan model (≈ 0.07–0.10 depending on pipeline). So **partner identity** explains more variation in judge scores than in fan support in our specification.
- **Scatter (Judge vs Fan)**: Partners with high Judge effect and high Fan effect (top-right) are “all-round boosts”; high Judge but low Fan (or vice versa) show that **judges and fans respond differently** to the same partner. Examples from the data: some pros (e.g., Emma Slater, Sharna Burgess) show positive effects on both; others show a clear split.

**Interpretation**: The “super-weighted” pro (e.g., Derek Hough–type) would appear as high Fan effect: they “bring” fan base. Our model quantifies this as partner fixed effects; the scatter shows that **partner impact on judges and on fans is not the same**.

---

## 5. Comparison: Original (Fan Estimates) vs Improved V2 (Fan Proxy)

From `comparison_original_vs_v2.csv`:

| Metric | Original (fan estimates) | Improved V2 (fan proxy) |
|--------|---------------------------|--------------------------|
| N | 2777 | 2777 |
| Method | OLS_LSDV | OLS_LSDV |
| Age coef (Judge) | −0.0312 | −0.0312 |
| Age coef (Fan) | −0.0089 | +0.0082 |
| Var(partner) Judge | 0.302 | 0.302 |
| Var(partner) Fan | 0.072 | 0.095 |
| R² Judge | 0.266 | 0.266 |
| R² Fan | **0.117** | 0.057 |

**Which is “better”?**

- **Judge model**: Identical between the two pipelines (same judge data and specification). R² ≈ 0.27; partner and industry explain a meaningful share of judge score variation.
- **Fan model**:  
  - **Original** uses **estimated fan votes** from mc_improved, so it targets “true” fan impact. **R² Fan = 0.12** and **age coefficient negative** (−0.009), consistent with “fans penalize age less than judges.”  
  - **V2** uses only **survival residual** as proxy. **R² Fan = 0.06** (lower) and **age coefficient positive** (+0.008). The proxy is noisier and can reverse the sign of age when “survival beyond judge prediction” is correlated with other factors (e.g., older stars with loyal fans).

**Conclusion**: For **answering the problem** (“Do age/industry/partner impact judge scores and fan votes in the same way?”), the **Original pipeline with fan estimates** is **preferred**: it gives a direct comparison of effects on **estimated fan vote**. The **Improved V2 pipeline** is **preferred for robustness and reproducibility**: it requires **no** fan vote model, avoids MixedLM convergence issues, and still yields **consistent** industry and partner stories (divergence of criteria, partner halo). In practice we report **both** and emphasize that **age and industry impact judges and fans differently**, with the Original pipeline providing the cleaner fan-vote interpretation.

---

## 6. Limitations and O-Award Oriented Discussion

### 6.1 Selection Bias (Endogenous Partner–Celebrity Matching)

We do **not** assume that partner–celebrity assignment is random. If producers pair **strong celebrities** with **popular pros**, then \(\gamma_{\text{partner}}\) captures a **mixture** of:

- The pro’s true “coaching + fan base” effect, and  
- The average **quality of celebrities** assigned to that pro.

So “Derek Hough effect” may be partly **selection**: he tends to get ringers. The draft should state this explicitly:

> *While our model identifies strong partner effects, we acknowledge **endogenous selection bias**: producers may pair top celebrities with popular pros. Thus \(\gamma_{\text{partner}}\) likely reflects both the partner’s direct impact and the unobserved quality of celebrities assigned to them.*

### 6.2 Season and Time

We do not include **season** (or time) as a fixed/random effect. Grade inflation or scoring changes over 34 seasons could be confounded with industry or partner. Adding **season fixed effects** (or a time trend) would strengthen causal interpretation.

### 6.3 Uncertainty in Fan Vote Estimates

Fan votes are **estimated** (mc_improved), not observed. Using them as the outcome in the Original pipeline **understates** standard errors. In a full Bayesian setup, one would propagate estimation uncertainty from the fan-vote model into the impact model. Here we **acknowledge** that fan coefficients and R² are conditional on the fan estimates.

### 6.4 Interaction Effects

The model is **additive**. An **Age × Partner** or **Industry × Partner** interaction could capture, e.g., “older celebrities benefit more from a certain pro.” This would be a natural extension for a longer paper.

---

## 7. Summary

- We fit **multilevel-style** equations (age + industry + partner) to **judge score (z)** and **fan support** (log-odds of estimated vote **or** fan proxy).
- **Industry grouping** and **OLS with partner fixed effects** ensure **convergence** and interpretable **industry** and **partner** effects.
- **Age** penalizes judge scores more than fan support; **industry** shows **divergence** (e.g., Reality TV: judges low, fans high); **partner** effects differ between judge and fan models.
- **Original pipeline (fan estimates)** is better for **direct** “impact on fan vote”; **Improved V2 (fan proxy)** is better for **robustness** and when fan estimates are unavailable. Both support the conclusion that **judges and fans are influenced differently** by age, industry, and partner.

All code, outputs, and figures are in **`/Users/linnoah/Documents/C/prob3_improved_v2`**. Regenerate with: `python run_analysis.py`.
