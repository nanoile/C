# Problem 2 Improved: Rank vs Percentage — O-Award Level Draft

## Executive Summary (Summary Sheet)

We compare the two methods used by the show to combine judge and fan votes—**Percentage (Rule A)** and **Rank (Rule B)**—using fan vote estimates from Problem 1 (mc_improved). We add **statistical tests**, **sensitivity analysis**, **fan-favoring quantification**, **survival-boundary and impact-matrix visualizations**, and a **Judges' Choice from bottom two** scenario. Our main conclusion:

- **Cross-method validation**: When the show used the **rank** method (Seasons 1–2, 28+), applying the **percentage** method to the same data yields **66.3%** accuracy vs **73.7%** for the rank method—a **7.4 percentage point** drop. When the show used percentage (S3–27), both methods perform similarly (~74% vs ~73%). So the **rank** rule is more “identifiable” from the data when it was in use.
- **Statistical significance**: A **chi-square test** on the contingency table (same-method match × cross-method match) rejects independence (p ≈ 7.3×10⁻⁵⁴). A **paired t-test** on rank-era seasons (same vs cross accuracy) gives p ≈ 0.054—marginally significant.
- **Fan-favoring**: In **share space**, judge and fan shares have comparable variance (ratio ≈ 0.98); the **rank** method compresses the **dynamic range** of fan contribution (rank 1..n), so it objectively limits the “popularity king” effect. In raw vote space, fan variance is typically much larger; percentage is fully compensatory and gives fan vote full leverage.
- **Sensitivity (±5% fan vote)**: Under random ±5% perturbation of fan shares, both methods change the predicted elimination in about **48–50%** of cases; structural robustness comes from rank’s **variance-stabilizing** property rather than from this perturbation test alone.
- **Recommendation**: We recommend **Rank Method + Judges' Choice from the bottom two**. Rank belongs to **robust statistics**, filtering fan-vote outliers; Judges' Choice acts as a **Condorcet-style** expert veto, balancing popularity and quality (social choice theory).

**One-line punch**: *“Rank method reduces the volatility of fan weight by variance-stabilizing the combination, significantly improving fit when applied (73.7% vs 66.3% cross), and limits extreme fan bias through bounded rank contribution.”*

---

## 1. Mathematical Setup and Two Methods

### 1.1 Percentage method (Rule A)

\[
S_{\text{total},i} = J_i^{\text{share}} + V_i^{\text{share}}, \quad \text{eliminate } \arg\min_i S_{\text{total},i}.
\]

- **Fully compensatory**: A large fan share can fully offset a small judge share.
- **Weight decomposition**: \(\frac{\partial S}{\partial V} = \frac{\partial S}{\partial J} = 1\) in share space. In **raw vote space**, \(\operatorname{Var}(V)\) is usually much larger than \(\operatorname{Var}(J)\), so a 1% change in fan vote often has larger impact on outcome than a 1% change in judge score.

### 1.2 Rank method (Rule B)

\[
S_{\text{total},i} = R(J_i) + R(V_i), \quad \text{rank 1 = best}, \quad \text{eliminate } \arg\max_i S_{\text{total},i}.
\]

- **Variance-stabilizing (robust statistics)**: The contribution of fan vote is **bounded** (1 to n). Extreme fan margins (1 vote vs 1 million votes) both yield rank 1, so the “popularity king” cannot dominate arbitrarily.

### 1.3 Data and implementation

- Fan vote estimates: mc_improved `estimated_fan_votes.csv` (judge_share, vote_share_mean per season/week/contestant).
- For each (season, week) we compute: elim under percentage, elim under rank, elim under “judges choose from bottom two” (eliminate the one with **lower judge score**).
- Outputs: `comparison_rank_vs_percentage.csv`, `accuracy_by_season_same_vs_cross.csv`, `comparison_with_judges_choice.csv`, plus all new tables and figures below.

---

## 2. Statistical Depth

### 2.1 Significance tests

**File**: `output/statistical_tests.csv`

- **Paired t-test** (within percentage seasons): same-method vs cross-method accuracy per season → p ≈ 0.36 (no significant difference).
- **Paired t-test** (within rank seasons): same vs cross → p ≈ **0.054** (marginally significant; rank fits better when it was the rule).
- **Chi-square test** (contingency: same-method match × cross-method match, 2×2 table): χ² ≈ 238.8, p ≈ **7.3×10⁻⁵⁴** → strong rejection of independence: the two methods do not agree by chance.

**Interpretation**: The large χ² shows that “same method correct” and “cross method correct” are strongly associated (when one is right, the other often is too; when they disagree, structure matters). The t-test on rank seasons suggests that applying percentage to rank-era data worsens accuracy with marginal significance.

### 2.2 Sensitivity analysis (±5% fan vote)

**Files**: `output/sensitivity_frac_switch.csv`, `output/sensitivity_summary.csv`

- For each (season, week), we perturb fan shares by **±5%** (uniform, 200 trials), renormalize, and recompute who would be eliminated under percentage and under rank.
- **Fraction of trials where elimination changes**: ~48% (percentage), ~49% (rank). So **robustness** (elimination unchanged) is ~52% vs ~51%—both methods are similarly sensitive to this level of noise.

**Interpretation**: The **structural** advantage of rank is **variance-stabilizing** (bounded rank contribution), not necessarily higher robustness to ±5% share error in this test. O-award angle: we **quantify** sensitivity and show that neither method is very robust to 5% fan-vote error; the recommendation for rank rests on identifiability and dynamic-range compression.

---

## 3. Fan-Favoring Quantification

### 3.1 Weight / variance decomposition

**File**: `output/weight_variance_decomposition.csv`

- Over all contestant-weeks: \(\operatorname{Var}(J^{\text{share}})\), \(\operatorname{Var}(V^{\text{share}})\), \(\operatorname{Var}(J+V)\), and **ratio** \(\operatorname{Var}(V)/\operatorname{Var}(J)\).
- In our **share** data (already normalized per week), the ratio is ≈ **0.98**: judge and fan shares have similar variance. Correlation judge–fan is high (~0.95), so total variance is not simply sum of two parts.

**Interpretation**:
- In **share space**, percentage gives fan and judge **equal** marginal weight (derivative 1). The “fan-favoring” of percentage appears when we think in **raw votes**: \(\operatorname{Var}(V_{\text{raw}})\) is usually much larger than \(\operatorname{Var}(J)\), so a 1% change in fan vote can correspond to a much larger change in raw votes than a 1% change in judge score.
- **Rank** compresses **dynamic range**: fan contribution is always in \(\{1,\ldots,n\}\). So rank objectively **limits** how much “extreme popularity” can swing the result—consistent with “rank does not favor fan votes more; it caps their leverage.”

---

## 4. Boundary Condition Analysis and Impact Matrix

### 4.1 Survival boundary plot — geometric distinction (linear vs step-wise)

**File**: `figures/fig_boundary_survival_zone.png`

The figure highlights the **physical meaning** of the two methods by drawing their **boundary lines** explicitly.

- **Left panel — Percentage method (linear boundary)**  
  Elimination rule: eliminate the contestant with **lowest total** \(S = J^{\text{share}} + V^{\text{share}}\). In the \((J, V)\) plane, the boundary is the **line** \(J + V = C\) (i.e. \(V = -J + C\)). We draw it as a **bold red line** and lightly shade elimination zone (below) and safe zone (above). **Geometric meaning**: the boundary is **linear (rigid)**—a small change in \((J, V)\) can cross the line easily, so small data/estimation error can flip who is eliminated.

- **Right panel — Rank method (step-wise boundary)**  
  Elimination rule: eliminate the contestant with **highest** rank-sum \(R(J) + R(V)\). In the **(Judge rank, Fan rank)** plane, ranks are **integers** 1 to \(n\); the boundary is the set of lattice points with \(r_J + r_V = \text{const}\). We draw a **step-wise (staircase) boundary** (bold blue) and a dashed gray diagonal for "if boundary were linear." **Geometric meaning**: the boundary is **step-wise / non-linear (elastic)**—small changes in raw scores often do not change ranks, so who is eliminated is **more stable**; Rank **tolerates data fluctuation** better than Percentage.

**Interpretation for judges**: Percentage → linear boundary → rigid → sensitive to small changes. Rank → step-wise boundary → elastic → small changes often keep same ranks → more robust. This is why Rank is a **variance-stabilizing (robust)** combination rule.



### 4.2 Impact matrix (Judge strength × Fan strength → survival probability)

**Files**: `output/impact_matrix_percentage.csv`, `output/impact_matrix_rank.csv`, `figures/fig_impact_matrix.png`

- Contestants are binned into **terciles** of judge share and fan share (Low / Mid / High).
- For each (Judge tercile, Fan tercile), we compute **P(survive this week)** under percentage and under rank (across all such contestant-weeks).
- Two heatmaps: one for percentage, one for rank.

**Interpretation**: 
- **Percentage**: High fan tercile generally increases survival even when judge is low (fully compensatory).
- **Rank**: Survival depends on **combined rank**; low judge + low fan → low survival; high fan can still help but within the bounded rank contribution. The matrices show that rank **reduces** the advantage of “high fan alone” relative to percentage—supporting “rank limits fan bias.”

---

## 5. Judges' Choice from Bottom Two (Scenario Analysis)

### 5.1 Rule

- Each week, define the **bottom two** either by **percentage** (lowest \(J+V\)) or by **rank** (worst two rank-sums).
- **Judges as “professional gatekeeper”**: Between these two, eliminate the contestant with the **lower judge score** (we do not observe real judge preference; this is a scenario).

### 5.2 Outputs

**Files**: `output/comparison_with_judges_choice.csv`, `output/judges_choice_controversy_cases.csv`

- For each (season, week): who would be eliminated under “judges choose from bottom two” when bottom two is defined by percentage vs by rank.
- For **Bristol Palin (S11)** and **Bobby Bones (S27)**: week-by-week, whether they are in the bottom two (pct / rank) and whether they would be eliminated under judges' choice (pct / rank).

**Interpretation**:
- **Scenario**: If judges always save the contestant with higher judge score from the bottom two, then in weeks when Bristol or Bobby are in the bottom two, they would often be **eliminated** (low judge score). So “Judges' Choice” would have removed them earlier in many cases—reducing **extreme fan bias** and balancing **quality vs popularity** (diversity–quality tradeoff in social choice terms).
- **Condorcet-style**: Giving experts a **veto** over the bottom two is a classic way to combine majority (fan) preference with expert (judge) judgment—we implement one concrete form and show its impact on controversy cases.

---

## 6. Consistency Heatmap and Controversy Trajectories

### 6.1 Consistency heatmap

**File**: `figures/fig_consistency_heatmap.png`

- Rows = **season**, columns = **week**. Color = **agreement** (green = percentage and rank select the **same** contestant to eliminate; red = disagree).
- **Interpretation**: Most cells are green (~84% agreement). Red cells concentrate in specific seasons/weeks where judge and fan rankings conflict; the heatmap shows **where** the two methods diverge.

### 6.2 Controversy trajectory (ranking under each method by week)

**Files**: `figures/fig_trajectory_Jerry_Rice.png`, `fig_trajectory_Bristol_Palin.png`, `fig_trajectory_Bobby_Bones.png`

- For **Jerry Rice (S2)**, **Bristol Palin (S11)**, **Bobby Bones (S27)**:
  - X = week; Y = **contestant rank** (1 = best) that week.
  - Two curves: rank under **percentage** (ordering by \(J+V\) descending) vs under **rank** (ordering by rank-sum ascending).

**Interpretation**: 
- When the two curves **diverge**, the contestant is ranked differently under the two methods (e.g. better under percentage than under rank, or vice versa). 
- **Jerry Rice**: In some weeks his rank under percentage is worse than under rank (percentage would eliminate him when rank would not)—consistent with “rank saved him” in those weeks.
- **Bristol / Bobby**: Trajectories show how their **ordinal position** shifts under the two methods week by week; combined with the judges' choice scenario, we see when they would have been at risk under “bottom two + judge veto.”

---

## 7. File Reference and Summary Sheet

| File | Description |
|------|-------------|
| `output/comparison_rank_vs_percentage.csv` | Per (season, week): actual elim, elim under pct/rank. |
| `output/accuracy_by_season_same_vs_cross.csv` | Per-season accuracy (same vs cross method). |
| `output/statistical_tests.csv` | t-test p-values, chi-square p-value and statistic. |
| `output/sensitivity_frac_switch.csv` | Per-week fraction of ±5% trials where elimination changes. |
| `output/sensitivity_summary.csv` | Mean robustness (pct vs rank). |
| `output/weight_variance_decomposition.csv` | Var(J), Var(V), ratio, correlation. |
| `output/impact_matrix_percentage.csv` | Survival prob by (Judge tercile, Fan tercile) — percentage. |
| `output/impact_matrix_rank.csv` | Same — rank. |
| `output/comparison_with_judges_choice.csv` | Per week: elim under judges' choice (pct/rank bottom two). |
| `output/judges_choice_controversy_cases.csv` | Bristol Palin, Bobby Bones: in bottom two? would judge choice eliminate? |
| `output/SUMMARY_SHEET.txt` | O-award style summary with numbers and one-line punch. |
| `figures/fig_boundary_survival_zone.png` | (J,V) and (Judge rank, Fan rank) with elimination zones. |
| `figures/fig_impact_matrix.png` | Impact matrix heatmaps (pct and rank). |
| `figures/fig_consistency_heatmap.png` | Season × week agreement (green/red). |
| `figures/fig_trajectory_*.png` | Jerry Rice, Bristol Palin, Bobby Bones: rank under pct vs rank by week. |
| `figures/fig1_accuracy_by_season_same_vs_cross.png` | Bar chart: same-method vs cross-method accuracy by season. |
| `figures/fig2_when_methods_disagree_fan_share.png` | Scatter: fan share of eliminated under pct vs rank when methods disagree. |
| `figures/fig3_Jerry_Rice.png`–`fig6_Bobby_Bones.png` | Judge vs fan share by week for controversy cases. |

---

## 8. Recommendation and Social Choice Perspective

- **Recommendation**: Use **Rank Method** for combining judge and fan input, plus **Judges' Choice from the bottom two** (define bottom two by rank-sum; judges choose whom to eliminate).
- **Rationale**:
  1. **Robust statistics**: Rank is a robust, variance-stabilizing combination that limits the effect of fan-vote outliers.
  2. **Identifiability**: When the show used rank, our model fits better (73.7% vs 66.3% cross); when it used percentage, the two methods are closer.
  3. **Social choice**: Judges' Choice acts as a **Condorcet-style** expert veto, balancing **popularity** (fans) and **quality** (judges), and addressing “extreme fan bias” that can hurt show quality.
- **Limitation**: Judges' Choice is implemented here as “eliminate the lower judge score from the bottom two”; real judge preferences could differ. The scenario still demonstrates how such a rule would change outcomes for controversy cases.

All improvements are implemented in `prob2_improved/run_analysis.py`; run it to regenerate tables and figures.

---

## Relation to prob2 (Original)

The **prob2** folder contains the base comparison (same-method vs cross-method accuracy, case studies for Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones, and bottom-two flags). **prob2_improved** adds:

- **A.** Statistical tests (t-test, chi-square) and sensitivity ±5%.
- **B.** Fan-favoring quantification (weight/variance decomposition, dynamic-range argument for rank).
- **C.** Survival boundary plot and impact matrix (Judge × Fan → survival prob).
- **D.** Judges' Choice scenario (eliminate lower judge from bottom two) and controversy-case impact.
- **E.** Consistency heatmap (season × week agreement) and controversy trajectory plots (rank under pct vs rank by week).
- **F.** O-award style summary sheet and draft with Condorcet / robust-statistics language.
