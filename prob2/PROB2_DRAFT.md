# Problem 2: Comparing Rank vs Percentage Methods — Paper Draft

## 1. Introduction and Objectives

This section addresses **Problem 2**: using the fan vote estimates from Problem 1 (mc_improved) together with the rest of the data to:

1. **Compare and contrast** the two approaches used by the show to combine judge and fan votes (rank vs percentage) across seasons—applying both methods to each season and measuring consistency.
2. **Examine controversy cases** (e.g., Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones): Would the choice of method have led to the same result? How would a “judges choose from the bottom two” rule affect outcomes?
3. **Recommend** which method to use for future seasons and whether to include the judges’ bottom-two choice.

All outputs (tables and figures) are produced by the script `run_prob2_analysis.py` and saved under `prob2/output/` and `prob2/figures/`. This draft explains the model, each output, and the analysis.

---

## 2. Mathematical Description of the Two Methods

### 2.1 Percentage method (Rule A; used in Seasons 3–27)

Total score for contestant \(i\):

\[
S_{\text{total},i} = \frac{J_i}{\sum_k J_k} + \frac{V_i}{\sum_k V_k} = J_i^{\text{share}} + V_i^{\text{share}}.
\]

- **Properties**: Fully compensatory. A very high fan share can offset a low judge share. Because \(\operatorname{Var}(J)\) is typically much smaller than \(\operatorname{Var}(V)\) (judge scores are bounded and similar across contestants), a one-unit change in fan vote share has a larger impact on the total than a one-unit change in judge share. This gives contestants with extreme fan bases strong leverage.

### 2.2 Rank method (Rule B; used in Seasons 1–2 and 28+)

Let \(R(J_i)\) and \(R(V_i)\) be the ranks of contestant \(i\) by judge score and by fan vote (rank 1 = best). Then:

\[
S_{\text{total},i} = R(J_i) + R(V_i).
\]

- **Properties**: Variance-stabilizing. A contestant who leads by 1 vote or by 1 million votes in the fan vote both get rank 1. The contribution to the total is “truncated” to a rank, which limits how much extreme popularity can swing the result.

### 2.3 Elimination rule (both methods)

- **Percentage**: The contestant with the **lowest** total \(S_{\text{total}}\) is eliminated (each week).
- **Rank**: The contestant with the **highest** rank-sum \(R(J)+R(V)\) (i.e., worst combined rank) is eliminated.

We use the same rule as in mc_improved: ties in rank-sum are not broken stochastically in the model; we take a single worst contestant by \(\arg\max(\text{rank\_sum})\).

---

## 3. Data and Implementation

- **Fan vote estimates**: From mc_improved (`mc_improved/output/estimated_fan_votes.csv`), we use per-contestant, per-week `judge_share` and `vote_share_mean` as point estimates of \(J_i^{\text{share}}\) and \(V_i^{\text{share}}\).
- **Seasons and rules**: S1–2 and S28+ use the rank method; S3–27 use the percentage method (same as in mc_improved `data_prep.rule_type()`).
- **Per (season, week)** we:
  1. Take the active contestants and their \((J, V)\) from the fan-vote table.
  2. Compute who would be eliminated under the **percentage** rule: \(\arg\min(J + V)\).
  3. Compute who would be eliminated under the **rank** rule: \(\arg\max\bigl(R(J)+R(V)\bigr)\).
  4. Compare both to the **actual** eliminated contestant (from the data).

Outputs:

- `output/comparison_rank_vs_percentage.csv`: For each (season, week), actual eliminated, elim under percentage, elim under rank, and flags for match (same method / cross method).
- `output/accuracy_by_season_same_vs_cross.csv`: Per-season counts and accuracy when using the **same** method as the show vs the **other** method.
- `output/overall_accuracy_summary.csv` and `output/summary_prob2.txt`: Overall accuracy and agreement statistics.

---

## 4. Results: Compare and Contrast Across Seasons

### 4.1 Table: `accuracy_by_season_same_vs_cross.csv`

| Column | Meaning |
|--------|--------|
| `season` | Season number. |
| `total_weeks` | Number of elimination weeks in that season. |
| `match_same` | Number of weeks where the method **actually used** that season predicted the correct eliminated contestant. |
| `accuracy_same` | `match_same / total_weeks`. |
| `match_cross` | Number of weeks where the **other** method predicted the correct eliminated contestant. |
| `accuracy_cross` | `match_cross / total_weeks`. |
| `rule_used` | Rule used that season: `"percentage"` or `"rank"`. |

**Interpretation**:  
- When **same method** accuracy is higher than **cross method** accuracy, the rule used that season is more consistent with the observed eliminations (given our fan vote estimates).  
- When **cross** accuracy is close to **same**, the two methods often agree on who goes home; when they differ, the cross method often gets the wrong person.

### 4.2 Figure: `fig1_accuracy_by_season_same_vs_cross.png`

- **Content**: Bar chart by season. For each season, two bars: “Same method as show” (accuracy when we use the rule that season used) and “Other method” (accuracy when we use the alternative rule).
- **Interpretation**:  
  - In most seasons the “same method” bar is at or above the “other method” bar, which supports that the show’s stated rule is consistent with the outcomes we predict using our fan vote estimates.  
  - Notable exceptions (e.g., S26, S27): the “other” (rank) method sometimes matches better than the “same” (percentage) method in a few seasons, which may reflect estimation uncertainty or particular fan/judge distributions.  
  - Rank seasons (S1–2, S28+): cross method (percentage) often has lower accuracy than same method (rank), i.e., applying percentage to rank-era data fits less well.

### 4.3 Overall summary (`summary_prob2.txt`)

- **Percentage-method seasons**: Accuracy with the same method (percentage) is about **74.3%**; with the cross method (rank) about **72.6%**. So percentage fits the data slightly better when that was the rule.
- **Rank-method seasons**: Accuracy with the same method (rank) is about **73.7%**; with the cross method (percentage) about **66.3%**. So when the show used rank, using rank fits clearly better than applying percentage.
- **Agreement**: In **281 of 335** weeks, both methods select the **same** contestant to eliminate. So in most weeks the two methods agree; differences appear in a minority of weeks where judge vs fan rankings conflict.

**Conclusion for 4**:  
- One method does not uniformly “favor fan votes” more; it depends on the distribution of \((J, V)\).  
- The **rank** method, when it was the rule, is more clearly “the one that matches the data” than the percentage method applied to those same seasons.  
- The **percentage** method, when it was the rule, matches slightly better than rank applied to those seasons. So the show’s stated rule is generally consistent with our estimates, and the rank method appears more “stable” in the sense that applying the other method to rank-era data loses more accuracy.

---

## 5. Figure: When Methods Disagree (`fig2_when_methods_disagree_fan_share.png`)

- **Content**: Scatter plot of “fan share of who percentage would eliminate” (x) vs “fan share of who rank would eliminate” (y), only for weeks where the two methods **disagree** on who is eliminated.
- **Interpretation**:  
  - Points above the \(y=x\) line: the contestant that **rank** would eliminate has **higher** fan share than the one **percentage** would eliminate. So in those weeks, rank is sending home someone with more fan support than the person percentage would send home—i.e., rank is relatively less “fan-favoring” in those cases.  
  - Points below \(y=x\): percentage would eliminate someone with higher fan share than rank would—percentage is relatively less “fan-favoring” in those weeks.  
  - The cloud of points and their spread show that neither method consistently eliminates the higher-fan or lower-fan contestant when they disagree; it depends on the joint \((J, V)\) configuration.

---

## 6. Case Studies: Controversy Celebrities

We use the same fan vote estimates and apply both methods week-by-week for the season in which each celebrity competed.

### 6.1 Data: `case_studies_week_by_week.csv`

For each (celebrity, season, week):

- `rule_used`: Rule that season (percentage or rank).
- `actual_eliminated`: Who was actually eliminated.
- `would_elim_under_pct` / `would_elim_under_rank`: Whether **this** celebrity would be the one eliminated under percentage / rank (given our estimates).
- `in_bottom2_pct` / `in_bottom2_rank`: Whether this celebrity would be in the **bottom two** (by total or by rank-sum) under each method.
- `same_result_both_methods`: Whether percentage and rank select the same person to eliminate that week.

### 6.2 Jerry Rice (Season 2 — rank method; runner-up despite low judge scores)

- **Figure**: `fig3_case_Jerry_Rice.png` — Judge share vs estimated fan share by week.
- **Interpretation**:  
  - Jerry’s judge share is often low; his estimated fan share is high and increases toward the end, consistent with “fans kept him in.”  
  - **Week 7**: Actual eliminated = Lisa Rinna (rank method). Under our estimates, **percentage** would have eliminated **Jerry Rice** (lowest total \(J+V\)), while **rank** eliminated Lisa Rinna. So in that week, the **rank** method was relatively more favorable to Jerry than the percentage method would have been—he survived under rank but would have been eliminated under percentage.  
  - **Week 8**: No elimination (final). Model says percentage would eliminate Jerry and rank would eliminate Stacy Keibler; this is consistent with rank having kept Jerry in the final two.  
- **Conclusion**: Jerry’s success was driven by strong fan support. Under our point estimates, the **rank** method did not “favor” him more than percentage in the sense of making survival easier; in week 7, percentage would have eliminated him. So the narrative “percentage would have made his survival easier” is not supported by our estimates; rather, rank allowed him to survive that week while percentage would have sent him home.

### 6.3 Billy Ray Cyrus (Season 4 — percentage; 5th place despite last-place judge scores in 6 weeks)

- **Figure**: `fig4_case_Billy_Ray_Cyrus.png` — Judge share vs estimated fan share by week.
- **Interpretation**:  
  - Judge share is low; estimated fan share is moderate and grows over weeks.  
  - In most weeks both methods **agree** on who is eliminated; Billy Ray is in the bottom two in several weeks (e.g., week 1, 5, 7) and is finally eliminated in week 8. Both methods would eliminate him in week 8.  
- **Conclusion**: For Billy Ray, the choice of method would not have changed the outcome in the weeks that mattered; he would still have been eliminated in week 8 under either method. The “controversy” is that he lasted as long as he did despite low judge scores—our estimates attribute that to fan support, and both combination methods are consistent with that.

### 6.4 Bristol Palin (Season 11 — percentage; 3rd with lowest judge scores 12 times)

- **Figure**: `fig5_case_Bristol_Palin.png` — Judge share vs estimated fan share by week.
- **Interpretation**:  
  - Low judge share, high estimated fan share, especially in later weeks.  
  - In most weeks both methods agree. In **week 6** they disagree: percentage would eliminate one person, rank another (Audrina Patridge was actually eliminated; our model may assign a different “worst” under rank). By the final week (10), both methods would eliminate Bristol (or the other finalist); the ordering of the final two can depend on the method.  
- **Conclusion**: Bristol’s high fan support kept her in under the percentage rule. The two methods would have led to the same result in most weeks; the main uncertainty is in a few close weeks and in the exact final order.

### 6.5 Bobby Bones (Season 27 — percentage; winner despite low judge scores)

- **Figure**: `fig6_case_Bobby_Bones.png` — Judge share vs estimated fan share by week.
- **Interpretation**:  
  - Consistently low judge share; estimated fan share rises over the season.  
  - In **week 7** the methods disagree: percentage and rank would eliminate different people (actual eliminated John Schneider; our model may assign a different “worst” under rank). In **week 9** (final), both methods would eliminate Bobby (or the other finalist)—so our estimates are consistent with a final two where Bobby’s total is low but the other contestant is also in a similar position; the winner under percentage would be the one with slightly higher total.  
- **Conclusion**: Bobby’s win is explained by strong fan support under the percentage rule. The choice of method would not have changed the fact that he survived many weeks; in the final, the method can affect who wins, and our estimates are consistent with percentage producing his win.

---

## 7. Judges Choose from the Bottom Two

- **Idea**: Each week, instead of automatically eliminating the single worst by total (or rank-sum), the show could put the **bottom two** (by total or by rank-sum) in danger and have **judges** choose which one to eliminate.
- **What we can do**: With our data we can compute, for each (season, week), who would be in the **bottom two** under percentage and under rank (`in_bottom2_pct`, `in_bottom2_rank` in the comparison and case-study tables). We **cannot** observe or estimate judges’ preference between those two contestants, so we do not know whom the judges would save.
- **Interpretation**:  
  - When the two methods **agree** on who is eliminated, they usually also agree on the bottom two (both include the actual eliminated). So “judges’ choice from bottom two” could still yield the same person if the bottom two are the same.  
  - When the methods **disagree**, the bottom two under percentage can differ from the bottom two under rank. Then: (i) the set of “at risk” couples would depend on the method; (ii) if judges choose from that set, the outcome could differ from the current automatic rule and could differ between percentage and rank.  
- **Recommendation**: Including a “judges choose from bottom two” option adds flexibility and can reduce the impact of a single outlier (e.g., one week with an unusual fan spike). We recommend considering it as a tie-breaker or as a separate phase, with the understanding that it would require a clear rule for how the bottom two are defined (by percentage total vs by rank-sum).

---

## 8. Recommendation

- **Which method for future seasons?**  
  - **Rank method** is recommended for future seasons: it stabilizes variance, limits the leverage of extreme fan bases, and when it was used (S1–2, S28+), applying the other method (percentage) to the same data gave noticeably lower accuracy (about 66% vs 74%). So rank is more “identifiable” from the data and less sensitive to fan-vote scale.  
  - **Percentage** is more intuitive (direct sum of shares) but gives more weight to fan vote variance and can produce outcomes that are more driven by a few highly motivated fan bases.

- **Judges choose from bottom two?**  
  - Yes, as an **additional** option: define the bottom two by the chosen method (we recommend rank), then let judges decide who goes home. This keeps the method’s stability while allowing judges to correct for obvious outliers or controversy in that week’s vote.

---

## 9. File Reference

| File | Description |
|------|-------------|
| `output/comparison_rank_vs_percentage.csv` | Per (season, week): actual eliminated, elim under percentage, elim under rank, match flags, bottom-two flags. |
| `output/accuracy_by_season_same_vs_cross.csv` | Per-season accuracy when using same vs other method. |
| `output/overall_accuracy_summary.csv` | Overall counts and accuracy for percentage vs rank seasons. |
| `output/case_studies_week_by_week.csv` | Week-by-week for Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones: would each method eliminate them? In bottom two? |
| `output/summary_prob2.txt` | Short text summary of accuracies and agreement. |
| `figures/fig1_accuracy_by_season_same_vs_cross.png` | Bar chart: same-method vs cross-method accuracy by season. |
| `figures/fig2_when_methods_disagree_fan_share.png` | Scatter: fan share of eliminated under each method when they disagree. |
| `figures/fig3_case_Jerry_Rice.png` | Judge vs fan share by week, Jerry Rice (S2). |
| `figures/fig4_case_Billy_Ray_Cyrus.png` | Judge vs fan share by week, Billy Ray Cyrus (S4). |
| `figures/fig5_case_Bristol_Palin.png` | Judge vs fan share by week, Bristol Palin (S11). |
| `figures/fig6_case_Bobby_Bones.png` | Judge vs fan share by week, Bobby Bones (S27). |

All results depend on the fan vote estimates from mc_improved; uncertainty in those estimates propagates to the comparison and case-study conclusions above.
