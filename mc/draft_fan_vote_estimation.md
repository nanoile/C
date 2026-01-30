# 潜在观众投票估计 — 建模与实证 Draft

## 1. 问题与模型概要

**任务**：在仅已知裁判分 \(J_{it}\) 与每周淘汰结果 \(E_{it}\) 的前提下，估计每位选手每周的潜在观众投票（粉丝投票份额）\(V_{it}\)，并评估：
1. 估计的投票是否与淘汰结果一致？**一致性度量 (Consistency)**  
2. 估计的确定性如何？是否因人/周而异？**确定性度量 (Certainty)**

**方法**：将观众投票份额 \(v_{it}\) 视为潜在变量，满足：
- **约束**：淘汰者在该周的总分（或排名和）最差，即 \(T_L \leq T_k\)（Rule A 百分比制）或淘汰者 rank_sum 最大（Rule B 排名制）。
- **先验**：在单纯形上采用 Dirichlet 先验，在满足上述约束下进行**拒绝采样**得到后验样本。
- **规则**：S1–2、S28+ 使用 Rule B（排名制）；S3–S27 使用 Rule A（百分比制，\(T_{it} = j_{it} + v_{it}\)，淘汰者 \(T\) 最低）。

**数据**：`clean/2026_MCM_Problem_C_Data_cleaned.csv`，按 (season, week) 整理当周参赛者、裁判份额 \(j_{it}\)（当周总分归一化为份额）、当周淘汰者；共 34 季、335 个 (season, week) 单元，2,777 条选手–周记录。

---

## 2. 汇总指标（表 1）

| 指标 | 数值 | 说明 |
|------|------|------|
| **整体一致性 C** | **0.954** | 仅统计「有淘汰」的周：后验样本中「模拟淘汰 = 实际淘汰」的比例的周均。 |
| **确定性均值** | 0.446 | 按选手–周：Certainty = 1 − (max(v) − min(v))，后验样本内投票份额范围越窄越高。 |
| **确定性标准差** | 0.205 | 确定性在不同选手/周之间差异较大。 |
| **选手–周总数** | 2,777 | 估计了投票的 (contestant, season, week) 数。 |
| **(季, 周) 单元数** | 335 | 参与分析的周数。 |

**对表 1 的解读**：  
- 一致性 **C ≈ 95.4%** 表明在绝大多数有淘汰的周，从后验中随机抽出的投票向量所对应的「被淘汰者」与实际淘汰者一致，说明约束条件与淘汰结果高度吻合，模型能复现淘汰逻辑。  
- 确定性均值约 **0.45** 表示后验投票份额的区间宽度中等：既非“几乎唯一解”（Certainty≈1），也非“几乎无信息”（Certainty≈0），估计具有可用的辨识度。  
- 确定性标准差约 0.21 说明**不同选手/周的确定性并不相同**：有的选手–周后验很集中（如被淘汰且裁判分很低时），有的较分散（如裁判分高且晋级时），与下文图表一致。

---

## 3. 一致性度量：是否与淘汰结果一致？

**定义**：  
对每一「有淘汰」的周，在后验样本中计算：  
\[
C_{\text{week}} = \frac{1}{N}\sum_{s=1}^{N} \mathbb{I}(\text{Simulated Elimination}_s = \text{Actual Elimination})
\]  
整体一致性 \(C\) 为所有有淘汰周的 \(C_{\text{week}}\) 的均值。

**结果**：  
- **整体 C = 0.954**（见表 1）。  
- 按季平均的一致性见 `output/consistency_by_season.csv`：S1 较低（约 0.53），S2 约 0.87，S3–S27（百分比制）多为 1.0，S28+（排名制）约 0.75–0.92。

**解读**：  
- 百分比制（S3–S27）下周均一致性普遍很高，约束 \(T_L \leq T_k\) 与淘汰结果高度一致，说明在这些季中「总分 = 裁判份额 + 投票份额」的假设与淘汰规则匹配良好。  
- 排名制（S1–2、S28+）下一致性略低但仍较高，可能与排名规则细节（如同分、多淘汰）或样本量有关；S1 样本少、规则早期形态可能不同，导致 C 最低。

---

## 4. 图表与逐项分析

### 图 1：每周一致性（Consistency by week）

**文件**：`figures/fig1_consistency_by_week.png`  

**内容**：横轴为 (season, week) 的序号，纵轴为该周的一致性 \(C_{\text{week}}\)（仅对「有淘汰」的周有值）；红色虚线为整体均值 C = 0.954。

**解读**：  
- 大部分柱子接近或等于 1.0，少数周（尤其早期季或排名制季）在 0.4–0.7 之间。  
- 该图直观展示「绝大多数周的估计淘汰与真实淘汰一致」，支撑整体 C ≈ 95.4% 的结论；低一致性周可作为后续敏感性分析或规则细化的对象。

---

### 图 2：按季平均一致性（Consistency by season）

**文件**：`figures/fig2_consistency_by_season.png`  

**内容**：横轴为赛季编号 (1–34)，纵轴为该季所有「有淘汰」周的 \(C_{\text{week}}\) 的平均值；红色虚线为整体 C。

**解读**：  
- S3–S27 各季均值多为 1.0，与 Rule A（百分比制）下约束强、与淘汰结果一致相符。  
- S1、S2 与 S28–S34 均值低于 1.0，对应 Rule B（排名制）或规则过渡期，与「一致性按规则类型分化」的结论一致。  
- 便于在论文中说明：模型在百分比制下几乎完全复现淘汰，在排名制下仍有较高但非完美的一致性。

---

### 图 3：确定性分布（Distribution of certainty）

**文件**：`figures/fig3_certainty_distribution.png`  

**内容**：横轴为 Certainty = 1 − (max(v) − min(v))（按选手–周），纵轴为频数。

**解读**：  
- 分布大致呈单峰，集中在 0.2–0.6 之间，与表 1 中均值 0.45、标准差 0.21 一致。  
- 少数选手–周 Certainty 接近 0（后验很宽）或接近 1（后验很窄），说明**确定性并非处处相同**，依选手与当周约束强度变化。  
- 该图直接回答「Is that certainty always the same for each contestant/week?」：**否**，确定性随选手/周变化，图 4 进一步按淘汰/晋级分组展示差异。

---

### 图 4：确定性 — 淘汰 vs 晋级（Certainty: eliminated vs survived）

**文件**：`figures/fig4_certainty_eliminated_vs_survived.png`  

**内容**：两组的直方图叠加——当周被淘汰者的 Certainty 与当周晋级者的 Certainty。

**解读**：  
- 被淘汰者（尤其是裁判分明显偏低者）的投票份额被约束在「必须足够低才能使总分最差」，后验可行域窄，Certainty 往往较高。  
- 晋级者中，裁判分高者其投票份额只需「不低于某下界」即可，可行域跨度大，Certainty 往往较低。  
- 该图量化展示「确定性因身份（淘汰/晋级）和约束强度不同而不同」，与文中“评委分极低但未淘汰→观众投票下界被锁定→确定性高”的论述一致。

---

### 图 5：示例 — 某季某周投票份额估计（Posterior mean ± 1 std）

**文件**：`figures/fig5_example_vote_estimates.png`  

**内容**：以某一季某一周（如 S3 第 5 周）为例，横条图为每位当周选手的**后验投票份额均值 ± 1 标准差**。

**解读**：  
- 展示估计值数量级合理（份额在 0–1 之间，同周之和为 1），且不确定度（标准差）因人而异。  
- 裁判分低、被淘汰或处于淘汰边缘的选手，其投票份额后验通常更集中（条形短）；裁判分高、安全晋级的选手，后验更分散（条形长），与图 4 的确定性结论一致。  
- 该图可作为论文中「估计的粉丝投票份额示例」，说明模型给出的不是单一数字而是带不确定性的分布，且范围合理、非“超级大而无意义”。

---

## 5. 确定性度量：是否处处相同？

**定义**：  
对每位选手在当周的投票份额后验样本，计算  
\[
\text{Certainty}_i = 1 - (\max_w v_{iw} - \min_w v_{iw})
\]  
其中 \(w\) 为后验样本下标；Certainty ∈ [0, 1]，越大表示后验越集中。

**结果**：  
- 均值 0.446，标准差 0.205（表 1）；分布见图 3；按淘汰/晋级分组见图 4。  

**结论**：  
- **确定性并非对每位选手、每周都相同**。  
- 当周被淘汰、且裁判分偏低时，约束强，投票份额被压向低端，后验窄，Certainty 高；当周晋级、且裁判分高时，约束弱，投票份额可行域大，Certainty 低。  
- 因此，在报告或使用估计值时，建议同时给出 Certainty（或后验区间），以便区分高置信与低置信估计。

---

## 6. 对赛题两问的直接回答

**Q1: Does your model correctly estimate fan votes that lead to results consistent with who was eliminated each week? Provide measures of the consistency.**  

- **答**：是。模型在**有淘汰的周**上，以「淘汰者总分最差（Rule A）或 rank_sum 最大（Rule B）」为约束，从投票份额后验中抽样；若抽样得到的淘汰者与实际淘汰者一致，则记该周一致。  
- **度量**：周一致性 \(C_{\text{week}}\) = 该周后验样本中「模拟淘汰 = 实际淘汰」的比例；**整体一致性 C = 0.954**（所有有淘汰周的 \(C_{\text{week}}\) 的均值）。  
- 表格与图：`output/consistency_by_week.csv`、`output/consistency_by_season.csv`，以及图 1、图 2。

**Q2: How much certainty is there in the fan vote totals you produced, and is that certainty always the same for each contestant/week? Provide measures of your certainty for the estimates.**  

- **答**：确定性用 Certainty = 1 − (后验投票份额的极差) 度量，均值约 **0.45**，**不是**对每位选手/每周都相同。  
- **度量**：每位选手–周的 Certainty ∈ [0, 1]；整体均值 0.446、标准差 0.205；按淘汰/晋级分组的分布见图 4。  
- **结论**：被淘汰且裁判分低的选手–周确定性高；晋级且裁判分高的选手–周确定性低。估计结果见 `output/estimated_fan_votes.csv`（含 vote_share_mean, vote_share_std, certainty）。

---

## 7. 输出文件一览（均位于 `mc/` 下）

| 路径 | 内容 |
|------|------|
| `output/consistency_by_week.csv` | 每周一致性、规则类型、样本数等。 |
| `output/consistency_by_season.csv` | 按季平均一致性。 |
| `output/estimated_fan_votes.csv` | 每位选手–周的投票份额后验均值、标准差、Certainty、是否当周淘汰。 |
| `output/summary_metrics.txt` | 整体 C、确定性均值/标准差、记录数。 |
| `figures/fig1_consistency_by_week.png` | 每周一致性柱状图。 |
| `figures/fig2_consistency_by_season.png` | 按季一致性柱状图。 |
| `figures/fig3_certainty_distribution.png` | 确定性分布直方图。 |
| `figures/fig4_certainty_eliminated_vs_survived.png` | 确定性：淘汰 vs 晋级。 |
| `figures/fig5_example_vote_estimates.png` | 示例周投票份额估计（后验均值±1 std）。 |

---

## 8. 小结与局限

- **一致性**：整体 C ≈ 95.4%，模型估计的投票与淘汰结果高度一致；百分比制季（S3–S27）尤其高，排名制季略低但仍高。  
- **确定性**：均值约 0.45，随选手/周变化；淘汰且低裁判分时确定性高，晋级且高裁判分时确定性低。  
- **局限**：  
  - 未引入人气时间自相关（如 AR(1)）与分层先验（industry/partner），当前为「约束 + Dirichlet 先验」的简化版；可在此基础上扩展为完整贝叶斯分层模型。  
  - 排名制周的一致性略低，可能与规则实现细节（同分、多淘汰）有关，需结合赛制说明进一步细化约束。  
  - 拒绝采样在部分周接受率较低，若扩展至高维或更复杂约束，可考虑 MCMC 或优化辅助的采样。

以上内容构成潜在观众投票估计的**建模与实证 draft**，所有图表与数据均可在 `mc/` 下复现（运行 `python run_analysis.py`）。
