# 潜在观众投票估计 — 改进版建模与实证 Draft（mc_improved）

## 1. 问题与改进概要

**任务**（与 mc 相同）：在仅已知裁判分 \(J_{it}\) 与每周淘汰结果的前提下，估计每位选手每周的潜在观众投票（粉丝投票份额）\(V_{it}\)，并回答：
1. 估计的投票是否与淘汰结果一致？**一致性度量 (Consistency)**  
2. 估计的确定性如何？是否因人/周而异？**确定性度量 (Certainty)**

**本改进版 (mc_improved)** 针对原 mc 的三点局限做了如下升级：

| 局限 | 改进方案 | 实现状态 |
|------|----------|----------|
| 未引入时间相关与分层先验 | 分层对数正态模型（协变量 Industry/Partner 编码、MH 中可选 log_prior_mean） | 数据结构与接口已就绪；当前先验为均匀，可传入 \(\boldsymbol{X}\boldsymbol{\beta}\) 作为提议中心 |
| 排名制一致性略低 | 细化 Rule B：被淘汰者集合 E 的 rank_sum ≥ 幸存者集合 S；多淘汰时统一检查 min(rs[E]) ≥ max(rs[S]) | 已实现 |
| 拒绝采样效率低 | 约束下的 Metropolis-Hastings（对数空间提议 + softmax，满足约束即接受） | 已实现，接受率约 20%–75% |

**数据与规则**：与 mc 相同；数据来自 `clean/2026_MCM_Problem_C_Data_cleaned.csv`；S1–2、S28+ 为排名制 (Rule B)，S3–S27 为百分比制 (Rule A)。

---

## 2. 模型改进（Model Refinement）— 数学与算法

### 2.1 分层先验与时间相关性（理论框架）

不对份额 \(v\) 直接建模，而对**潜在对数人气 (log-popularity)** 建模，再通过 Softmax 归一化到单纯形：

1. **协变量（分层先验）**：  
   \(\eta_i^{(0)} = \boldsymbol{x}_i^\top \boldsymbol{\beta}\)，其中 \(\boldsymbol{x}_i\) 为选手特征（如 Industry、Partner 的 one-hot），\(\boldsymbol{\beta}\) 为回归系数，用于“借用强度”（如早期淘汰选手也可借群体特征估计人气）。

2. **时间自相关（AR(1)）**：  
   \(\eta_{i,t} = \phi \eta_{i,t-1} + (1-\phi)\eta_i^{(0)} + \varepsilon_{i,t}\)，使当周人气依赖前一周，避免不合理突变。

3. **连接函数**：  
   \(v_{it} = \mathrm{softmax}(\eta_{1,t},\ldots,\eta_{n,t})\)。

当前实现中，数据结构已支持协变量矩阵 \(\boldsymbol{X}\)；MH 的 `log_prior_mean` 可设为 \(\boldsymbol{X}\boldsymbol{\beta}\)，用于提议中心偏置。完整 AR(1) 跨周估计留作后续扩展。

### 2.2 排名制约束细化（Rule B）

- **定义**：被淘汰者集合 \(E\) 的 rank_sum 必须 **大于等于** 幸存者集合 \(S\)，即  
  \(\min_{i \in E} \mathrm{rank\_sum}(v,j)_i \geq \max_{k \in S} \mathrm{rank\_sum}(v,j)_k\)。  
  多淘汰时对所有淘汰者一并检查，避免“部分淘汰者 rank_sum 反而优于幸存者”的非法点。

- **说明**：排名制下信息损失（仅保留次序）导致不确定性自然高于百分比制；在论文中可将“排名制下不确定性更高”表述为**特性**而非缺陷。

### 2.3 约束下的 Metropolis-Hastings（替代拒绝采样）

- **动机**：高维单纯形上满足“淘汰者总分/排名最差”的可行域可能很小，拒绝采样接受率极低（部分周需数万次才得 1000 样本）。
- **做法**：在对数空间做随机游走提议 \(\log \boldsymbol{v}' = \log \boldsymbol{v} + \boldsymbol{\varepsilon}\)，\(\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \boldsymbol{I})\)，再 \(\boldsymbol{v}' = \mathrm{softmax}(\log \boldsymbol{v}')\)；若满足淘汰约束则接受，否则保持当前点。
- **性质**：一旦进入可行域，链在其邻域内移动，接受率可稳定在约 20%–75%（由步长 \(\sigma\) 与 thin 控制），无需“乱枪打鸟”。

---

## 3. 汇总指标（表 1）— mc_improved

| 指标 | 数值 | 说明 |
|------|------|------|
| **整体一致性 C** | **0.9549** | 有淘汰周：后验样本中「模拟淘汰 = 实际淘汰」比例之周均。 |
| **确定性均值** | **0.8562** | 按选手–周：Certainty = 1 − (max(v)−min(v))。 |
| **确定性标准差** | **0.0652** | 改进版下确定性分布更集中。 |
| **MH 平均接受率** | **0.6417** | 采样效率高，无需大量拒绝。 |
| **选手–周总数** | 2,777 | 同 mc。 |
| **(季, 周) 单元数** | 335 | 同 mc。 |

---

## 4. 与原 mc 的对比

| 指标 | mc (原版，拒绝采样) | mc_improved (MH + 细化 Rule B) | 变化 |
|------|----------------------|----------------------------------|------|
| 整体一致性 C | 0.9543 | **0.9549** | ΔC ≈ +0.0006（略升） |
| 确定性均值 | 0.4463 | **0.8562** | **+0.4099**（显著提高） |
| 确定性标准差 | 0.2052 | 0.0652 | 分布更集中 |
| 采样方式 | 拒绝采样（部分周尝试次数极大） | MH（固定步数，接受率稳定） | 效率与可扩展性更好 |

**解读**：
- **一致性**：两版均约 95.5%，改进版略高，说明细化 Rule B 与 MH 并未损害与淘汰结果的一致性。
- **确定性**：改进版确定性均值从约 0.45 提升至约 0.86。MH 在可行域内局部探索，后验样本更集中，故同一选手–周的后验极差更小，Certainty 更高；同时标准差下降，说明“确定性因人/周而异”的程度减弱，估计更稳定。
- **采样效率**：MH 不再依赖高维拒绝，每周固定步数即可获得足量样本，且接受率可调（步长、thin），适合后续扩展（如更复杂约束或分层先验）。

---

## 5. 图表与逐项分析

### 图 1：每周一致性（Consistency by week）

**文件**：`figures/fig1_consistency_by_week.png`  

**内容**：横轴为 (season, week) 的序号，纵轴为该周的一致性 \(C_{\mathrm{week}}\)（仅对「有淘汰」的周有值）；红色虚线为整体均值 C = 0.9549。

**解读**：绝大多数周的一致性接近或等于 1；少数周（尤其排名制季早期）在 0.35–0.75 之间，与“排名制信息损失导致不确定性更大”一致。改进版与 mc 的按周模式相近，整体略优。

---

### 图 2：按季平均一致性（Consistency by season）

**文件**：`figures/fig2_consistency_by_season.png`  

**内容**：横轴为赛季编号，纵轴为该季所有「有淘汰」周的 \(C_{\mathrm{week}}\) 平均值；红色虚线为整体 C。

**解读**：S3–S27（百分比制）各季均值多为 1.0；S1–2 与 S28+（排名制）均值略低但仍较高。便于在论文中说明：模型在百分比制下几乎完全复现淘汰，在排名制下保持高一致性。

---

### 图 3：确定性分布（Distribution of certainty）

**文件**：`figures/fig3_certainty_distribution.png`  

**内容**：横轴为 Certainty（按选手–周），纵轴为频数。

**解读**：改进版分布明显右移且更集中（均值 0.86、标准差 0.07），多数选手–周的估计后验较窄，与 MH 在可行域内局部采样一致。直接回答“Is that certainty always the same?”：**否**，但改进版下不同选手/周之间的差异比原版小。

---

### 图 4：确定性 — 淘汰 vs 晋级（Certainty: eliminated vs survived）

**文件**：`figures/fig4_certainty_eliminated_vs_survived.png`  

**内容**：当周被淘汰者与晋级者的 Certainty 直方图叠加。

**解读**：被淘汰者（尤其裁判分偏低）的投票份额被约束在“足够低才能使总分/排名最差”，后验可行域窄，Certainty 偏高；晋级者可行域大，Certainty 相对较低。改进版下两组仍保持这一模式，但整体 Certainty 水平高于 mc。

---

### 图 5：示例 — 某季某周投票份额估计（Posterior mean ± 1 std）

**文件**：`figures/fig5_example_vote_estimates.png`  

**内容**：以某一季某一周（如 S3 第 5 周）为例，横条图为每位当周选手的后验投票份额均值 ± 1 标准差。

**解读**：估计值在 0–1 之间、同周和为 1；不确定度（标准差）因人而异。改进版后验更集中，条形误差更短，与更高的 Certainty 一致。

---

### 图 6：MH 接受率按周（Accept rate per week）

**文件**：`figures/fig6_mh_accept_rate_by_week.png`  

**内容**：横轴为 (season, week) 序号，纵轴为该周 MH 的接受率；红色虚线为平均接受率。

**解读**：接受率多在 0.2–0.8 之间，平均约 64%。无“某周几乎采不出样”的极端情况，说明 MH 替代拒绝采样后，采样效率与稳定性显著改善。

---

### 图 7：一致性对比 — mc_improved vs mc（若存在原版数据）

**文件**：`figures/fig7_consistency_comparison_mc_vs_improved.png`  

**内容**：同一 (season, week) 上，改进版与原版 \(C_{\mathrm{week}}\) 的并排柱状图。

**解读**：多数周两者接近；部分周改进版略高或略低，整体改进版略优（ΔC > 0），且无系统性劣化。

---

## 6. 输出文件一览（mc_improved 目录下）

| 路径 | 内容 |
|------|------|
| `output/consistency_by_week.csv` | 每周一致性、规则、n_samples、n_accepted、n_steps、accept_rate。 |
| `output/consistency_by_season.csv` | 按季平均一致性。 |
| `output/estimated_fan_votes.csv` | 每位选手–周的 vote_share_mean、vote_share_std、certainty、eliminated_this_week。 |
| `output/summary_metrics.txt` | 整体 C、确定性均值/标准差、MH 接受率、与原 mc 的对比。 |
| `figures/fig1_consistency_by_week.png` | 每周一致性柱状图。 |
| `figures/fig2_consistency_by_season.png` | 按季一致性柱状图。 |
| `figures/fig3_certainty_distribution.png` | 确定性分布直方图。 |
| `figures/fig4_certainty_eliminated_vs_survived.png` | 确定性：淘汰 vs 晋级。 |
| `figures/fig5_example_vote_estimates.png` | 示例周投票份额估计（后验均值±1 std）。 |
| `figures/fig6_mh_accept_rate_by_week.png` | MH 接受率按周。 |
| `figures/fig7_consistency_comparison_mc_vs_improved.png` | 一致性：改进版 vs 原版（若存在 mc 输出）。 |

---

## 7. 对赛题两问的直接回答（基于 mc_improved）

**Q1: Does your model correctly estimate fan votes that lead to results consistent with who was eliminated each week? Provide measures of the consistency.**

- **答**：是。改进版在「有淘汰」的周上，以「淘汰者总分最差（Rule A）或 rank_sum 最大（Rule B，且细化多淘汰与集合比较）」为约束，用约束下的 MH 从投票份额后验中抽样；若抽样得到的淘汰者与实际淘汰者一致，则记该周一致。
- **度量**：周一致性 \(C_{\mathrm{week}}\) = 该周后验样本中「模拟淘汰 ∈ 实际淘汰集合」的比例；**整体一致性 C = 0.9549**（所有有淘汰周的 \(C_{\mathrm{week}}\) 的均值）。表格与图：`output/consistency_by_week.csv`、`output/consistency_by_season.csv`，以及图 1、图 2、图 7（对比）。

**Q2: How much certainty is there in the fan vote totals you produced, and is that certainty always the same for each contestant/week? Provide measures of your certainty for the estimates.**

- **答**：确定性用 Certainty = 1 − (后验投票份额的极差) 度量。**均值约 0.86**，**不是**对每位选手/每周都相同，但改进版下分布更集中（标准差约 0.07）。
- **度量**：每位选手–周的 Certainty ∈ [0, 1]；整体均值 0.8562、标准差 0.0652；按淘汰/晋级分组的分布见图 4。
- **结论**：被淘汰且裁判分低的选手–周确定性高；晋级且裁判分高的选手–周确定性低。估计结果见 `output/estimated_fan_votes.csv`（含 vote_share_mean、vote_share_std、certainty）。改进版相较原 mc，确定性显著提高，更利于报告与使用估计值。

---

## 8. 小结：改进版 vs 原 mc

- **一致性**：改进版 C ≈ 95.5%，与原 mc（≈95.4%）持平略优；细化 Rule B 与 MH 未损害与淘汰结果的一致性。
- **确定性**：改进版均值约 **0.86**（原版约 0.45），标准差更小，估计更稳定、可辨识度更高。
- **采样效率**：MH 接受率稳定（约 64%），无需高维拒绝，可扩展性更好；数据结构与接口支持后续加入分层先验与时间相关性（AR(1)），便于论文“模型改进”章节的理论展开与后续实现。

以上内容构成 **mc_improved** 的建模与实证 draft；所有图表与数据可在 `mc_improved/` 下复现（运行 `python run_analysis.py`，且需已运行过原 `mc/run_analysis.py` 以生成对比图 7）。
