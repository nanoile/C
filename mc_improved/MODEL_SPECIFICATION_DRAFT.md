# mc_improved 完整模型规格说明（论文用）

本文档给出 **mc_improved** 的完整、可复现的数学模型与算法，便于直接用于论文的「模型构建」与「推断方法」章节。记号与公式与代码一一对应。

---

## 1. 问题设定与记号

### 1.1 观测数据

- **赛季 (season)**：\(s = 1, 2, \ldots, S\)（如 \(S=34\)）。
- **周 (week)**：\(t = 1, 2, \ldots, T_s\)，表示该季第 \(t\) 周。
- **当周参赛选手**：第 \(s\) 季第 \(t\) 周有 \(n_{st}\) 名选手在赛，记其**在当周名单中的下标**为 \(i \in \{0,1,\ldots,n_{st}-1\}\)；对应的**全局选手编号**（该季内）由数据表给出。
- **裁判分（已知）**：  
  第 \(s\) 季第 \(t\) 周，选手 \(i\) 的裁判**总分**经当周归一化后得到**裁判份额**，记为  
  \[
  j_{i}^{(s,t)} \in (0,1], \quad \sum_{i=0}^{n_{st}-1} j_{i}^{(s,t)} = 1.
  \]  
  在代码中记为 `j_active`，长度为 \(n = n_{st}\)。
- **淘汰结果（已知）**：  
  第 \(s\) 季第 \(t\) 周被淘汰的选手在**当周名单中的下标**集合记为 \(E_{st} \subseteq \{0,1,\ldots,n_{st}-1\}\)；当周无淘汰时 \(E_{st} = \emptyset\)。  
  代码中记为 `elim_pos`（列表）。

**目标**：在已知 \(\{j_{i}^{(s,t)}\}\) 与 \(E_{st}\) 的前提下，估计当周每位选手的**观众投票份额**（潜在变量）\(v_{i}^{(s,t)}\)，并评估估计的**与淘汰结果的一致性**以及**估计的确定性**。

### 1.2 规则类型（由赛季决定）

- **Rule A（百分比制）**：赛季 \(s \in [3, 27]\)。当周选手 \(i\) 的**综合得分**为  
  \[
  T_i = j_i + v_i, \quad \text{（已省略 } (s,t) \text{ 上标）}
  \]  
  淘汰规则：**综合得分最低**的选手被淘汰（若有并列，按赛制 tie-breaker，本模型取“任意一名最低分者被淘汰”的约束）。
- **Rule B（排名制）**：赛季 \(s \in \{1,2\} \cup \{28,29,\ldots\}\)。  
  定义**裁判排名** \(\mathrm{rank}_J(i)\)：裁判份额 \(j\) 从高到低排序，最高为 1，次高为 2，依此类推。  
  定义**观众投票排名** \(\mathrm{rank}_V(i)\)：观众份额 \(v\) 从高到低排序，最高为 1。  
  定义**综合排名和**：  
  \[
  R_i = \mathrm{rank}_J(i) + \mathrm{rank}_V(i).
  \]  
  淘汰规则：**综合排名和最大**（即名次最差）的选手被淘汰；多淘汰时，所有被淘汰者的排名和都应“不优于”所有幸存者（见下节约束形式化）。

以下将“当周”固定，省略 \((s,t)\)，记当周选手数为 \(n\)，裁判份额向量为 \(\boldsymbol{j} = (j_0,\ldots,j_{n-1})^\top\)，观众投票份额向量为 \(\boldsymbol{v} = (v_0,\ldots,v_{n-1})^\top\)，淘汰者下标集合为 \(E\)。

---

## 2. 潜在变量与状态空间

### 2.1 观众投票份额

- **潜在变量**：\(\boldsymbol{v} = (v_0, v_1, \ldots, v_{n-1})^\top\) 表示当周 \(n\) 名选手的**观众投票份额**（占当周总观众票的比例）。
- **约束**：
  \[
  v_i > 0, \quad \sum_{i=0}^{n-1} v_i = 1.
  \]
  即 \(\boldsymbol{v}\) 落在 **\((n-1)\)-维单纯形** \(\mathcal{S}_{n}\) 上（开单纯形，排除边界 0）。

### 2.2 综合得分与排名（由规则决定）

- **Rule A**：综合得分  
  \[
  T_i = j_i + v_i.
  \]
- **Rule B**：  
  - 裁判排名：\(\mathrm{rank}_J(i) = 1 + \#\{k : j_k > j_i\}\)（分数越高排名数字越小）；  
  - 观众排名：\(\mathrm{rank}_V(i) = 1 + \#\{k : v_k > v_i\}\)；  
  - 综合排名和：\(R_i = \mathrm{rank}_J(i) + \mathrm{rank}_V(i)\)。  
  代码中 `rank_sum(v, j)` 返回向量 \((R_0,\ldots,R_{n-1})^\top\)。

---

## 3. 淘汰约束（似然的信息等价）

我们**不**对淘汰结果写显式似然函数，而是用**约束**表示“与淘汰结果一致”的 \(\boldsymbol{v}\) 的集合。即：给定 \(\boldsymbol{j}\) 和 \(E\)，只有满足下列约束的 \(\boldsymbol{v}\) 才与观测一致。

### 3.1 Rule A（百分比制）

淘汰者集合 \(E\) 中的选手综合得分必须**不高于**任意幸存者。形式化：

\[
\max_{i \in E} T_i \;\leq\; \min_{k \notin E} T_k.
\]

即
\[
\max_{i \in E} (j_i + v_i) \;\leq\; \min_{k \notin E} (j_k + v_k).
\]

若 \(E = \emptyset\)（当周无淘汰），不施加约束。  
代码：`check_constraint_rule_a(v, j, elim_pos)` 检查上述不等式（右端允许 \(+\varepsilon\) 容差，\(\varepsilon=10^{-9}\)）。

### 3.2 Rule B（排名制，含多淘汰）

淘汰者集合 \(E\) 的“最差”性：**任意淘汰者的综合排名和 \(\geq\) 任意幸存者的综合排名和**。即：

\[
\min_{i \in E} R_i \;\geq\; \max_{k \notin E} R_k.
\]

- 单淘汰：\(|E|=1\)，即该淘汰者的 \(R\) 不小于所有幸存者的 \(R\)。  
- 多淘汰：所有淘汰者的 \(R\) 都不小于所有幸存者的 \(R\)，且至少有一名淘汰者的 \(R\) 达到“最差”。  

若 \(E = \emptyset\)，不施加约束。  
代码：`check_constraint_rule_b(v, j, elim_pos)` 检查 \(\min_{i \in E} R_i \geq \max_{k \notin E} R_k - \varepsilon\)。

### 3.3 可行域

记满足上述约束且 \(\boldsymbol{v} \in \mathcal{S}_n\) 的集合为 \(\mathcal{V}(\boldsymbol{j}, E; \mathrm{rule})\)。  
后验推断仅在 \(\mathcal{V}\) 内进行；即**约束与淘汰结果等价于将支持集限制在 \(\mathcal{V}\)**。

---

## 4. 先验分布

### 4.1 基准先验（当前实现）

在单纯形上采用**无信息先验**：在 \(\mathcal{S}_n\) 上均匀（或等价地，取 Dirichlet 形状参数 \(\boldsymbol{\alpha} = \mathbf{1}\)）。  
即先验密度（在 \(\mathcal{S}_n\) 上）为  
\[
p(\boldsymbol{v}) \propto 1, \quad \boldsymbol{v} \in \mathcal{S}_n.
\]  
实现上，**初始化**时用 Dirichlet(1) 的拒绝采样得到一个可行点；MH 提议不显式依赖先验密度（见下），等价于在可行域内均匀探索。

### 4.2 分层先验（扩展，用于论文模型讨论）

为便于“借用强度”和人气平滑，可对**潜在对数人气**建模，再映射到单纯形：

1. **潜在变量**：\(\boldsymbol{\eta} = (\eta_0,\ldots,\eta_{n-1})^\top \in \mathbb{R}^n\)，表示对数人气（log-popularity）。
2. **连接函数**：  
   \[
   v_i = \frac{e^{\eta_i}}{\sum_{k=0}^{n-1} e^{\eta_k}} = \mathrm{softmax}(\boldsymbol{\eta})_i.
   \]
3. **分层先验均值**：设选手 \(i\) 有协变量向量 \(\boldsymbol{x}_i\)（如 Industry、Partner 的 one-hot），则  
   \[
   \eta_i^{(0)} = \boldsymbol{x}_i^\top \boldsymbol{\beta}, \quad \boldsymbol{\beta} \sim \text{prior}.
   \]
4. **时间自相关（AR(1)，跨周扩展）**：若引入周索引 \(t\)，可设  
   \[
   \eta_{i,t} = \phi \eta_{i,t-1} + (1-\phi)\eta_i^{(0)} + \varepsilon_{i,t}, \quad \varepsilon_{i,t} \sim \mathcal{N}(0, \sigma^2).
   \]

当前代码中**数据结构**已支持协变量矩阵 \(\boldsymbol{X}\)；MH 的 `log_prior_mean` 可设为 \(\boldsymbol{X}\boldsymbol{\beta}\)，用于提议中心偏置。**默认** \(\boldsymbol{\beta}=\mathbf{0}\)，即不偏置，与“单纯形上均匀”的基准先验一致。

---

## 5. 后验与目标分布

### 5.1 约束后验

给定 \(\boldsymbol{j}\)、\(E\) 和规则类型，**后验**为在可行域 \(\mathcal{V}\) 上的截断分布：

\[
p(\boldsymbol{v} \mid \boldsymbol{j}, E) \;\propto\; p(\boldsymbol{v})\,\mathbb{I}\bigl(\boldsymbol{v} \in \mathcal{V}(\boldsymbol{j}, E)\bigr).
\]

在均匀先验下，即**可行域上的均匀分布**。  
该分布无闭式，需通过采样近似。

### 5.2 推断目标

- 得到来自 \(p(\boldsymbol{v} \mid \boldsymbol{j}, E)\) 的样本 \(\boldsymbol{v}^{(1)}, \ldots, \boldsymbol{v}^{(N)}\)（如 \(N=1000\)）。
- 用样本计算：
  - 每位选手的**后验均值** \(\hat{v}_i = \frac{1}{N}\sum_{w=1}^{N} v_i^{(w)}\) 与**后验标准差**等；
  - **一致性**与**确定性**（见第 7 节）。

---

## 6. 推断算法：约束下的 Metropolis-Hastings（MH）

### 6.1 动机

在 \(\mathcal{S}_n\) 上满足淘汰约束的 \(\mathcal{V}\) 可能体积很小，**拒绝采样**（从 Dirichlet 提议，拒绝不满足约束的样本）在部分周接受率极低。  
**MH**：从某一可行点出发，在其邻域内提议新点；若新点仍在 \(\mathcal{V}\) 内则接受，否则保持当前点。这样链始终在可行域内，接受率可稳定在约 20%–75%。

### 6.2 参数化：对数空间

为保证提议点仍在单纯形上且数值稳定，在**对数空间**操作：

- **从 \(\boldsymbol{v}\) 到 \(\boldsymbol{\eta}\)**：取 \(\boldsymbol{\eta} = \log \boldsymbol{v}\)（分量取对数；若需唯一性可减去常数，softmax 不变）。代码中 `inv_softmax(v)` 实现 \(\boldsymbol{\eta}\)。
- **从 \(\boldsymbol{\eta}\) 到 \(\boldsymbol{v}\)**：\(\boldsymbol{v} = \mathrm{softmax}(\boldsymbol{\eta})\)，即 \(v_i = e^{\eta_i} / \sum_k e^{\eta_k}\)。代码中 `softmax(eta)`。

### 6.3 初始化

用**拒绝采样**在 \(\mathcal{V}\) 内取一点作为链的起点：

1. 重复：从 Dirichlet(1) 抽取 \(\boldsymbol{v}\)，若 \(\boldsymbol{v} \in \mathcal{V}\) 则停止；否则重试（上限约 \(10^5\) 次；若仍失败则用 Dirichlet(0.5) 再试）。
2. 将该 \(\boldsymbol{v}\) 记为 \(\boldsymbol{v}^{(0)}\)，并令 \(\boldsymbol{\eta}^{(0)} = \mathrm{inv\_softmax}(\boldsymbol{v}^{(0)})\)。

代码：`find_one_valid_sample(j_active, elim_pos, rule, ...)`。

### 6.4 提议与接受（单步）

设当前状态为 \(\boldsymbol{v}^{(\mathrm{curr})}\)，对应 \(\boldsymbol{\eta}^{(\mathrm{curr})} = \mathrm{inv\_softmax}(\boldsymbol{v}^{(\mathrm{curr})})\)。

1. **提议**：  
   \[
   \boldsymbol{\eta}^{(\mathrm{prop})} = \boldsymbol{\eta}^{(\mathrm{curr})} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n).
   \]  
   步长 \(\sigma\) 为超参数（代码中 `step_size=0.15`）。  
   可选：将 \(\boldsymbol{\eta}^{(\mathrm{curr})}\) 替换为 \((1-\alpha)\boldsymbol{\eta}^{(\mathrm{curr})} + \alpha\,\boldsymbol{\eta}_{\mathrm{prior}}\)（如 \(\alpha=0.3\)），其中 \(\boldsymbol{\eta}_{\mathrm{prior}}\) 为分层先验均值（如 \(\boldsymbol{X}\boldsymbol{\beta}\)），以偏置提议中心。
2. **映射回单纯形**：\(\boldsymbol{v}^{(\mathrm{prop})} = \mathrm{softmax}(\boldsymbol{\eta}^{(\mathrm{prop})})\)。
3. **约束检查**：若 \(\boldsymbol{v}^{(\mathrm{prop})} \in \mathcal{V}\)，则**接受**：\(\boldsymbol{v}^{(\mathrm{curr})} \leftarrow \boldsymbol{v}^{(\mathrm{prop})}\)；否则**拒绝**（保持 \(\boldsymbol{v}^{(\mathrm{curr})}\) 不变）。

在**均匀先验**下，可行域内目标分布为常数，对称正态提议的 MH 接受率在满足约束时为 1（即“满足约束即接受”）；不满足约束时接受率为 0。

### 6.5 采样与稀释

- 重复上述 MH 步足够多次（如 \(M = N \times \mathrm{thin}\)，\(N=1000\)，\(\mathrm{thin}=2\)，则 \(M \geq 2000\)）。
- 每 \(\mathrm{thin}\) 步取一次当前 \(\boldsymbol{v}^{(\mathrm{curr})}\) 作为后验样本；共得 \(N\) 个样本。  
代码：`sample_v_mh(..., n_samples=1000, step_size=0.15, thin=2, ...)` 返回形状为 \((N, n)\) 的样本数组。

### 6.6 算法小结（伪代码）

```
输入: j_active, elim_pos, rule, n_samples, step_size, thin
1. v_current := find_one_valid_sample(j_active, elim_pos, rule)
2. eta_current := inv_softmax(v_current)
3. samples := []
4. for step = 1, 2, ... until len(samples) >= n_samples:
5.     epsilon ~ N(0, step_size^2 * I_n)
6.     eta_proposal := eta_current + epsilon
7.     v_proposal := softmax(eta_proposal)
8.     if v_proposal in V(j_active, elim_pos, rule):
9.         v_current := v_proposal; eta_current := inv_softmax(v_current)
10.    if step % thin == 0: append v_current to samples
11. 输出: samples[0:n_samples]
```

---

## 7. 输出量：一致性 与 确定性

### 7.1 一致性（Consistency）

**定义**：对“有淘汰”的周（\(E \neq \emptyset\)），**周一致性**  
\[
C_{\mathrm{week}} = \frac{1}{N} \sum_{w=1}^{N} \mathbb{I}\bigl( \text{模拟淘汰}(\boldsymbol{v}^{(w)}) \in E \bigr).
\]  
其中：
- **Rule A**：模拟淘汰 = \(\arg\min_i (j_i + v_i^{(w)})\)（总分最低者）。
- **Rule B**：模拟淘汰 = \(\arg\max_i R_i^{(w)}\)（综合排名和最差者）。

**整体一致性**：\(C = \frac{1}{|\text{有淘汰的周}|} \sum_{\text{有淘汰的周}} C_{\mathrm{week}}\)。  
代码：`consistency_per_week(samples, j_active, elim_pos, rule)` 返回 \(C_{\mathrm{week}}\)。

### 7.2 确定性（Certainty）

**定义**：对选手 \(i\)（当周），后验样本中其投票份额的**极差**为  
\[
r_i = \max_{w=1,\ldots,N} v_i^{(w)} - \min_{w=1,\ldots,N} v_i^{(w)}.
\]  
**确定性**  
\[
\mathrm{Certainty}_i = 1 - \min(r_i, 1) \;\in [0,1].
\]  
越大表示该选手当周的后验越集中，估计越“确定”。  
代码：`certainty_per_contestant(samples)` 返回长度 \(n\) 的向量。

**汇总**：对所有选手–周计算 Certainty，得到均值、标准差及按“当周淘汰 / 晋级”分组的分布（用于回答“是否每位选手/每周确定性相同”）。

---

## 8. 数据与实现对应

| 概念 | 符号 / 名称 | 代码 / 文件 |
|------|-------------|-------------|
| 当周裁判份额 | \(\boldsymbol{j}\), \(j_i\) | `j_active` |
| 当周淘汰者下标集合 | \(E\) | `elim_pos` |
| 当周选手数 | \(n\) | `len(j_active)` |
| 规则类型 | Rule A / Rule B | `rule` ∈ {"percentage", "rank"} |
| 可行域 | \(\mathcal{V}\) | `check_constraint_rule_a` / `check_constraint_rule_b` |
| 后验样本 | \(\boldsymbol{v}^{(1)}, \ldots, \boldsymbol{v}^{(N)}\) | `sample_v_mh(...)` 返回的数组 |
| 周一致性 | \(C_{\mathrm{week}}\) | `consistency_per_week(...)` |
| 选手确定性 | \(\mathrm{Certainty}_i\) | `certainty_per_contestant(samples)[i]` |
| 协变量矩阵（分层扩展） | \(\boldsymbol{X}\) | `covariate_matrix`（按当周下标取行） |
| 分层先验均值 | \(\boldsymbol{\eta}_{\mathrm{prior}} = \boldsymbol{X}\boldsymbol{\beta}\) | `log_prior_mean`（可选传入 `sample_v_mh`） |

---

## 9. 论文书写建议（可直接引用的结构）

- **2.1 问题与记号**：采用 §1–§2，定义 \(s,t,n,\boldsymbol{j},\boldsymbol{v},E,T_i,R_i\) 与 Rule A/B。
- **2.2 约束与可行域**：采用 §3，写出 \(\mathcal{V}\) 的两种形式（Rule A 与 Rule B 不等式）。
- **2.3 先验与后验**：采用 §4–§5，先验为单纯形上均匀，后验为 \(\mathcal{V}\) 上截断均匀；可选小节“分层与时间扩展”简述 §4.2 与 AR(1)。
- **2.4 推断**：采用 §6，给出 MH 在对数空间提议、softmax 回单纯形、约束接受/拒绝，以及初始化与稀释；可配算法框（§6.6）。
- **2.5 评估指标**：采用 §7，定义 \(C_{\mathrm{week}}\)、\(C\) 和 \(\mathrm{Certainty}_i\)，并说明与赛题两问的对应关系。

以上即 **mc_improved** 的完整模型规格，与当前代码一致，可直接用于论文的模型构建与推断方法部分。
