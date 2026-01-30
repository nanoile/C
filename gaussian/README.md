# GMM+EM 建模人气 α 与潜在观众投票

用**高斯混合模型 (GMM)** 建模人气 α，**EM** 从约束下接受的 α 样本更新 GMM 参数；投票份额 `v = softmax(α)`，在淘汰约束下拒绝采样得到后验。

## 模型

- **人气**：当周活跃选手的 α 向量，各分量 α_i 独立同分布于 1D GMM：  
  `α_i ~ Σ_k π_k N(μ_k, σ_k)`，K=3。
- **投票份额**：`v = softmax(α)`，满足淘汰约束（Rule A/B 同 mc）。
- **EM**：每轮对所有 (season, week) 用当前 GMM 采样 α→v，拒绝采样；用接受的 α 池拟合 GMM（EM 更新 π, μ, σ）；迭代 2 轮后做最终采样并计算一致性 C 与确定性。

## 运行

```bash
cd /mnt/data/disk2/zyu/C/gaussian
python run_analysis.py
```

依赖：`../mc/data_prep`（或父目录下 `clean/2026_MCM_Problem_C_Data_cleaned.csv`）。

## 输出

- `output/consistency_by_week.csv`、`consistency_by_season.csv`、`estimated_fan_votes.csv`
- `output/summary_metrics.txt`：整体 C、确定性均值/标准差、最终 GMM 参数
- `figures/fig1_consistency_by_week.png` … `fig5_example_vote_estimates.png`

## 结果小结（本机一次运行）

- **一致性 C**：约 **0.954**，与 Dirichlet 版 (mc) 相当，约束仍高度满足。
- **确定性**：均值约 **0.26**（Dirichlet 版约 0.45）。本实现中 GMM 先验经 EM 拟合后更贴近“可行 α”的分布形态，但后验 v 的极差略增大，确定性下降；可通过收紧 GMM（如减小 σ、或对 α 做 MAP 后只报告点估计）或增加 EM 轮数/采样量再比较。

## 文件

| 文件 | 说明 |
|------|------|
| `gmm_em_inference.py` | GMM 采样、softmax、约束检查、EM 拟合、一致性/确定性 |
| `run_analysis.py` | 数据加载、warm-up GMM、EM 迭代、最终采样与出图 |
