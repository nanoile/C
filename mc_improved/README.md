# mc_improved — 改进版潜在观众投票估计

在 `mc` 基础上的三项改进：

1. **采样**：约束下的 Metropolis-Hastings（替代拒绝采样），接受率稳定（约 20%–75%）。
2. **Rule B 细化**：排名制下被淘汰者集合的 rank_sum ≥ 幸存者集合；多淘汰时统一检查。
3. **数据结构**：支持协变量（Industry/Partner）编码，便于后续分层先验与 AR(1) 时间相关性。

## 运行

```bash
# 需先有清洗数据：clean/2026_MCM_Problem_C_Data_cleaned.csv
# 若需图 7（与原 mc 对比），请先运行 mc/run_analysis.py
cd mc_improved
python run_analysis.py
```

## 输出

- `output/`：`consistency_by_week.csv`、`consistency_by_season.csv`、`estimated_fan_votes.csv`、`summary_metrics.txt`（含与原 mc 对比）。
- `figures/`：fig1–fig10（一致性、确定性、示例投票、MH 接受率、对比；**冰山图 fig8、漏斗图 fig9、山峦图 fig10**）。

## 文档

- **`VISUALIZATION_ANALYSIS.md`**：**三张亮点图**（冰山图、漏斗图、山峦图）的设计目的、读法、解读及与赛题两问的对应。
- **`MODEL_SPECIFICATION_DRAFT.md`**：**完整、详细的模型构建说明**（论文用）：记号、状态空间、淘汰约束、先验/后验、MH 算法步骤、一致性与确定性定义、与代码对应表；可直接用于论文的「模型」与「推断方法」章节。
- `draft_fan_vote_estimation.md`：建模改进概要、图表与表解读、对赛题两问的回答、与原 mc 的对比。
