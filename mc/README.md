# 潜在观众投票估计 (Fan Vote Estimation)

本目录包含：数据准备、约束下投票后验采样、一致性/确定性度量与图表、以及建模 draft 报告。

## 依赖与数据

- Python 3.8+，pandas，numpy  
- 清理后数据：`../clean/2026_MCM_Problem_C_Data_cleaned.csv`（若不存在则尝试 `../2026_MCM_Problem_C_Data_cleaned.csv`）

## 运行方式

```bash
cd /mnt/data/disk2/zyu/C/mc
python run_analysis.py
```

将生成：
- `output/consistency_by_week.csv`：每周一致性、规则、样本数  
- `output/consistency_by_season.csv`：按季平均一致性  
- `output/estimated_fan_votes.csv`：每位选手–周的投票份额后验均值、标准差、Certainty  
- `output/summary_metrics.txt`：整体 C、确定性均值/标准差  
- `figures/fig1_consistency_by_week.png` … `fig5_example_vote_estimates.png`：一致性与确定性图表  

## 文件说明

| 文件 | 作用 |
|------|------|
| `data_prep.py` | 从 cleaned CSV 按 (season, week) 整理参赛者、裁判份额、淘汰者、规则类型。 |
| `model_inference.py` | Rule A/B 约束检查、Dirichlet 拒绝采样、一致性/确定性计算。 |
| `run_analysis.py` | 主流程：加载数据 → 逐周采样 → 汇总指标与绘图。 |
| `draft_fan_vote_estimation.md` | 建模与实证 draft，含每张图/表的解读及对赛题两问的回答。 |

## 赛题对应

- **一致性**：后验样本中「模拟淘汰 = 实际淘汰」的比例（周均 C ≈ 0.95）。  
- **确定性**：Certainty = 1 − (后验投票份额极差)，非处处相同；淘汰且低裁判分时高，晋级且高裁判分时低。
