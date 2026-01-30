# 2026 MCM Problem C 数据清理说明

## 输入
- `2026_MCM_Problem_C_Data.csv`：原始赛题数据（选手信息、结果、每周评委得分）

## 运行
```bash
pip install -r requirements.txt
python data_cleaning.py
```

## 输出文件

### 1. `2026_MCM_Problem_C_Data_cleaned.csv`（宽表）
- 保留原始所有列。
- 对每一周新增 4 列：
  - `weekX_n_judges`：该周有效评委数（3 或 4，仅统计非 N/A 且分数 > 0）
  - `weekX_total_score`：该周有效得分总和（**已排除 0 分**，0 表示已淘汰）
  - `weekX_normalized`：归一化分数 = total_score / (n_judges × 10)，范围 [0, 1]
  - `weekX_pct`：百分比 = normalized × 100，便于阅读
- 若该周选手已淘汰（全为 0）或该季无该周（全为 N/A），则 `n_judges` 为 0，`total_score` 为 0，`normalized`/`pct` 为空（NaN）。

### 2. `2026_MCM_Problem_C_Data_long.csv`（长表）
- 每行一条「选手–赛季–周」的**有效得分**记录（仅包含 score > 0 的周）。
- 列：`celebrity_name`, `ballroom_partner`, `season`, `placement`, `results`, `week`, `total_score`, `n_judges`, `max_score`, `normalized`, `pct`。
- 适合做按周时序、面板回归或只分析“在场周”的模型。

## 清理规则（与 Table 1 及建模建议一致）

| 规则 | 说明 |
|------|------|
| **N/A → NaN** | 读取时 `na_values=["N/A","NA",""]`，缺评委或该季无该周为 NaN。 |
| **0 分不参与统计** | 0 表示该周已淘汰，不纳入该周总分与归一化；仅用 score > 0 的评委分。 |
| **按周归一化** | 每周满分 = 该周有效评委数 × 10（3 评委=30，4 评委=40），归一化 = 总分/满分，便于跨周、跨季比较。 |
| **评委身份** | Judge1/2/3/4 仅表示当周座位顺序，不表示同一评委，建模时按“第几号评分”使用即可。 |

## 建模使用建议
- 预测 `placement` 或 `results` 时，特征中的周得分请使用 `weekX_normalized` 或 `weekX_pct`，不要直接用原始总分（避免 3 评委周与 4 评委周不可比）。
- Season 15 为全明星赛，若做跨季模型可考虑单独标记或敏感性分析。
