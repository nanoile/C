#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - 数据清理脚本
根据 Table 1 数据说明与建模建议实现：
1. N/A 读为 NaN
2. 仅分析 score > 0 的有效分数（0 表示已淘汰）
3. 按周归一化：考虑每周评委数 3 或 4，归一化到 0-1 或百分比
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# 配置
DATA_PATH = Path(__file__).resolve().parent / "2026_MCM_Problem_C_Data.csv"
OUT_CLEAN_CSV = Path(__file__).resolve().parent / "2026_MCM_Problem_C_Data_cleaned.csv"
OUT_LONG_CSV = Path(__file__).resolve().parent / "2026_MCM_Problem_C_Data_long.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    """读取 CSV，将 N/A 转为 NaN。"""
    df = pd.read_csv(path, na_values=["N/A", "NA", ""], keep_default_na=True)
    return df


def get_score_columns(df: pd.DataFrame):
    """获取所有 weekX_judgeY_score 列，并解析出 week 与 judge。"""
    pattern = re.compile(r"^week(\d+)_judge(\d+)_score$")
    cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            cols.append((c, int(m.group(1)), int(m.group(2))))
    return cols


def compute_weekly_scores(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """
    对每一行、每一周：
    - 只使用 score > 0 且非 NaN 的评委分（0 表示已淘汰，不参与统计）
    - 若该周所有评委都是 N/A，则该周为 NaN（该季未进行该周）
    - 归一化：该周总分 / (该周有效评委数 * 10)，得到 0~1 的分数
    """
    # 按 week 分组列名
    by_week = {}
    for col, week, judge in score_cols:
        by_week.setdefault(week, []).append((col, judge))

    weeks = sorted(by_week.keys())
    out = df.copy()

    # 新增列：每周原始总分、有效评委数、归一化分数（0-1）
    for w in weeks:
        cols_this_week = [c for c, _ in by_week[w]]
        # 原始分数矩阵 (行 x 评委)
        raw = out[cols_this_week].astype(float)
        # 有效分数：>0 且非 NaN
        valid = raw.where(raw > 0)
        # 该周有效评委数（每行）
        n_judges = valid.notna().sum(axis=1)
        # 该周总分（只加有效分）
        total = valid.sum(axis=1)
        # 满分 = 有效评委数 * 10
        max_score = n_judges * 10
        # 归一化到 0-1；若该周无有效分数则为 NaN
        normalized = total / max_score
        normalized = normalized.where(n_judges > 0, np.nan)

        out[f"week{w}_n_judges"] = n_judges
        out[f"week{w}_total_score"] = total
        out[f"week{w}_normalized"] = normalized
        # 百分比 0-100，便于阅读
        out[f"week{w}_pct"] = (normalized * 100).round(2)

    return out, weeks


def build_long_format(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """
    构建长表：每行一条「选手-赛季-周」的有效得分记录（仅 score > 0）。
    便于按周、按人做时序或回归。
    """
    by_week = {}
    for col, week, judge in score_cols:
        by_week.setdefault(week, []).append(col)

    key_cols = ["celebrity_name", "ballroom_partner", "season", "placement", "results"]
    key_cols = [c for c in key_cols if c in df.columns]

    rows = []
    for w, cols in sorted(by_week.items()):
        raw = df[key_cols + cols].copy()
        raw["week"] = w
        # 只统计 >0 的有效分（0=已淘汰，NaN=该评委未打分）
        raw["total_score"] = raw[cols].apply(
            lambda r: r[r > 0].sum() if (r > 0).any() else np.nan,
            axis=1
        )
        raw["n_judges"] = raw[cols].apply(
            lambda r: (r > 0).sum(),
            axis=1
        )
        # 只保留该周有有效分数的行
        mask = raw["n_judges"] > 0
        raw = raw.loc[mask].copy()
        raw["max_score"] = raw["n_judges"] * 10
        raw["normalized"] = (raw["total_score"] / raw["max_score"]).round(4)
        raw["pct"] = (raw["normalized"] * 100).round(2)
        for c in cols:
            raw = raw.drop(columns=[c])
        rows.append(raw)

    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    return long


def main():
    print("1. 读取原始数据（N/A → NaN）...")
    df = load_raw_data(DATA_PATH)
    print(f"   行数: {len(df)}, 列数: {len(df.columns)}")

    score_cols = get_score_columns(df)
    print(f"2. 识别到 {len(score_cols)} 个得分列（weekX_judgeY_score）")

    print("3. 按周计算：有效分数(>0)、评委数、总分、归一化(0-1)、百分比...")
    df_clean, weeks = compute_weekly_scores(df, score_cols)

    # 长表：仅含有效得分记录
    print("4. 构建长表（仅 score > 0 的记录）...")
    long_df = build_long_format(df_clean, score_cols)

    # 保存
    df_clean.to_csv(OUT_CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"   已保存宽表: {OUT_CLEAN_CSV}")

    if len(long_df) > 0:
        long_df.to_csv(OUT_LONG_CSV, index=False, encoding="utf-8-sig")
        print(f"   已保存长表: {OUT_LONG_CSV}")

    # 简要统计
    print("\n5. 清理规则小结")
    print("   - N/A 已读为 NaN（缺评委或该季无该周）。")
    print("   - 分数 = 0 视为已淘汰，不参与该周总分与归一化。")
    print("   - 归一化 = 该周总分 / (该周有效评委数 × 10)，范围 [0, 1]。")
    print("   - 新增列：weekX_n_judges, weekX_total_score, weekX_normalized, weekX_pct。")

    # 检查列名
    if "celebrity_homecountry/region" in df.columns:
        print("\n   注意：列名 celebrity_homecountry/region 含斜杠，代码中已按实际列名处理。")

    return df_clean, long_df


if __name__ == "__main__":
    main()
