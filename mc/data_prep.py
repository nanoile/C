#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为贝叶斯潜在观众投票模型准备数据。
按 (season, week) 整理：竞争者列表、裁判份额 j_it、当周淘汰者、规则类型。
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE / "clean" / "2026_MCM_Problem_C_Data_cleaned.csv"
RAW_PATH = BASE / "2026_MCM_Problem_C_Data.csv"

# 规则：S1-2, S28+ 为排名制(Rule B)；S3-27 为百分比制(Rule A)
def rule_type(season: int) -> str:
    if season <= 2 or season >= 28:
        return "rank"
    return "percentage"


def parse_elimination_week(results: str) -> int | None:
    """从 results 解析淘汰周；若未淘汰或 Withdrew 返回 None 或特殊标记。"""
    if pd.isna(results):
        return None
    s = str(results).strip()
    if s.startswith("Eliminated Week"):
        try:
            return int(s.replace("Eliminated Week", "").strip())
        except ValueError:
            return None
    if "Withdrew" in s or "Place" in s:
        return None  # 未淘汰
    return None


def load_season_data(season: int, df_clean: pd.DataFrame) -> dict:
    """
    返回单季结构：
    - contestants: list of (name, partner, industry, elim_week)
    - weeks: list of week numbers 1..T
    - week_data[t]: dict with 'idx' (contestant indices), 'j' (judge shares), 'eliminated' (indices of eliminated this week)
    """
    sub = df_clean[df_clean["season"] == season].copy()
    if len(sub) == 0:
        return {}
    # 解析淘汰周
    sub["elim_week"] = sub["results"].apply(parse_elimination_week)
    # 选手列表（按 placement 或行序）
    contestants = []
    for _, r in sub.iterrows():
        contestants.append({
            "name": r["celebrity_name"],
            "partner": r["ballroom_partner"],
            "industry": r.get("celebrity_industry", "Unknown"),
            "elim_week": r["elim_week"],
        })
    n = len(contestants)
    # 裁判总分列
    j_cols = [c for c in df_clean.columns if c.startswith("week") and c.endswith("_total_score")]
    week_nums = sorted({int(c.replace("week", "").split("_")[0]) for c in j_cols})
    week_data = {}
    for t in week_nums:
        col = f"week{t}_total_score"
        if col not in sub.columns:
            continue
        totals = sub[col].values.astype(float)
        # 只考虑当周参赛者：total > 0
        in_week = totals > 0
        if not np.any(in_week):
            continue
        j_t = np.zeros(n)
        j_t[in_week] = totals[in_week]
        s = j_t.sum()
        if s <= 0:
            continue
        j_t = j_t / s  # 裁判份额
        # 当周淘汰者：elim_week == t
        eliminated_this_week = [i for i in range(n) if sub.iloc[i]["elim_week"] == t]
        week_data[t] = {
            "idx": np.where(in_week)[0].tolist(),
            "j": j_t.copy(),
            "eliminated": eliminated_this_week,
        }
    return {
        "season": season,
        "rule": rule_type(season),
        "contestants": contestants,
        "n_contestants": n,
        "weeks": sorted(week_data.keys()),
        "week_data": week_data,
    }


def load_long_for_season_week(season: int, week: int, df_long: pd.DataFrame) -> pd.DataFrame:
    """某季某周的长表子集（用于核对）。"""
    return df_long[(df_long["season"] == season) & (df_long["week"] == week)]


def build_all_seasons(cleaned_path: Path | None = None) -> list:
    """构建所有季节的数据结构列表。"""
    path = cleaned_path or CLEAN_PATH
    if not path.exists():
        path = BASE / "2026_MCM_Problem_C_Data_cleaned.csv"
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    seasons = sorted(df["season"].unique())
    out = []
    for s in seasons:
        d = load_season_data(s, df)
        if d and d.get("weeks"):
            out.append(d)
    return out


if __name__ == "__main__":
    all_seasons = build_all_seasons()
    print(f"Loaded {len(all_seasons)} seasons")
    # 示例：季节 3
    for d in all_seasons:
        if d["season"] == 3:
            print("Season 3:", d["rule"], "weeks:", d["weeks"])
            print("Contestants:", len(d["contestants"]))
            for t in d["weeks"][:3]:
                wd = d["week_data"][t]
                print(f"  Week {t}: n={len(wd['idx'])}, elim={wd['eliminated']}")
