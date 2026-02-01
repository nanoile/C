#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2: Compare rank vs percentage methods for combining judge and fan votes.
Uses fan vote estimates from mc_improved. Outputs tables, figures, and supports
case studies (Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
MC_BASE = BASE.parent / "mc_improved"
FIG_DIR = BASE / "figures"
OUT_DIR = BASE / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add mc_improved for data_prep
sys.path.insert(0, str(MC_BASE.parent))
from mc_improved.data_prep import rule_type, build_all_seasons


def rank_sum(v: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Rule B: rank(j) + rank(v), rank 1 = best (highest score)."""
    rj = np.argsort(np.argsort(-j)) + 1
    rv = np.argsort(np.argsort(-v)) + 1
    return rj + rv


def elim_under_percentage(j: np.ndarray, v: np.ndarray) -> int:
    """Index of contestant with lowest total = j + v."""
    return int(np.argmin(j + v))


def elim_under_rank(j: np.ndarray, v: np.ndarray) -> int:
    """Index of contestant with highest rank_sum (worst)."""
    rs = rank_sum(v, j)
    return int(np.argmax(rs))


def bottom_two_percentage(j: np.ndarray, v: np.ndarray) -> list:
    """Indices of two contestants with lowest total (percentage method)."""
    t = j + v
    return np.argsort(t)[:2].tolist()


def bottom_two_rank(j: np.ndarray, v: np.ndarray) -> list:
    """Indices of two contestants with highest rank_sum (worst two)."""
    rs = rank_sum(v, j)
    return np.argsort(-rs)[:2].tolist()


def run_comparison():
    """Load fan votes, apply both methods each season/week, compare outcomes."""
    votes_path = MC_BASE / "output" / "estimated_fan_votes.csv"
    df = pd.read_csv(votes_path)
    all_seasons = build_all_seasons()

    season_rule = {d["season"]: d["rule"] for d in all_seasons}

    rows = []
    by_season_week = []

    for (season, week), grp in df.groupby(["season", "week"]):
        grp = grp.reset_index(drop=True)
        j = grp["judge_share"].values
        v = grp["vote_share_mean"].values
        names = grp["celebrity_name"].tolist()
        n = len(names)

        actual_elim = grp[grp["eliminated_this_week"]]
        actual_name = actual_elim["celebrity_name"].iloc[0] if len(actual_elim) else None
        actual_idx = None
        for i in range(len(grp)):
            if grp.iloc[i]["eliminated_this_week"]:
                actual_idx = i
                break

        rule_used = season_rule.get(season, "percentage")
        idx_pct = elim_under_percentage(j, v)
        idx_rank = elim_under_rank(j, v)
        name_pct = names[idx_pct]
        name_rank = names[idx_rank]

        match_actual = (actual_name == name_pct) or (actual_name == name_rank)
        match_pct = actual_name == name_pct
        match_rank = actual_name == name_rank

        bt_pct = bottom_two_percentage(j, v)
        bt_rank = bottom_two_rank(j, v)
        in_bottom2_pct = actual_idx in bt_pct if actual_idx is not None else False
        in_bottom2_rank = actual_idx in bt_rank if actual_idx is not None else False

        rows.append({
            "season": season,
            "week": week,
            "rule_used": rule_used,
            "actual_eliminated": actual_name,
            "elim_under_pct": name_pct,
            "elim_under_rank": name_rank,
            "match_actual": match_actual,
            "match_same_method": match_pct if rule_used == "percentage" else match_rank,
            "match_cross_method": match_pct if rule_used == "rank" else match_rank,
            "in_bottom2_pct": in_bottom2_pct,
            "in_bottom2_rank": in_bottom2_rank,
        })
        by_season_week.append({
            "season": season,
            "week": week,
            "rule_used": rule_used,
            "names": names,
            "j": j,
            "v": v,
            "actual_idx": actual_idx,
            "actual_name": actual_name,
            "idx_pct": idx_pct,
            "idx_rank": idx_rank,
            "name_pct": name_pct,
            "name_rank": name_rank,
            "bt_pct": bt_pct,
            "bt_rank": bt_rank,
        })

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "comparison_rank_vs_percentage.csv", index=False)

    # By-season accuracy: same method vs cross method
    same_method = comp.groupby("season").agg(
        total_weeks=("week", "count"),
        match_same=("match_same_method", "sum"),
    ).reset_index()
    same_method["accuracy_same"] = same_method["match_same"] / same_method["total_weeks"]

    cross = comp.groupby("season").agg(
        match_cross=("match_cross_method", "sum"),
    ).reset_index()
    same_method = same_method.merge(cross, on="season")
    same_method["accuracy_cross"] = same_method["match_cross"] / same_method["total_weeks"]
    same_method["rule_used"] = same_method["season"].map(season_rule)
    same_method.to_csv(OUT_DIR / "accuracy_by_season_same_vs_cross.csv", index=False)

    # Overall: percentage seasons vs rank seasons
    pct_seasons = comp[comp["rule_used"] == "percentage"]
    rank_seasons = comp[comp["rule_used"] == "rank"]
    overall = {
        "percentage_seasons_weeks": len(pct_seasons),
        "percentage_same_method_match": pct_seasons["match_same_method"].sum(),
        "percentage_cross_method_match": pct_seasons["match_cross_method"].sum(),
        "rank_seasons_weeks": len(rank_seasons),
        "rank_same_method_match": rank_seasons["match_same_method"].sum(),
        "rank_cross_method_match": rank_seasons["match_cross_method"].sum(),
    }
    overall["pct_accuracy_same"] = overall["percentage_same_method_match"] / max(1, overall["percentage_seasons_weeks"])
    overall["pct_accuracy_cross"] = overall["percentage_cross_method_match"] / max(1, overall["percentage_seasons_weeks"])
    overall["rank_accuracy_same"] = overall["rank_same_method_match"] / max(1, overall["rank_seasons_weeks"])
    overall["rank_accuracy_cross"] = overall["rank_cross_method_match"] / max(1, overall["rank_seasons_weeks"])
    pd.DataFrame([overall]).to_csv(OUT_DIR / "overall_accuracy_summary.csv", index=False)

    return comp, same_method, by_season_week, season_rule


def case_study_weeks(by_season_week: list, celebrity_key: str, season: int | None = None) -> list:
    """Return list of (season, week, ...) for a celebrity; key is substring of name. Optional season filter."""
    out = []
    for sw in by_season_week:
        if season is not None and sw["season"] != season:
            continue
        names = sw["names"]
        if any(celebrity_key.lower() in n.lower() for n in names):
            idx = next(i for i, n in enumerate(names) if celebrity_key.lower() in n.lower())
            out.append({
                **sw,
                "celebrity_idx": idx,
                "celebrity_name": names[idx],
                "would_elim_pct_this_week": sw["idx_pct"] == idx,
                "would_elim_rank_this_week": sw["idx_rank"] == idx,
                "in_bottom2_pct": idx in sw["bt_pct"],
                "in_bottom2_rank": idx in sw["bt_rank"],
            })
    return out


def plot_accuracy_by_season(same_method: pd.DataFrame):
    """Bar chart: accuracy same method vs cross method by season."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(same_method))
    w = 0.35
    ax.bar(x - w/2, same_method["accuracy_same"], w, label="Same method as show", color="steelblue", alpha=0.9)
    ax.bar(x + w/2, same_method["accuracy_cross"], w, label="Other method", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(same_method["season"].astype(int), fontsize=9)
    ax.set_xlabel("Season")
    ax.set_ylabel("Elimination match rate")
    ax.set_title("Accuracy: predicted elimination vs actual\nSame method = rule used that season; Other = the alternative rule")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_accuracy_by_season_same_vs_cross.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rule_fan_favor(comp: pd.DataFrame, same_method: pd.DataFrame):
    """Does one method favor fan votes more? Compare outcomes when methods disagree."""
    # When same method says A, cross method says B: who has higher fan share?
    comp["fan_share_actual"] = np.nan
    comp["fan_share_elim_pct"] = np.nan
    comp["fan_share_elim_rank"] = np.nan
    for (season, week), grp in comp.groupby(["season", "week"]):
        # We need v for actual, elim_pct, elim_rank - get from by_season_week in run
        pass
    # Simpler: by season, average fan share of eliminated (under each method) vs judge share
    # We'll do a different plot: when methods disagree, what's the fan share of who each would eliminate?
    disagree = comp[comp["elim_under_pct"] != comp["elim_under_rank"]]
    if len(disagree) == 0:
        return
    # Merge back vote shares for elim_under_pct and elim_under_rank
    votes = pd.read_csv(MC_BASE / "output" / "estimated_fan_votes.csv")
    fan_pct = []
    fan_rank = []
    judge_pct = []
    judge_rank = []
    for _, r in disagree.iterrows():
        s, w = r["season"], r["week"]
        sub = votes[(votes["season"] == s) & (votes["week"] == w)]
        n_pct = r["elim_under_pct"]
        n_rank = r["elim_under_rank"]
        row_pct = sub[sub["celebrity_name"] == n_pct]
        row_rank = sub[sub["celebrity_name"] == n_rank]
        if len(row_pct):
            fan_pct.append(row_pct["vote_share_mean"].iloc[0])
            judge_pct.append(row_pct["judge_share"].iloc[0])
        if len(row_rank):
            fan_rank.append(row_rank["vote_share_mean"].iloc[0])
            judge_rank.append(row_rank["judge_share"].iloc[0])
    if not fan_pct or not fan_rank:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fan_pct, fan_rank, alpha=0.6, s=40, c="purple", label="When methods disagree")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Fan share of who percentage would eliminate")
    ax.set_ylabel("Fan share of who rank would eliminate")
    ax.set_title("When percentage vs rank disagree on who goes home\nAbove line: rank eliminates higher-fan contestant")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_when_methods_disagree_fan_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_case_study_judge_vs_fan(case_data: list, title: str, filename: str):
    """Line plot: judge share and fan share by week for one celebrity."""
    if not case_data:
        return
    weeks = [d["week"] for d in case_data]
    j = [d["j"][d["celebrity_idx"]] for d in case_data]
    v = [d["v"][d["celebrity_idx"]] for d in case_data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(weeks, j, "o-", color="navy", label="Judge share", linewidth=2, markersize=8)
    ax.plot(weeks, v, "s-", color="green", label="Est. fan share", linewidth=2, markersize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Share")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, None)
    ax.set_xlim(min(weeks) - 0.5, max(weeks) + 0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Problem 2: Rank vs percentage comparison")
    comp, same_method, by_season_week, season_rule = run_comparison()
    print(f"  Comparison table: {OUT_DIR / 'comparison_rank_vs_percentage.csv'}")
    print(f"  Accuracy by season: {OUT_DIR / 'accuracy_by_season_same_vs_cross.csv'}")

    plot_accuracy_by_season(same_method)
    print(f"  Figure: {FIG_DIR / 'fig1_accuracy_by_season_same_vs_cross.png'}")

    plot_rule_fan_favor(comp, same_method)
    if (FIG_DIR / "fig2_when_methods_disagree_fan_share.png").exists():
        print(f"  Figure: {FIG_DIR / 'fig2_when_methods_disagree_fan_share.png'}")

    # Case studies (one season per celebrity: S2 Jerry, S4 Billy Ray, S11 Bristol, S27 Bobby)
    case_seasons = {"Jerry Rice": 2, "Billy Ray Cyrus": 4, "Bristol Palin": 11, "Bobby Bones": 27}
    for name_key, title_suffix, fname in [
        ("Jerry Rice", "Season 2 (rank method) — runner-up despite low judge scores", "fig3_case_Jerry_Rice.png"),
        ("Billy Ray Cyrus", "Season 4 (percentage) — 5th despite last-place judge scores in 6 weeks", "fig4_case_Billy_Ray_Cyrus.png"),
        ("Bristol Palin", "Season 11 (percentage) — 3rd with lowest judge scores 12 times", "fig5_case_Bristol_Palin.png"),
        ("Bobby Bones", "Season 27 (percentage) — won despite consistently low judge scores", "fig6_case_Bobby_Bones.png"),
    ]:
        season_filter = case_seasons.get(name_key)
        case = case_study_weeks(by_season_week, name_key, season=season_filter)
        if case:
            plot_case_study_judge_vs_fan(case, f"Judge vs fan share: {title_suffix}", fname)
            print(f"  Case: {fname}")

    # Case study table: would the other method have eliminated them each week?
    case_rows = []
    for name_key in ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]:
        season_filter = case_seasons.get(name_key)
        case = case_study_weeks(by_season_week, name_key, season=season_filter)
        for c in case:
            rule_used = c["rule_used"]
            case_rows.append({
                "celebrity": c["celebrity_name"],
                "season": c["season"],
                "week": c["week"],
                "rule_used": rule_used,
                "actual_eliminated": c["actual_name"],
                "would_elim_under_pct": c["would_elim_pct_this_week"],
                "would_elim_under_rank": c["would_elim_rank_this_week"],
                "in_bottom2_pct": c["in_bottom2_pct"],
                "in_bottom2_rank": c["in_bottom2_rank"],
                "same_result_both_methods": c["idx_pct"] == c["idx_rank"],
            })
    if case_rows:
        pd.DataFrame(case_rows).to_csv(OUT_DIR / "case_studies_week_by_week.csv", index=False)
        print(f"  Case studies table: {OUT_DIR / 'case_studies_week_by_week.csv'}")

    # Summary: method favor
    pct_weeks = comp[comp["rule_used"] == "percentage"]
    rank_weeks = comp[comp["rule_used"] == "rank"]
    with open(OUT_DIR / "summary_prob2.txt", "w") as f:
        f.write("Problem 2 summary\n")
        f.write(f"Percentage-method seasons: accuracy same method = {same_method[same_method['rule_used']=='percentage']['accuracy_same'].mean():.4f}\n")
        f.write(f"Percentage-method seasons: accuracy cross (rank)   = {same_method[same_method['rule_used']=='percentage']['accuracy_cross'].mean():.4f}\n")
        f.write(f"Rank-method seasons:       accuracy same method = {same_method[same_method['rule_used']=='rank']['accuracy_same'].mean():.4f}\n")
        f.write(f"Rank-method seasons:       accuracy cross (pct)   = {same_method[same_method['rule_used']=='rank']['accuracy_cross'].mean():.4f}\n")
        f.write(f"Weeks where both methods agree on who is eliminated: {(comp['elim_under_pct']==comp['elim_under_rank']).sum()} / {len(comp)}\n")
    print(f"  Summary: {OUT_DIR / 'summary_prob2.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
