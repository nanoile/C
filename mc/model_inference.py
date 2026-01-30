#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潜在观众投票后验采样：约束为「淘汰者总分最低」(Rule A) 或「淘汰者排名和最高」(Rule B)。
采样方法：在单纯形上先验 + 淘汰约束下的拒绝采样；输出后验样本用于一致性 C 与确定性度量。
"""

import numpy as np
from typing import List, Tuple, Optional


def check_constraint_rule_a(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    """
    Rule A (百分比制): T_i = j_i + v_i，淘汰者 L 满足 T_L <= T_k 对所有 k。
    v, j 为当周活跃选手上的向量（同长度）；elim_pos 为淘汰者在 v/j 中的下标列表。
    """
    if len(elim_pos) == 0:
        return True
    T = j + v
    T_elim_max = max(T[i] for i in elim_pos)
    T_surv_min = min(T[k] for k in range(len(T)) if k not in elim_pos)
    return T_elim_max <= T_surv_min + 1e-9


def rank_sum(v: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Rule B: 返回每个选手的 rank(j) + rank(v)，排名 1=最好（最小）。"""
    rj = np.argsort(np.argsort(-j)) + 1  # 分数高者 rank 小
    rv = np.argsort(np.argsort(-v)) + 1
    return rj + rv


def check_constraint_rule_b(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    """
    Rule B (排名制): 淘汰者的 rank_sum 最大（即排名最差）。
    """
    if len(elim_pos) == 0:
        return True
    rs = rank_sum(v, j)
    rs_elim_min = min(rs[i] for i in elim_pos)
    rs_surv_max = max(rs[k] for k in range(len(rs)) if k not in elim_pos)
    return rs_elim_min >= rs_surv_max - 1e-9


def sample_v_rejection(
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
    n_samples: int,
    alpha_prior: Optional[np.ndarray] = None,
    max_attempts: int = 500_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """
    从 Dirichlet(alpha) 采样 v，拒绝直到满足淘汰约束，收集 n_samples 个样本。
    返回 (samples, n_accepted)，samples 形状 (n_samples, len(j_active))。
    """
    rng = rng or np.random.default_rng()
    n = len(j_active)
    alpha = alpha_prior if alpha_prior is not None else np.ones(n)
    check = check_constraint_rule_b if rule == "rank" else check_constraint_rule_a

    samples = []
    attempts = 0
    while len(samples) < n_samples and attempts < max_attempts:
        v = rng.dirichlet(alpha)
        if check(v, j_active, elim_pos):
            samples.append(v.copy())
        attempts += 1

    if len(samples) < n_samples:
        # 若接受率过低，用更平坦的先验再试一轮补足
        alpha_flat = np.ones(n) * 0.5
        while len(samples) < n_samples and attempts < max_attempts * 2:
            v = rng.dirichlet(alpha_flat)
            if check(v, j_active, elim_pos):
                samples.append(v.copy())
            attempts += 1

    return np.array(samples[:n_samples]), attempts


def simulate_elimination_rule_a(v: np.ndarray, j: np.ndarray) -> int:
    """给定 v, j，返回总分最低者的下标（模拟淘汰）。"""
    T = j + v
    return int(np.argmin(T))


def simulate_elimination_rule_b(v: np.ndarray, j: np.ndarray) -> int:
    """给定 v, j，返回 rank_sum 最大者的下标（模拟淘汰）。"""
    rs = rank_sum(v, j)
    return int(np.argmax(rs))


def consistency_per_week(
    samples: np.ndarray,
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
) -> float:
    """
    一致性：模拟淘汰与实际淘汰吻合的比例。
    elim_pos 为实际淘汰者在 active 中的下标（可能多个，取第一个为“主要淘汰”用于单淘汰周）。
    """
    sim_fn = simulate_elimination_rule_b if rule == "rank" else simulate_elimination_rule_a
    correct = 0
    for v in samples:
        pred = sim_fn(v, j_active)
        if pred in elim_pos:
            correct += 1
    return correct / len(samples) if len(samples) > 0 else 0.0


def certainty_per_contestant(samples: np.ndarray) -> np.ndarray:
    """
    确定性：Certainty_i = 1 - (max(v_i) - min(v_i))，范围 [0,1]。
    samples 形状 (n_samples, n_contestants)。
    """
    r = samples.max(axis=0) - samples.min(axis=0)
    return 1.0 - np.clip(r, 0, 1)
