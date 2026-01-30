#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版潜在观众投票后验采样：
1. 约束：Rule A（淘汰者总分最低）/ Rule B（淘汰者 rank_sum 最大，多淘汰与平局见下）
2. 采样：约束下的 Metropolis-Hastings（替代拒绝采样），可选分层先验（对数人气 + 协变量）
3. Rule B 细化：被淘汰者集合 E 的 rank_sum 必须 >= 幸存者集合 S（即 min(rs[E]) >= max(rs[S])），
   多淘汰时对所有淘汰者一并检查。
"""

import numpy as np
from typing import List, Tuple, Optional


# ---------- 工具函数 ----------
def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax，输出和为 1 的正向量。"""
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    exp_x = np.exp(np.clip(x, -500, 500))
    return exp_x / exp_x.sum()


def inv_softmax(v: np.ndarray) -> np.ndarray:
    """v 为单纯形上的点（和为 1），返回一个对数空间向量 log(v) - c，使 softmax 还原为 v。"""
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 1e-15, 1.0)
    return np.log(v)


# ---------- 约束 ----------
def check_constraint_rule_a(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    """
    Rule A (百分比制): T_i = j_i + v_i，淘汰者 L 满足 T_L <= T_k 对所有 k。
    """
    if len(elim_pos) == 0:
        return True
    T = j + v
    T_elim_max = max(T[i] for i in elim_pos)
    T_surv_min = min(T[k] for k in range(len(T)) if k not in elim_pos)
    return T_elim_max <= T_surv_min + 1e-9


def rank_sum(v: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Rule B: 每个选手的 rank(j) + rank(v)，排名 1=最好（分数高者 rank 小）。"""
    rj = np.argsort(np.argsort(-j)) + 1
    rv = np.argsort(np.argsort(-v)) + 1
    return rj + rv


def check_constraint_rule_b(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    """
    Rule B (排名制，细化): 被淘汰者集合 E 的 rank_sum 必须 >= 幸存者集合 S。
    即：min(rank_sum[E]) >= max(rank_sum[S]) - tol。
    多淘汰时：所有淘汰者的 rank_sum 都应 >= 所有幸存者的 rank_sum（最差者出局）。
    """
    if len(elim_pos) == 0:
        return True
    rs = rank_sum(v, j)
    surv_pos = [k for k in range(len(rs)) if k not in elim_pos]
    if not surv_pos:
        return True
    rs_elim_min = min(rs[i] for i in elim_pos)
    rs_surv_max = max(rs[k] for k in surv_pos)
    return rs_elim_min >= rs_surv_max - 1e-9


def check_elimination_constraint(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int], rule: str
) -> bool:
    """统一入口：根据 rule 检查淘汰约束。"""
    if rule == "rank":
        return check_constraint_rule_b(v, j, elim_pos)
    return check_constraint_rule_a(v, j, elim_pos)


# ---------- 找一个可行点（用于 MH 初始化）----------
def find_one_valid_sample(
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
    max_attempts: int = 100_000,
    alpha_prior: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """用拒绝采样找一个满足约束的 v，用于 MH 的初始点。"""
    rng = rng or np.random.default_rng()
    n = len(j_active)
    alpha = alpha_prior if alpha_prior is not None else np.ones(n)
    check = (
        lambda v, j, e: check_constraint_rule_b(v, j, e)
        if rule == "rank"
        else check_constraint_rule_a(v, j, e)
    )
    for _ in range(max_attempts):
        v = rng.dirichlet(alpha)
        if check(v, j_active, elim_pos):
            return v.copy()
    # 更平坦先验再试
    alpha_flat = np.ones(n) * 0.5
    for _ in range(max_attempts):
        v = rng.dirichlet(alpha_flat)
        if check(v, j_active, elim_pos):
            return v.copy()
    return None


# ---------- Metropolis-Hastings 采样 ----------
def sample_v_mh(
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
    n_samples: int,
    step_size: float = 0.15,
    thin: int = 2,
    log_prior_mean: Optional[np.ndarray] = None,
    max_init_attempts: int = 50_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    约束下的 MH：在对数空间提议，softmax 得 v，满足约束则接受。
    返回 (samples, n_accepted, n_steps)，samples 形状 (n_samples, len(j_active))。
    """
    rng = rng or np.random.default_rng()
    n = len(j_active)
    v_current = find_one_valid_sample(
        j_active, elim_pos, rule, max_attempts=max_init_attempts, rng=rng
    )
    if v_current is None:
        return np.array([]).reshape(0, n), 0, 0

    check = (
        lambda v: check_constraint_rule_b(v, j_active, elim_pos)
        if rule == "rank"
        else check_constraint_rule_a(v, j_active, elim_pos)
    )
    # 可选：提议时向 log_prior_mean 偏置（分层先验）
    center = inv_softmax(v_current)
    if log_prior_mean is not None and len(log_prior_mean) == n:
        center = 0.7 * center + 0.3 * log_prior_mean

    samples = []
    n_accepted = 0
    n_steps = 0
    target_steps = max(n_samples * thin, 2000)
    max_steps = target_steps * 3
    while len(samples) < n_samples and n_steps < max_steps:
        noise = rng.normal(0, step_size, size=n)
        log_proposal = center + noise
        v_proposal = softmax(log_proposal)
        if check(v_proposal):
            v_current = v_proposal.copy()
            center = inv_softmax(v_current)
            if log_prior_mean is not None and len(log_prior_mean) == n:
                center = 0.7 * center + 0.3 * log_prior_mean
            n_accepted += 1
        n_steps += 1
        if n_steps % thin == 0:
            samples.append(v_current.copy())

    out = np.array(samples[:n_samples])
    return out, n_accepted, n_steps


def simulate_elimination_rule_a(v: np.ndarray, j: np.ndarray) -> int:
    """给定 v, j，返回总分最低者的下标。"""
    T = j + v
    return int(np.argmin(T))


def simulate_elimination_rule_b(v: np.ndarray, j: np.ndarray) -> int:
    """给定 v, j，返回 rank_sum 最大者的下标。"""
    rs = rank_sum(v, j)
    return int(np.argmax(rs))


def consistency_per_week(
    samples: np.ndarray,
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
) -> float:
    """一致性：模拟淘汰与实际淘汰吻合的比例。"""
    sim_fn = (
        simulate_elimination_rule_b if rule == "rank" else simulate_elimination_rule_a
    )
    correct = 0
    for v in samples:
        pred = sim_fn(v, j_active)
        if pred in elim_pos:
            correct += 1
    return correct / len(samples) if len(samples) > 0 else 0.0


def certainty_per_contestant(samples: np.ndarray) -> np.ndarray:
    """确定性：Certainty_i = 1 - (max(v_i) - min(v_i))。"""
    r = samples.max(axis=0) - samples.min(axis=0)
    return 1.0 - np.clip(r, 0, 1)


def hierarchical_log_prior_mean(
    X_active: np.ndarray, beta: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    由协变量 X_active (n_active, n_features) 与系数 beta 得到对数人气先验均值。
    beta 若为 None 则用零向量（无偏置）。输出用于 MH 的 log_prior_mean。
    """
    if X_active is None or len(X_active) == 0:
        return np.zeros(X_active.shape[0]) if X_active is not None else np.array([])
    n, p = X_active.shape
    if beta is None:
        return np.zeros(n)
    if len(beta) != p:
        return np.zeros(n)
    return X_active @ beta
