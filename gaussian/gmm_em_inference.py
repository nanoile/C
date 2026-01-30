#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM 建模人气 α：α_i ~ GMM(π, μ, σ²)，v = softmax(α)，在淘汰约束下采样；
用 EM 从接受的 α 样本更新 GMM 参数，以提高先验信息量、提升确定性。
"""

import numpy as np
from typing import List, Tuple, Optional


def softmax(alpha: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    a = alpha - alpha.max()
    exp_a = np.exp(a)
    return exp_a / exp_a.sum()


def check_constraint_rule_a(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    if len(elim_pos) == 0:
        return True
    T = j + v
    T_elim_max = max(T[i] for i in elim_pos)
    T_surv_min = min(T[k] for k in range(len(T)) if k not in elim_pos)
    return T_elim_max <= T_surv_min + 1e-9


def rank_sum(v: np.ndarray, j: np.ndarray) -> np.ndarray:
    rj = np.argsort(np.argsort(-j)) + 1
    rv = np.argsort(np.argsort(-v)) + 1
    return rj + rv


def check_constraint_rule_b(
    v: np.ndarray, j: np.ndarray, elim_pos: List[int]
) -> bool:
    if len(elim_pos) == 0:
        return True
    rs = rank_sum(v, j)
    rs_elim_min = min(rs[i] for i in elim_pos)
    rs_surv_max = max(rs[k] for k in range(len(rs)) if k not in elim_pos)
    return rs_elim_min >= rs_surv_max - 1e-9


# ---------- GMM 采样与 EM ----------

def sample_alpha_from_gmm(
    n: int,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    从 1D GMM 采样 n 个标量 α。
    weights: (K,), means: (K,), stds: (K,)；stds > 0。
    """
    K = len(weights)
    comp = rng.choice(K, size=n, p=weights)
    alpha = np.zeros(n)
    for k in range(K):
        mask = comp == k
        alpha[mask] = rng.normal(means[k], stds[k], size=mask.sum())
    return alpha


def fit_gmm_em(
    alphas: np.ndarray,
    n_components: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对一维数据 alphas 用 EM 拟合 GMM。
    返回 weights (K,), means (K,), stds (K,)；stds 保证 >= min_std。
    """
    rng = rng or np.random.default_rng()
    alphas = np.asarray(alphas).ravel()
    N = len(alphas)
    if N < n_components:
        # 样本太少，返回单组分
        return (
            np.ones(1),
            np.array([np.mean(alphas)]),
            np.array([max(np.std(alphas), 0.1)]),
        )
    min_std = 0.05

    # 初始化：k-means 风格
    idx = rng.permutation(N)[:n_components]
    means = alphas[idx].astype(float).copy()
    weights = np.ones(n_components) / n_components
    stds = np.full(n_components, max(np.std(alphas), min_std))

    for _ in range(max_iter):
        # E-step: γ_nk = P(comp=k | α_n)
        # log N(α_n; μ_k, σ_k) = -0.5*log(2π) - log(σ_k) - (α_n-μ_k)^2/(2σ_k^2)
        log_prob = np.zeros((N, n_components))
        for k in range(n_components):
            log_prob[:, k] = (
                np.log(weights[k] + 1e-10)
                - np.log(stds[k] + 1e-10)
                - 0.5 * ((alphas - means[k]) / (stds[k] + 1e-10)) ** 2
            )
        log_prob -= log_prob.max(axis=1, keepdims=True)
        gamma = np.exp(log_prob)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M-step
        nk = gamma.sum(axis=0)
        nk = np.maximum(nk, 1e-10)
        weights_new = nk / N
        means_new = (gamma.T @ alphas) / nk
        # (N,) - (K,) -> (N,K) via broadcast; then (N,K) * gamma -> sum over N per k
        diff = alphas.reshape(-1, 1) - means_new.reshape(1, -1)
        var_k = (gamma * (diff ** 2)).sum(axis=0) / nk
        stds_new = np.sqrt(np.maximum(var_k, min_std ** 2))

        if np.abs(weights_new - weights).max() < tol and np.abs(means_new - means).max() < tol and np.abs(stds_new - stds).max() < tol:
            break
        weights, means, stds = weights_new, means_new, stds_new

    return weights, means, stds


def sample_v_via_gmm(
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
    gmm_weights: np.ndarray,
    gmm_means: np.ndarray,
    gmm_stds: np.ndarray,
    n_samples: int,
    max_attempts: int = 150_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    从 GMM 先验采样 α，v = softmax(α)，在约束下拒绝采样。
    返回 (v_samples, alpha_samples, attempts)。
    """
    rng = rng or np.random.default_rng()
    n = len(j_active)
    check = check_constraint_rule_b if rule == "rank" else check_constraint_rule_a

    v_samples = []
    alpha_samples = []
    attempts = 0
    while len(v_samples) < n_samples and attempts < max_attempts:
        alpha = sample_alpha_from_gmm(n, gmm_weights, gmm_means, gmm_stds, rng)
        v = softmax(alpha)
        if check(v, j_active, elim_pos):
            v_samples.append(v.copy())
            alpha_samples.append(alpha.copy())
        attempts += 1

    # 若接受率过低，用更平坦的 GMM 补足（单组分、大方差）
    if len(v_samples) < n_samples:
        flat_means = np.array([0.0])
        flat_stds = np.array([2.0])
        flat_weights = np.array([1.0])
        while len(v_samples) < n_samples and attempts < max_attempts * 2:
            alpha = sample_alpha_from_gmm(n, flat_weights, flat_means, flat_stds, rng)
            v = softmax(alpha)
            if check(v, j_active, elim_pos):
                v_samples.append(v.copy())
                alpha_samples.append(alpha.copy())
            attempts += 1

    return (
        np.array(v_samples[:n_samples]),
        np.array(alpha_samples[:n_samples]),
        attempts,
    )


def simulate_elimination_rule_a(v: np.ndarray, j: np.ndarray) -> int:
    T = j + v
    return int(np.argmin(T))


def simulate_elimination_rule_b(v: np.ndarray, j: np.ndarray) -> int:
    rs = rank_sum(v, j)
    return int(np.argmax(rs))


def consistency_per_week(
    samples: np.ndarray,
    j_active: np.ndarray,
    elim_pos: List[int],
    rule: str,
) -> float:
    sim_fn = simulate_elimination_rule_b if rule == "rank" else simulate_elimination_rule_a
    correct = sum(1 for v in samples if sim_fn(v, j_active) in elim_pos)
    return correct / len(samples) if len(samples) > 0 else 0.0


def certainty_per_contestant(samples: np.ndarray) -> np.ndarray:
    r = samples.max(axis=0) - samples.min(axis=0)
    return 1.0 - np.clip(r, 0, 1)
