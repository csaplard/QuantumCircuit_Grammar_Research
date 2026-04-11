"""
Fisher–Rao geometry on a single multinomial simplex and Ricci scalar.

Coordinates: free probabilities θ = (p_0, …, p_{k-2}) with p_{k-1} = 1 - Σ θ
(standard chart on the open (k-1)-simplex). Fisher metric (expectation coords):

    G_{ab} = δ_{ab} / p_a + 1 / p_{k-1},   a,b = 0 … k-2.

For a row-stochastic transition matrix, each row is an independent categorical model;
the full Fisher metric on (Δ^{d-1})^r rows is block-diagonal → the scalar curvature
of the Riemannian **product** is the **sum** of scalar curvatures of the factors.

This module computes Ricci scalar R for one (k-1)-dim simplex factor (k outcomes),
then `ricci_scalar_transition_matrix` sums over rows (k = alphabet size, e.g. 7).

References: Amari (1985); standard multinomial Fisher metric in probability chart.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fisher_metric_multinomial",
    "derivative_fisher_metric_multinomial",
    "christoffel_from_metric",
    "riemann_from_christoffel_and_gamma_grad",
    "ricci_tensor_from_riemann",
    "ricci_scalar_from_metric",
    "ricci_scalar_one_simplex",
    "ricci_scalar_transition_matrix",
    "theta_from_transition_matrix",
    "fisher_metric_block_diagonal",
    "log_det_fisher_block",
    "fisher_speed_wrt_N",
]


def _p_from_theta(theta: np.ndarray) -> np.ndarray:
    """theta length k-1, returns p of length k (last component from closure)."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    p_last = 1.0 - float(theta.sum())
    return np.concatenate([theta, np.array([p_last], dtype=np.float64)])


def fisher_metric_multinomial(theta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Fisher metric G_{ab} for multinomial, a,b = 0..k-2; θ are first k-1 probabilities.
    """
    p = _p_from_theta(theta)
    p = np.maximum(p, eps)
    p = p / p.sum()
    d = len(theta)
    pk = p[-1]
    g = np.zeros((d, d), dtype=np.float64)
    inv_p = 1.0 / p[:d]
    g[np.diag_indices(d)] = inv_p
    g += 1.0 / pk
    return g


def derivative_fisher_metric_multinomial(theta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    ∂_c G_{ab} with ∂_c = ∂/∂θ^c, θ^c = p_c for c = 0..d-1.
    Returns array dg[c, a, b] of shape (d, d, d).
    """
    p = _p_from_theta(theta)
    p = np.maximum(p, eps)
    p = p / p.sum()
    d = len(theta)
    pk = p[-1]
    dg = np.zeros((d, d, d), dtype=np.float64)

    # ∂(1/p_k)/∂p_c = +(1/p_k^2)  for c in 0..d-1 (since ∂p_k/∂p_c = -1)
    d_inv_pk = 1.0 / (pk * pk)
    for c in range(d):
        dg[c, :, :] += d_inv_pk

    # ∂(δ_{ab}/p_a)/∂p_c = -δ_{ac} δ_{ab} / p_a^2
    inv_sq = 1.0 / (p[:d] * p[:d])
    for a in range(d):
        for b in range(d):
            if a == b:
                for c in range(d):
                    if c == a:
                        dg[c, a, b] -= inv_sq[a]
    return dg


def christoffel_from_metric(
    g: np.ndarray,
    dg: np.ndarray,
    inv_g: np.ndarray | None = None,
) -> np.ndarray:
    """
    Γ^m_{ij} = (1/2) g^{ml} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
    dg[c,a,b] = ∂_c g_{ab}
    Returns Gamma[m,i,j] with shape (d,d,d).
    """
    if inv_g is None:
        inv_g = np.linalg.inv(g)
    d = g.shape[0]
    gamma = np.zeros((d, d, d), dtype=np.float64)
    for m in range(d):
        for i in range(d):
            for j in range(d):
                s = 0.0
                for l in range(d):
                    s += inv_g[m, l] * (
                        dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                    )
                gamma[m, i, j] = 0.5 * s
    return gamma


def riemann_from_christoffel_and_gamma_grad(
    gamma: np.ndarray,
    d_gamma: np.ndarray,
) -> np.ndarray:
    """
    R^r_{smn} = ∂_m Γ^r_{ns} - ∂_n Γ^r_{ms} + Γ^r_{mλ} Γ^λ_{ns} - Γ^r_{nλ} Γ^λ_{ms}

    d_gamma[r,i,j,k] = ∂_k Γ^r_{ij}  (derivative w.r.t. θ^k).
    """
    d = gamma.shape[0]
    riem = np.zeros((d, d, d, d), dtype=np.float64)
    for r in range(d):
        for s in range(d):
            for m in range(d):
                for n in range(d):
                    t1 = d_gamma[r, n, s, m]  # ∂_m Γ^r_{ns}
                    t2 = d_gamma[r, m, s, n]  # ∂_n Γ^r_{ms}
                    acc = t1 - t2
                    for lam in range(d):
                        acc += gamma[r, m, lam] * gamma[lam, n, s]
                        acc -= gamma[r, n, lam] * gamma[lam, m, s]
                    riem[r, s, m, n] = acc
    return riem


def ricci_tensor_from_riemann(riem: np.ndarray, g_inv: np.ndarray) -> np.ndarray:
    """R_{μν} = R^ρ_{μρν} = sum_ρ R^ρ_{μρν} with R^ρ_{σμν} = riem[ρ,σ,μ,ν]."""
    d = riem.shape[0]
    ric = np.zeros((d, d), dtype=np.float64)
    for mu in range(d):
        for nu in range(d):
            s = 0.0
            for rho in range(d):
                s += riem[rho, mu, rho, nu]
            ric[mu, nu] = s
    return ric


def ricci_scalar_from_metric(g: np.ndarray, ric: np.ndarray) -> float:
    g_inv = np.linalg.inv(g)
    return float(np.einsum("mn,mn->", g_inv, ric))


def _finite_diff_gamma_gradient(
    theta: np.ndarray,
    eps: float,
    eps_fisher: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gamma, d_gamma[r,s,m,n] = ∂_m Γ^r_{ns})."""
    d = len(theta)

    def gamma_at(th: np.ndarray) -> np.ndarray:
        g = fisher_metric_multinomial(th, eps=eps_fisher)
        dg = derivative_fisher_metric_multinomial(th, eps=eps_fisher)
        return christoffel_from_metric(g, dg, inv_g=None)

    g0 = gamma_at(theta)
    d_gamma = np.zeros((d, d, d, d), dtype=np.float64)
    for m in range(d):
        thp = theta.copy()
        thm = theta.copy()
        thp[m] += eps
        thm[m] -= eps
        gp = gamma_at(thp)
        gm = gamma_at(thm)
        d_gamma[:, :, :, m] = (gp - gm) / (2.0 * eps)
    return g0, d_gamma


def ricci_scalar_one_simplex(
    p_row: np.ndarray,
    *,
    fd_eps: float = 1e-5,
    eps_fisher: float = 1e-10,
) -> float:
    """
    Scalar Ricci curvature at p_row (length k, sums to 1) for Fisher metric on Δ^{k-1}.
    """
    p_row = np.asarray(p_row, dtype=np.float64).reshape(-1)
    p_row = np.maximum(p_row, eps_fisher)
    p_row = p_row / p_row.sum()
    theta = p_row[:-1].copy()
    g = fisher_metric_multinomial(theta, eps=eps_fisher)
    gamma, d_gamma = _finite_diff_gamma_gradient(theta, fd_eps, eps_fisher)
    riem = riemann_from_christoffel_and_gamma_grad(gamma, d_gamma)
    ric = ricci_tensor_from_riemann(riem, np.linalg.inv(g))
    return ricci_scalar_from_metric(g, ric)


def theta_from_transition_matrix(T: np.ndarray) -> np.ndarray:
    """Stack first 6 free probabilities per row → length (k-1)*n_rows (e.g. 42 for 7×7)."""
    T = np.asarray(T, dtype=np.float64)
    return T[:, :-1].reshape(-1)


def fisher_metric_block_diagonal(T: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Block-diagonal Fisher metric for independent rows (product manifold)."""
    T = np.asarray(T, dtype=np.float64)
    n_rows, k = T.shape
    d = k - 1
    dim = n_rows * d
    g = np.zeros((dim, dim), dtype=np.float64)
    for r in range(n_rows):
        theta_r = T[r, :-1].copy()
        blk = fisher_metric_multinomial(theta_r, eps=eps)
        sl = slice(r * d, (r + 1) * d)
        g[sl, sl] = blk
    return g


def log_det_fisher_block(T: np.ndarray, eps: float = 1e-10) -> float:
    """log det G — varies with T even when Ricci scalar is (almost) constant."""
    g = fisher_metric_block_diagonal(T, eps=eps)
    sign, ld = np.linalg.slogdet(g)
    if sign <= 0:
        return float("nan")
    return float(ld)


def fisher_speed_wrt_N(
    T_prev: np.ndarray,
    T_next: np.ndarray,
    n_prev: float,
    n_next: float,
    eps: float = 1e-10,
) -> float:
    """
    ‖Δθ‖_Ḡ / |ΔN| with Ḡ the Fisher block metric at the midpoint θ = (θ_prev+θ_next)/2.
    N-dependent scalar for a discrete T(N) curve (embedding / path speed).
    """
    if n_next == n_prev:
        return float("nan")
    th0 = theta_from_transition_matrix(T_prev)
    th1 = theta_from_transition_matrix(T_next)
    dth = th1 - th0
    Tm = 0.5 * (np.asarray(T_prev) + np.asarray(T_next))
    g = fisher_metric_block_diagonal(Tm, eps=eps)
    spd = float(np.sqrt(np.clip(dth @ g @ dth, 0.0, np.inf)))
    return spd / abs(float(n_next - n_prev))


def ricci_scalar_transition_matrix(
    T: np.ndarray,
    *,
    fd_eps: float = 1e-5,
    eps_fisher: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """
    Sum of per-row Ricci scalars (product-manifold / block-diagonal Fisher).

    Returns
    -------
    total : float
    per_row : (n_rows,) array of Ricci per row
    """
    T = np.asarray(T, dtype=np.float64)
    rows = []
    for i in range(T.shape[0]):
        r = ricci_scalar_one_simplex(T[i], fd_eps=fd_eps, eps_fisher=eps_fisher)
        rows.append(r)
    per_row = np.asarray(rows, dtype=np.float64)
    return float(per_row.sum()), per_row


def _self_test() -> None:
    rng = np.random.default_rng(0)
    # uniform row
    u = np.ones(7) / 7.0
    R_u, _ = ricci_scalar_transition_matrix(np.tile(u, (7, 1)))
    print("R (uniform rows):", R_u)
    # random stochastic rows
    T = rng.random((7, 7))
    T /= T.sum(axis=1, keepdims=True)
    R_r, pr = ricci_scalar_transition_matrix(T)
    print("R (random T):", R_r, "per_row", pr)


if __name__ == "__main__":
    _self_test()
