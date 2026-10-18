"""Exact 2-variable convex quadratic program.

The safety filter decides acceleration and a steering command, so the
projection of the nominal command onto linear inequality constraints is a
convex polygon in R^2. The minimizer of a diagonal weighted squared distance
is either the unconstrained command, its projection onto an edge, or a vertex.
Enumerating those candidates is exact up to a feasibility tolerance and does
not depend on an iterative solver.
"""

from __future__ import annotations

import numpy as np


def solve_weighted_qp_2d(
    target: np.ndarray,
    weights: np.ndarray,
    constraints: list[tuple[np.ndarray, float]],
    lo: np.ndarray,
    hi: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray | None:
    """Minimize ``sum_i w_i (u_i - target_i)^2`` subject to ``g·u >= rhs`` and bounds.

    Returns ``None`` when the feasible set is empty.
    """
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if np.any(hi < lo - 1e-9):
        return None

    rows: list[tuple[np.ndarray, float]] = []
    for i in range(2):
        eye = np.zeros(2, dtype=np.float64)
        eye[i] = 1.0
        rows.append((eye, float(lo[i])))
        rows.append((-eye, float(-hi[i])))
    for raw_g, rhs in constraints:
        g = np.asarray(raw_g, dtype=np.float64)
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-12:
            if rhs > tol:
                return None
            continue
        rows.append((g, float(rhs)))

    def feasible(u: np.ndarray) -> bool:
        if np.any(u < lo - tol) or np.any(u > hi + tol):
            return False
        for g, rhs in constraints:
            if float(np.asarray(g) @ u) < rhs - tol:
                return False
        return True

    def cost(u: np.ndarray) -> float:
        d = u - target
        return float(np.sum(weights * d * d))

    if feasible(target):
        return np.clip(target, lo, hi)

    candidates: list[np.ndarray] = []

    def consider_line(g: np.ndarray, rhs: float) -> None:
        denom = float(np.sum((g * g) / weights))
        if denom < 1e-14:
            return
        lam = (rhs - float(g @ target)) / denom
        u = target + (g / weights) * lam
        if feasible(u):
            candidates.append(u)

    for g, rhs in rows:
        consider_line(g, rhs)

    n = len(rows)
    for i in range(n):
        g1, r1 = rows[i]
        for j in range(i + 1, n):
            g2, r2 = rows[j]
            mat = np.vstack([g1, g2])
            det = float(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
            if abs(det) < 1e-12:
                continue
            try:
                u = np.linalg.solve(mat, np.array([r1, r2], dtype=np.float64))
            except np.linalg.LinAlgError:
                continue
            if feasible(u):
                candidates.append(u)

    if not candidates:
        return None
    best = min(candidates, key=cost)
    return np.clip(best, lo, hi)
