"""Barrier Frank-Wolfe algorithm for optimization over polytopes.

Implements the Barrier Frank-Wolfe (BFW) method from arXiv:2508.03474v1,
adapted for arbitrage detection in prediction markets.

Algorithm overview
------------------
The BFW algorithm minimizes a convex objective F(μ) over a polytope M,
maintaining an iterate μ_t as a convex combination of vertices:

    For t = 0, 1, 2, ...
        1. Compute gradient ∇F(μ_t)
        2. Find descent vertex: z_t = argmin_{z ∈ vertices(M)} ⟨∇F(μ_t), z⟩
           (Linear minimization oracle — solved via LP)
        3. Compute Frank-Wolfe gap: g(μ_t) = ⟨∇F(μ_t), μ_t - z_t⟩
           If g(μ_t) < threshold → converged
        4. Add z_t to active set S_t
        5. Find optimal weights over active set:
           μ_{t+1} = argmin_{μ ∈ conv(S_t)} F(μ) + ε·barrier(μ)
        6. Adaptively shrink ε when gap is small
        7. Drop vertices with near-zero weight

The barrier term prevents weights from hitting zero, maintaining
the iterate in the relative interior of conv(S_t).

For LMSR markets, F(μ) = D_KL(μ ‖ θ) and the vertices of M
correspond to valid outcome vectors.

References
----------
- arXiv:2508.03474v1, Algorithm 1 (Barrier Frank-Wolfe)
- Braun et al. (2022), Conditional Gradients and applications to convex optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FrankWolfeResult:
    """Result of the Barrier Frank-Wolfe optimization.

    Attributes
    ----------
    x : np.ndarray
        Optimal point μ*.
    value : float
        Objective value F(μ*).
    gap : float
        Final Frank-Wolfe duality gap (upper bound on suboptimality).
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged within tolerance.
    active_vertices : list
        Final active set of vertices.
    weights : np.ndarray
        Convex combination weights for active vertices.
    """

    x: np.ndarray
    value: float
    gap: float
    iterations: int
    converged: bool
    active_vertices: List[np.ndarray] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


class FrankWolfe:
    """Barrier Frank-Wolfe solver for convex optimization over polytopes.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    epsilon_init : float
        Initial barrier parameter ε. Controls how far the iterate stays
        from the boundary of the active set.
    convergence_threshold : float
        Frank-Wolfe gap threshold for convergence.
    epsilon_shrink : float
        Factor to shrink ε when gap is small (0 < shrink < 1).
    drop_threshold : float
        Minimum vertex weight before dropping from active set.
    """

    def __init__(
        self,
        max_iter: int = 150,
        epsilon_init: float = 1e-4,
        convergence_threshold: float = 1e-6,
        epsilon_shrink: float = 0.7,
        drop_threshold: float = 1e-10,
    ):
        self.max_iter = max_iter
        self.epsilon_init = epsilon_init
        self.convergence_threshold = convergence_threshold
        self.epsilon_shrink = epsilon_shrink
        self.drop_threshold = drop_threshold

    def solve(
        self,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        vertices: Optional[np.ndarray] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        x0: Optional[np.ndarray] = None,
    ) -> FrankWolfeResult:
        """Run the Barrier Frank-Wolfe algorithm.

        Can operate in two modes:
        1. **Vertex mode**: If `vertices` is provided, the LMO picks from
           the explicit vertex list (fast for small outcome spaces).
        2. **LP mode**: If constraint matrices are provided, the LMO
           solves an LP via scipy.optimize.linprog.

        Parameters
        ----------
        objective_fn : callable
            F(μ) → float. The convex objective to minimize.
        gradient_fn : callable
            ∇F(μ) → np.ndarray. Gradient of the objective.
        vertices : optional np.ndarray of shape (k, n)
            Explicit vertices of the polytope.
        A_ub, b_ub : optional
            Inequality constraints A_ub @ x ≤ b_ub for LP mode.
        A_eq, b_eq : optional
            Equality constraints A_eq @ x = b_eq for LP mode.
        bounds : optional
            Variable bounds for LP mode.
        x0 : optional
            Initial feasible point. If None, uses first vertex or LP solution.

        Returns
        -------
        FrankWolfeResult
            Optimization result.
        """
        use_explicit_vertices = vertices is not None and len(vertices) > 0

        if use_explicit_vertices:
            vertices = np.atleast_2d(vertices)

        # Determine dimensionality
        if use_explicit_vertices:
            n = vertices.shape[1]
        elif A_eq is not None:
            n = A_eq.shape[1]
        elif A_ub is not None:
            n = A_ub.shape[1]
        else:
            raise ValueError("Must provide either vertices or constraint matrices")

        # Initialize from x0 or first vertex / LP solution
        if x0 is not None:
            mu = np.asarray(x0, dtype=np.float64).copy()
            init_vertex = mu.copy()
        elif use_explicit_vertices:
            # Start from the vertex that minimizes the objective
            best_idx = 0
            best_val = float("inf")
            for i in range(len(vertices)):
                val = objective_fn(vertices[i])
                if val < best_val:
                    best_val = val
                    best_idx = i
            mu = vertices[best_idx].copy().astype(np.float64)
            init_vertex = mu.copy()
        else:
            # Solve LP to find a feasible starting point (minimize gradient at uniform)
            c = np.ones(n)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
            if not res.success:
                raise ValueError(f"Could not find feasible starting point: {res.message}")
            mu = res.x.copy()
            init_vertex = mu.copy()

        # Active set: vertices currently in the convex combination
        active_set: List[np.ndarray] = [init_vertex]
        weights = np.array([1.0])

        epsilon = self.epsilon_init
        best_gap = float("inf")
        converged = False

        for iteration in range(self.max_iter):
            grad = gradient_fn(mu)

            # --- Linear Minimization Oracle (LMO) ---
            # Find z_t = argmin_{z ∈ M} ⟨∇F(μ_t), z⟩
            if use_explicit_vertices:
                dots = vertices @ grad
                z_idx = np.argmin(dots)
                z_t = vertices[z_idx].copy()
            else:
                res = linprog(
                    grad,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method="highs",
                )
                if not res.success:
                    logger.warning("LMO LP failed", iteration=iteration, msg=res.message)
                    break
                z_t = res.x.copy()

            # --- Frank-Wolfe gap ---
            # g(μ_t) = ⟨∇F(μ_t), μ_t - z_t⟩  (duality gap, always ≥ 0)
            gap = float(grad @ (mu - z_t))

            if gap < 0:
                # Numerical issue; gap should be non-negative for convex F
                gap = 0.0

            if gap < best_gap:
                best_gap = gap

            logger.debug(
                "fw_iteration",
                iteration=iteration,
                gap=gap,
                obj=objective_fn(mu),
                epsilon=epsilon,
                n_active=len(active_set),
            )

            # --- Convergence check ---
            if gap < self.convergence_threshold:
                converged = True
                break

            # --- Add z_t to active set if not already present ---
            is_new = True
            for v in active_set:
                if np.allclose(v, z_t, atol=1e-10):
                    is_new = False
                    break
            if is_new:
                active_set.append(z_t)
                weights = np.append(weights, 0.0)

            # --- Optimize over active set with barrier ---
            # Solve: min F(Σ w_i v_i) + ε · barrier(w)
            # where barrier(w) = -Σ ln(w_i)
            k = len(active_set)
            V = np.array(active_set)  # (k, n)

            if k == 1:
                # Trivial: mu = active_set[0]
                mu = active_set[0].copy()
                weights = np.array([1.0])
            else:
                # Optimize weights w over the simplex with log-barrier
                def _subproblem_obj(w):
                    w_safe = np.maximum(w, 1e-15)
                    point = V.T @ w_safe  # (n,)
                    barrier = -epsilon * np.sum(np.log(w_safe))
                    return objective_fn(point) + barrier

                def _subproblem_grad(w):
                    w_safe = np.maximum(w, 1e-15)
                    point = V.T @ w_safe
                    grad_f = gradient_fn(point)
                    # Chain rule: ∂/∂w_i = ∂F/∂μ · v_i - ε/w_i
                    grad_w = V @ grad_f - epsilon / w_safe
                    return grad_w

                # Constraints: Σ w_i = 1, w_i ≥ small_eps
                w_eps = 1e-12
                w_bounds = [(w_eps, 1.0)] * k
                w_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                                  "jac": lambda w: np.ones(k)}]

                # Start from current weights (extended if needed)
                w0 = weights.copy()
                if np.sum(w0) > 0:
                    w0 = np.maximum(w0, w_eps)
                    w0 /= w0.sum()
                else:
                    w0 = np.ones(k) / k

                res = minimize(
                    _subproblem_obj,
                    w0,
                    jac=_subproblem_grad,
                    method="SLSQP",
                    bounds=w_bounds,
                    constraints=w_constraints,
                    options={"maxiter": 200, "ftol": 1e-14},
                )

                weights = np.maximum(res.x, 0.0)
                weights /= weights.sum()
                mu = (V.T @ weights).ravel()

            # --- Drop near-zero weight vertices ---
            # Only drop if we have at least 2 remaining vertices
            keep = weights > self.drop_threshold
            n_keep = int(np.sum(keep))
            if n_keep < len(keep) and n_keep >= 2:
                active_set = [active_set[i] for i in range(len(active_set)) if keep[i]]
                weights = weights[keep]
                weights /= weights.sum()
                mu = (np.array(active_set).T @ weights).ravel()

            # --- Adaptive barrier shrinking ---
            # Shrink barrier over time to let weights approach boundaries
            epsilon *= self.epsilon_shrink
            epsilon = max(epsilon, 1e-15)

        return FrankWolfeResult(
            x=mu,
            value=objective_fn(mu),
            gap=best_gap,
            iterations=iteration + 1 if not converged else iteration + 1,
            converged=converged,
            active_vertices=active_set,
            weights=weights,
        )
