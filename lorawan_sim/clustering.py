"""The four clustering algorithms compared in Correia et al. (2023, Sec. 2.2):
K-Means, MiniBatch K-Means, Bisecting K-Means (all via scikit-learn) and
Fuzzy c-Means (implemented here, since it is not in scikit-learn).

Each function takes ED coordinates and the number of gateways K, and
returns the K gateway (cluster-centroid) coordinates -- Figure 2 of the
paper: dataset D + number of clusters K -> centroid coordinates.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

ALGORITHMS = ("kmeans", "minibatch", "bisecting", "fcm")
ALGORITHMS_WITH_PSO = ALGORITHMS + ("pso",)


def kmeans_gateways(points: np.ndarray, k: int, seed: int, **_ignored) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    model.fit(points)
    return model.cluster_centers_


def minibatch_gateways(points: np.ndarray, k: int, seed: int, **_ignored) -> np.ndarray:
    model = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=seed)
    model.fit(points)
    return model.cluster_centers_


def bisecting_gateways(points: np.ndarray, k: int, seed: int, **_ignored) -> np.ndarray:
    model = BisectingKMeans(n_clusters=k, random_state=seed)
    model.fit(points)
    return model.cluster_centers_


def fcm_gateways(
    points: np.ndarray, k: int, seed: int, m: float = 2.0, max_iter: int = 200, tol: float = 1e-5, **_ignored
) -> np.ndarray:
    """Standard Fuzzy c-Means (Bezdek 1981).

    Minimizes sum_i sum_j u_ij^m ||x_i - c_j||^2 with soft membership
    degrees u_ij in [0, 1], sum_j u_ij = 1 (Section 2.2 of the paper).
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    u = rng.dirichlet(np.ones(k), size=n)  # random init membership, rows sum to 1

    centroids = None
    for _ in range(max_iter):
        um = u**m
        centroids = (um.T @ points) / um.sum(axis=0)[:, None]
        dist_sq = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        dist_sq = np.maximum(dist_sq, 1e-12)

        inv_ratio = dist_sq[:, :, None] / dist_sq[:, None, :]  # (n, k, k)
        u_new = 1.0 / np.sum(inv_ratio ** (1.0 / (m - 1.0)), axis=-1)

        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new

    return centroids


def _uldr_fitness_sample(
    points: np.ndarray, seed: int, sample_size: int
) -> np.ndarray:
    """A fixed random subsample of the ED population, reused across every
    PSO fitness evaluation for a given (points, seed) so all candidates
    are compared on the same "mini-network" -- otherwise per-evaluation
    sampling noise would swamp the placement signal."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if sample_size >= n:
        return points
    idx = rng.choice(n, size=sample_size, replace=False)
    return points[idx]


def pso_gateways(
    points: np.ndarray,
    k: int,
    seed: int,
    *,
    cycle_period_s: float = 600.0,
    payload_bytes: int = 20,
    n_particles: int = 12,
    n_iterations: int = 12,
    inertia: float = 0.6,
    cognitive: float = 1.4,
    social: float = 1.4,
    fitness_sample_size: int = 120,
    fitness_warmup_cycles: int = 25,
    fitness_measure_cycles: int = 3,
    **_ignored,
) -> np.ndarray:
    """Collision-and-coverage-aware gateway placement via Particle Swarm
    Optimization (innovation over Correia et al. 2023: their four
    algorithms all minimize *Euclidean* within-cluster distance -- a
    coverage-only proxy -- rather than the metric the network designer
    actually cares about). Each particle is a candidate set of k gateway
    coordinates; fitness is the simulated ULDR itself (via
    `simulator.run_simulation`, on a fixed random subsample of the ED
    population and a short warm-up, as a fast surrogate for the full
    N=500 run used for the final evaluation).

    An earlier version of this function used a cheap closed-form coverage
    proxy (population-average SF needed to reach the nearest gateway)
    instead of running the simulator. In this study's regime (R=9 km,
    n=3.66) that proxy turned out to be *degenerate*: every point in the
    area can already reach a gateway at SF7, so "coverage" was never the
    bottleneck -- collisions were -- and optimizing a metric that is flat
    almost everywhere just let the swarm wander. Directly optimizing ULDR
    fixes that at the cost of a more expensive fitness evaluation, which
    is why the search uses a small ED subsample and few iterations rather
    than the full population.

    Standard synchronous PSO (Kennedy & Eberhart 1995) with an unbounded
    velocity/position update, no learning-rate schedule -- kept simple on
    purpose so the mechanism is transparent, not because more elaborate
    variants (constriction, adaptive inertia) wouldn't help."""
    import simulator as sim  # local import: simulator.py does not import clustering.py, but keeps the module-level import graph one-directional

    rng = np.random.default_rng(seed)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = hi - lo
    fitness_points = _uldr_fitness_sample(points, seed, fitness_sample_size)

    shape = (n_particles, k, 2)
    position = lo + rng.random(shape) * span
    velocity = (rng.random(shape) - 0.5) * span * 0.1

    def cost(gw_xy: np.ndarray) -> float:
        result = sim.run_simulation(
            fitness_points, gw_xy,
            cycle_period_s=cycle_period_s,
            n_warmup_cycles=fitness_warmup_cycles,
            n_measure_cycles=fitness_measure_cycles,
            payload_bytes=payload_bytes,
            seed=seed,
        )
        return 1.0 - result.uldr  # minimize (1 - ULDR)

    personal_best = position.copy()
    personal_best_cost = np.array([cost(position[p]) for p in range(n_particles)])
    g_idx = int(np.argmin(personal_best_cost))
    global_best = personal_best[g_idx].copy()
    global_best_cost = personal_best_cost[g_idx]

    for _ in range(n_iterations):
        r1, r2 = rng.random(shape), rng.random(shape)
        velocity = (
            inertia * velocity
            + cognitive * r1 * (personal_best - position)
            + social * r2 * (global_best[None, :, :] - position)
        )
        position = np.clip(position + velocity, lo, hi)

        for p in range(n_particles):
            c = cost(position[p])
            if c < personal_best_cost[p]:
                personal_best_cost[p] = c
                personal_best[p] = position[p]
                if c < global_best_cost:
                    global_best_cost = c
                    global_best = position[p].copy()

    return global_best


GATEWAY_FUNCS = {
    "kmeans": kmeans_gateways,
    "minibatch": minibatch_gateways,
    "bisecting": bisecting_gateways,
    "fcm": fcm_gateways,
    "pso": pso_gateways,
}


def place_gateways(algorithm: str, points: np.ndarray, k: int, seed: int, **kwargs) -> np.ndarray:
    return GATEWAY_FUNCS[algorithm](points, k, seed, **kwargs)
